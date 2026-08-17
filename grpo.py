"""
GRPO (Group Relative Policy Optimization) for the Toy VLM.

The third arm of the STaR / DAgger / GRPO comparison. STaR keeps the chains the
model already got right; DAgger splices a gold correction after the first wrong
step; GRPO keeps *nothing* -- it reweights the model's own sampling
distribution by how each sampled chain scored against its group-mates.

Why the toy setting is unusually clean for this:

  * The gold rationale is a DETERMINISTIC function of (question, scene) by
    construction, so "the sampled chain equals gold" is complete oracle
    verification -- no learned reward model, no judge, no noise floor. The
    reward is the environment.
  * The vocabulary is 92 tokens, so the KL penalty is computed EXACTLY from
    full softmaxes at every generated position rather than from the usual
    single-sample k3 estimator.
  * mu = 1 (one gradient step per batch of rollouts), so the importance ratio
    pi_current / pi_rollout is identically 1 and no clipping is needed. The
    loss is the plain group-baselined REINFORCE objective.

Per training step:

  1. Draw P prompts. With probability --hard_frac the type is drawn uniformly
     from the hard bucket (ordinal, compositional_h3/h4, relative_count), else
     uniformly from all 14 types. Fresh noisy scene per prompt.
  2. Sample G rollouts per prompt at --temperature, BOTH stages sampled (unlike
     onpolicy.generate_chains_batched, whose answer stage is always greedy --
     a greedy answer carries no gradient signal of its own and would make the
     answer half of the reward unattributable to any policy action).
  3. Reward each rollout against the oracle: --reward exact (chain and answer
     both right) or --reward dense (half credit for the matched gold-step
     prefix, half for the answer).
  4. Advantage A_i = r_i - mean(group), optionally / (std + 1e-4). A group whose
     rollouts all scored the same contributes exactly zero gradient, which is
     the whole point of the group baseline: only *within-prompt* differences
     teach anything.
  5. One AdamW step on  L = L_pg + beta * KL(pi || pi_ref).

Exactness notes that make this a *pure* on-policy update:

  * The recompute forward runs in model.eval() -- dropout off -- exactly as the
    rollouts did, so the distribution the gradient is taken against is the
    distribution the tokens were drawn from, to the last bit. Running the
    recompute in train() mode (the usual habit) would silently make this
    off-policy by the dropout noise.
  * No autocast. The rollout sampler and the recompute both run in fp32, so
    the two forwards agree numerically; fp16 would also turn the ban sentinel
    into -inf and poison the entropy/KL sums with 0 * inf.
  * Only tokens the POLICY CHOSE are credited. The forced </THINK> that closes
    an unterminated span, the structural <FINAL>, and every padding position
    are writes the sampler made on the model's behalf, not actions the model
    took, and are excluded from the generated-position mask. Crediting them
    would train the model to emit tokens it never emitted.
  * log pi is taken from the SAME banned-vocabulary softmax the sampler drew
    from (specials are banned per stage), so the density that appears in the
    gradient is the density that produced the sample.

Usage:
    uv run python grpo.py --checkpoint toy_vlm_cot.pth --vocab tokenizer_vocab.json \\
        --steps 1000 --prompts 16 --group 8 --temperature 1.0 --lr 1e-5 \\
        --kl_beta 0.02 --out_dir grpo_run1 --log grpo_log.jsonl --seed 0

    uv run python grpo.py --selftest        # sampler/mask/reward/no-op checks
"""

import argparse
import copy
import hashlib
import json
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from evaluate import VLMEvaluator
from onpolicy import (MAX_GEN_LEN, _special_ids, best_device,
                      first_mismatch, generate_chains_batched, load_model)
from questions import RationaleGenerator, split_steps
from runtime import setup_runtime
from shapes import MAX_OBJECTS, MIN_OBJECTS, ShapeGenerator
from text import MAX_SEQ_LEN, NUM_IMG_TOKENS
from train_model import build_param_groups

# The residual-gap bucket this arm targets (HISTORY.md "Current state"):
# dense-rank ordinal at 62.5%, deep chains at ~84% with ~41% faithful h4, and
# relative_count's clarification behavior.
HARD_TYPES: Tuple[str, ...] = ('ordinal', 'compositional_h3',
                               'compositional_h4', 'relative_count')

# Per-position provenance of a sampled token. Only STAGE_* > 0 are policy
# actions; STAGE_NONE covers the prompt, the forced </THINK>, the structural
# <FINAL>, and padding.
STAGE_NONE, STAGE_RATIONALE, STAGE_ANSWER = 0, 1, 2

# Divisor floor for --std_norm (Dr.GRPO-style; the group std can be exactly 0).
STD_EPS = 1e-4

REWARD_MODES = ('exact', 'dense')


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


@dataclass
class Prompt:
    """One (scene, question) with its oracle answer and chain.

    `gold_rationale` is stored *as the tokenizer round-trips it*, which is the
    string a perfect policy could actually emit. In practice the round trip is
    the identity for generator output (the traces are already lowercase, and
    ' . ' survives `_preprocess_text` verbatim); normalizing anyway means the
    reward can never be unreachable because of a spelling the vocabulary
    cannot represent.
    """
    qtype: str
    image_u8: torch.Tensor          # (3,64,64) uint8
    question: str
    q_ids: List[int]
    gold_answer: str
    gold_rationale: str
    gold_steps: List[str] = field(default_factory=list)


def draw_prompt(shape_gen: ShapeGenerator, rationale_gen: RationaleGenerator,
                tok, qtype: str, max_attempts: int = 400) -> Prompt:
    """One fresh noisy scene and a committed draw of `qtype` over it.

    Noise is on, matching onpolicy._draw_sample: the perception study found
    training noise worth +21 points on count aggregation, and rollouts must be
    collected on the distribution the policy is trained for.
    """
    for _ in range(max_attempts):
        num_shapes = random.randint(MIN_OBJECTS, MAX_OBJECTS)
        image, metadata = shape_gen.generate_multi_shape_image(num_shapes, True)
        question, answer, rationale = rationale_gen.generators[qtype](metadata)
        if question is None or answer is None or not rationale:
            continue
        gold = tok.decode(tok.tokenize(rationale), skip_special_tokens=True)
        return Prompt(
            qtype=qtype,
            image_u8=torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).contiguous(),
            question=question,
            q_ids=tok.tokenize(question),
            gold_answer=answer,
            gold_rationale=gold,
            gold_steps=split_steps(gold) if gold else [],
        )
    raise RuntimeError(f"[grpo] '{qtype}' declined {max_attempts} scenes in a row")


def draw_prompt_batch(shape_gen, rationale_gen, tok, n: int, hard_frac: float,
                      hard_types: Sequence[str], all_types: Sequence[str]) -> List[Prompt]:
    prompts = []
    for _ in range(n):
        pool = hard_types if random.random() < hard_frac else all_types
        prompts.append(draw_prompt(shape_gen, rationale_gen, tok, random.choice(list(pool))))
    return prompts


# ---------------------------------------------------------------------------
# Rollout sampling
# ---------------------------------------------------------------------------
# A near-copy of onpolicy.generate_chains_batched -- same prompt layout, same
# per-stage ban lists, same two length guards, same force-close, same
# frozen-row bookkeeping -- and verified token-for-token identical to it at
# temperature 0 (see selftest check 1). It differs in exactly two ways:
#
#   * both stages sample at `temperature` (the reference greedy-decodes the
#     answer), and
#   * it returns the full padded transcript plus a per-position stage tag,
#     not just the decoded slices, because the gradient needs to know which
#     positions the policy actually chose.


@torch.no_grad()
def sample_rollouts(model, images: torch.Tensor, question_ids_list: Sequence[Sequence[int]],
                    temperature: float = 1.0, max_length: int = MAX_GEN_LEN
                    ) -> Dict[str, Any]:
    """Sample B chains in parallel.

    Returns a dict with
        ids    : (B, MAX_SEQ_LEN) long   -- the full padded transcript
        stage  : (B, MAX_SEQ_LEN) int8   -- STAGE_* per position
        slices : list of (prompt_len, rat_end, ans_start, ans_end)
    both tensors on the model's device.
    """
    model.eval()
    device = next(model.parameters()).device
    tok = model.text_processor.tokenizer

    B = len(question_ids_list)
    assert images.ndim == 4 and images.shape[1:] == (3, 64, 64), \
        f"expected images (B,3,64,64), got {tuple(images.shape)}"
    assert images.shape[0] == B, f"{images.shape[0]} images for {B} questions"
    assert images.dtype == torch.float32, f"expected float32 images, got {images.dtype}"
    images = images.to(device)

    img_block = [tok.img_start_id] + [tok.img_token_id] * NUM_IMG_TOKENS + [tok.img_end_id]
    prompts = [
        [tok.bos_token_id] + img_block + [tok.user_token_id] + list(q_ids) +
        [tok.assistant_token_id] + [tok.think_start_id]
        for q_ids in question_ids_list
    ]

    ids = torch.full((B, MAX_SEQ_LEN), tok.pad_token_id, dtype=torch.long, device=device)
    stage = torch.full((B, MAX_SEQ_LEN), STAGE_NONE, dtype=torch.int8, device=device)
    for b, prompt in enumerate(prompts):
        assert len(prompt) <= MAX_SEQ_LEN - 4, \
            f"prompt of {len(prompt)} tokens leaves no room to decode"
        ids[b, :len(prompt)] = torch.tensor(prompt, dtype=torch.long, device=device)

    prompt_lens = torch.tensor([len(p) for p in prompts], dtype=torch.long, device=device)
    lengths = prompt_lens.clone()
    rows = torch.arange(B, device=device)
    all_special = _special_ids(tok)

    def run_stage(lengths, allowed_special, cap, temp, stage_id):
        banned = torch.tensor(sorted(all_special - {allowed_special}),
                              dtype=torch.long, device=device)
        done = lengths >= cap
        for _ in range(max_length):
            if bool(done.all()):
                break
            logits = model(images, ids)                       # (B, T, V)
            pos = (lengths - 1).clamp(0, MAX_SEQ_LEN - 1)
            next_logits = logits[rows, pos, :].clone()        # (B, V)
            next_logits[:, banned] = float('-inf')
            if temp > 0:
                nxt = torch.multinomial(F.softmax(next_logits / temp, dim=-1), 1).squeeze(1)
            else:
                nxt = next_logits.argmax(dim=-1)

            write = lengths.clamp(max=MAX_SEQ_LEN - 1)
            keep = ids[rows, write]
            ids[rows, write] = torch.where(done, keep, nxt)
            # Same frozen-row discipline for the provenance tag: a done row
            # rewrites what it already had, so only live draws are tagged.
            keep_stage = stage[rows, write]
            stage[rows, write] = torch.where(done, keep_stage,
                                             torch.full_like(keep_stage, stage_id))
            lengths = lengths + (~done).long()
            done = done | (nxt == allowed_special) | (lengths >= cap)
        return lengths

    # Stage 1: rationale until </THINK>.
    lengths = run_stage(lengths, tok.think_end_id, MAX_SEQ_LEN - 4,
                        temperature, STAGE_RATIONALE)
    closed = ids[rows, (lengths - 1).clamp(min=0)] == tok.think_end_id
    rat_end = torch.where(closed, lengths - 1, lengths)
    # Force-closing an open span is the sampler's write, not the model's, so
    # `stage` is deliberately left at STAGE_NONE there. For a row that DID emit
    # </THINK> the tag was already set by run_stage and this rewrites the same
    # token, so the emitted close keeps its credit.
    ids[rows, rat_end] = tok.think_end_id
    lengths = torch.where(closed, lengths, lengths + 1)
    ids[rows, lengths] = tok.final_start_id                   # structural: untagged
    lengths = lengths + 1

    # Stage 2: answer until </FINAL>, sampled at the same temperature.
    ans_start = lengths.clone()
    lengths = run_stage(lengths, tok.final_end_id, MAX_SEQ_LEN - 1,
                        temperature, STAGE_ANSWER)
    closed = ids[rows, (lengths - 1).clamp(min=0)] == tok.final_end_id
    ans_end = torch.where(closed, lengths - 1, lengths)

    slices = list(zip(prompt_lens.cpu().tolist(), rat_end.cpu().tolist(),
                      ans_start.cpu().tolist(), ans_end.cpu().tolist()))
    return {'ids': ids, 'stage': stage, 'slices': slices}


def rollout_batch(model, prompts: Sequence[Prompt], group: int, temperature: float,
                  rollout_chunk: int, max_gen_len: int = MAX_GEN_LEN) -> Dict[str, Any]:
    """Sample `group` rollouts for each prompt; returns concatenated P*G rows.

    Row r belongs to prompt r // group, so the group structure is a reshape
    rather than a bookkeeping table.
    """
    row_prompts = [p for p in prompts for _ in range(group)]
    images = torch.stack([p.image_u8 for p in row_prompts]).to(torch.float32) / 255.0
    q_ids = [p.q_ids for p in row_prompts]

    ids_parts, stage_parts, slices = [], [], []
    for i in range(0, len(row_prompts), rollout_chunk):
        out = sample_rollouts(model, images[i:i + rollout_chunk], q_ids[i:i + rollout_chunk],
                              temperature=temperature, max_length=max_gen_len)
        ids_parts.append(out['ids'])
        stage_parts.append(out['stage'])
        slices.extend(out['slices'])

    return {
        'ids': torch.cat(ids_parts, dim=0),
        'stage': torch.cat(stage_parts, dim=0),
        'slices': slices,
        'images': images,               # CPU float32, fed back for the recompute
        'row_prompts': row_prompts,
    }


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


def answer_matches(predicted: str, gold: str) -> bool:
    return VLMEvaluator.normalize_answer(predicted) == VLMEvaluator.normalize_answer(gold)


def prefix_fraction(gold_steps: Sequence[str], model_steps: Sequence[str]) -> float:
    """Fraction of the gold chain reproduced before the first divergence."""
    if not gold_steps:
        return 1.0 if not model_steps else 0.0
    index = first_mismatch(gold_steps, model_steps)
    if index is None:
        return 1.0
    return index / len(gold_steps)


def compute_reward(model_rationale: str, model_answer: str, prompt: Prompt,
                   mode: str) -> float:
    """Oracle reward for one rollout. See module docstring for why it is exact."""
    assert mode in REWARD_MODES, f"[grpo] unknown reward mode {mode!r}"
    ans_ok = answer_matches(model_answer, prompt.gold_answer)
    if mode == 'exact':
        return 1.0 if (model_rationale == prompt.gold_rationale and ans_ok) else 0.0
    steps = split_steps(model_rationale) if model_rationale.strip() else []
    return 0.5 * prefix_fraction(prompt.gold_steps, steps) + 0.5 * float(ans_ok)


def decode_rollouts(tok, ids: torch.Tensor, slices: Sequence[Tuple[int, int, int, int]]
                    ) -> List[Tuple[str, str]]:
    ids_cpu = ids.cpu()
    out = []
    for b, (p_len, r_end, a_start, a_end) in enumerate(slices):
        rat = tok.decode(ids_cpu[b, p_len:r_end].tolist(), skip_special_tokens=True)
        ans = tok.decode(ids_cpu[b, a_start:a_end].tolist(), skip_special_tokens=True)
        out.append((rat, ans))
    return out


def grouped_advantages(rewards: torch.Tensor, group: int, std_norm: bool) -> torch.Tensor:
    """A_i = r_i - mean(group), optionally / (std + eps).

    A group whose rollouts all scored alike has every A_i == 0 exactly, so it
    drops out of the sum with no special case: the only thing GRPO can learn
    from a prompt is which of its own attempts beat the others.
    """
    r = rewards.view(-1, group)
    adv = r - r.mean(dim=1, keepdim=True)
    if std_norm:
        adv = adv / (r.std(dim=1, unbiased=False, keepdim=True) + STD_EPS)
    return adv.reshape(-1)


# ---------------------------------------------------------------------------
# Policy terms
# ---------------------------------------------------------------------------


def build_ban_table(tok, vocab_size: int, device) -> torch.Tensor:
    """(3, V) bool: which vocabulary entries the sampler banned, per stage.

    The gradient must be taken against the density the sample was drawn from,
    and the sampler bans every special token except the one that ends the
    current stage. Row STAGE_NONE bans nothing (those positions are masked out
    of every sum anyway).
    """
    all_special = _special_ids(tok)
    table = torch.zeros(3, vocab_size, dtype=torch.bool, device=device)
    for stage_id, allowed in ((STAGE_RATIONALE, tok.think_end_id),
                              (STAGE_ANSWER, tok.final_end_id)):
        for tid in all_special - {allowed}:
            table[stage_id, tid] = True
    return table


def policy_log_probs(logits: torch.Tensor, ids: torch.Tensor, stage: torch.Tensor,
                     ban_table: torch.Tensor):
    """Per-position masked log-softmax aligned to the token each logit predicts.

    logits[:, t-1] predicts ids[:, t], so everything is shifted by one and the
    stage tag of the *predicted* token selects the ban row. The ban sentinel is
    finfo.min rather than -inf on purpose: exp() still underflows to exactly
    0, but the finite value keeps p * (logp - logp_ref) and p * logp at 0
    instead of 0 * inf = nan.
    """
    pred_stage = stage[:, 1:]                                  # (b, T-1) int8
    shifted = logits[:, :-1, :]
    ban = ban_table[pred_stage.long()]                         # (b, T-1, V)
    logp = F.log_softmax(shifted.masked_fill(ban, torch.finfo(shifted.dtype).min), dim=-1)
    tok_logp = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return logp, tok_logp, pred_stage > 0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state())


def _restore_rng(state) -> None:
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


def update_step(model, ref_model, optimizer, batch: Dict[str, Any],
                advantages: torch.Tensor, args, ban_table: torch.Tensor,
                device) -> Dict[str, float]:
    """One gradient step over the whole rollout batch (mu=1, no ratio clipping).

    The batch is walked in --grad_batch micro-chunks with gradient
    accumulation. Because both terms are sums normalized by the SAME global
    generated-token count, accumulation is exact rather than approximate: the
    chunks add up to the batch loss, not to an average of chunk losses.
    """
    ids, stage = batch['ids'], batch['stage']
    images = batch['images']
    n_rows = ids.shape[0]

    total_gen = int((stage[:, 1:] > 0).sum().item())
    assert total_gen > 0, "[grpo] rollout batch contains no generated tokens"

    optimizer.zero_grad(set_to_none=True)
    # Dropout stays OFF: the recompute distribution must equal the rollout
    # distribution exactly, otherwise this quietly becomes an off-policy
    # update with an unaccounted-for ratio.
    model.eval()

    pg_sum = 0.0
    kl_sum = 0.0
    ent_sum = 0.0
    for i in range(0, n_rows, args.grad_batch):
        sl = slice(i, min(i + args.grad_batch, n_rows))
        img_c = images[sl].to(device)
        ids_c = ids[sl]
        stage_c = stage[sl]
        adv_c = advantages[sl].to(device)

        logits = model(img_c, ids_c)
        logp, tok_logp, gen = policy_log_probs(logits, ids_c, stage_c, ban_table)
        gen_f = gen.to(logp.dtype)

        with torch.no_grad():
            ref_logits = ref_model(img_c, ids_c)
            ref_logp, _, _ = policy_log_probs(ref_logits, ids_c, stage_c, ban_table)

        probs = logp.exp()
        kl = (probs * (logp - ref_logp)).sum(-1)               # (b, T-1)
        entropy = -(probs * logp).sum(-1)

        pg_term = -(adv_c.unsqueeze(1) * tok_logp * gen_f).sum()
        kl_term = (kl * gen_f).sum()
        loss = (pg_term + args.kl_beta * kl_term) / total_gen
        loss.backward()

        pg_sum += float(pg_term.detach())
        kl_sum += float(kl_term.detach())
        ent_sum += float((entropy.detach() * gen_f).sum())

        del logits, logp, tok_logp, ref_logits, ref_logp, probs, kl, entropy

    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
    optimizer.step()

    return {
        'pg_loss': pg_sum / total_gen,
        'kl': kl_sum / total_gen,
        'entropy': ent_sum / total_gen,
        'grad_norm': grad_norm,
        'gen_tokens': total_gen,
    }


def hard_type_eval(model, text_processor, hard_types: Sequence[str], n: int,
                   max_gen_len: int, seed: int) -> Dict[str, Any]:
    """Greedy answer EM on the hard bucket, on the project's standard eval scenes.

    Runs on its own RNG stream and restores the training streams afterwards, so
    turning evaluation on cannot change the sequence of prompts a run trains on.
    """
    state = _rng_state()
    try:
        set_seed(seed)
        evaluator = VLMEvaluator(model, text_processor, self_consistency_k=1,
                                 temperature=0.0, max_gen_len=max_gen_len)
        out: Dict[str, Any] = {}
        for qtype in hard_types:
            test_set = evaluator.generate_test_set(
                qtype, evaluator.rationale_gen.generators[qtype], n)
            if not test_set:
                continue
            metrics, _ = evaluator.evaluate_test_set(qtype, test_set, show_examples=0)
            out[qtype] = {'n': metrics['n'], 'accuracy': metrics['accuracy'],
                          'rationale_exact': metrics['rationale_exact']}
        total = sum(v['n'] for v in out.values())
        if total:
            out['overall'] = {
                'n': total,
                'accuracy': sum(v['accuracy'] * v['n'] for v in out.values()) / total,
                'rationale_exact': sum(v['rationale_exact'] * v['n']
                                       for v in out.values()) / total,
            }
        return out
    finally:
        model.eval()
        _restore_rng(state)


def train(args) -> None:
    device = best_device()
    print(f"Loading {args.checkpoint} on {device}")
    model, text_processor = load_model(args.checkpoint, args.vocab, device)
    tok = text_processor.tokenizer

    if args.temperature != 1.0:
        # The sampler divides logits by --temperature but policy_log_probs
        # scores the unscaled distribution, so the gradient density only
        # matches the sampling density at T=1.0. Any other value is an
        # uncorrected off-policy update.
        print(f"WARNING: --temperature {args.temperature} != 1.0 makes the "
              f"update off-policy (no importance correction is applied)")

    # Frozen reference policy = the starting checkpoint. The KL term is what
    # stops GRPO from trading the other 10 question types for the 4 it is
    # being rewarded on.
    ref_model = copy.deepcopy(model).to(device).eval()
    ref_model.requires_grad_(False)

    runtime = setup_runtime(prefer_compile=False)
    optimizer = runtime["make_optimizer_from_groups"](
        build_param_groups(model, wd=0.01), lr=args.lr)

    ban_table = build_ban_table(tok, tok.get_vocab_size(), device)
    shape_gen = ShapeGenerator()
    rationale_gen = RationaleGenerator()
    all_types = list(rationale_gen.generators.keys())
    hard_types = list(args.hard_types)
    unknown = [t for t in hard_types if t not in rationale_gen.generators]
    assert not unknown, f"[grpo] unknown --hard_types {unknown}; have {all_types}"

    os.makedirs(args.out_dir, exist_ok=True)
    meta = {
        'script': 'grpo.py',
        'args': vars(args),
        'base_checkpoint': os.path.abspath(args.checkpoint),
        'base_checkpoint_sha256': sha256_of(args.checkpoint),
        'base_checkpoint_bytes': os.path.getsize(args.checkpoint),
        'vocab': os.path.abspath(args.vocab) if args.vocab else None,
        'vocab_size': tok.get_vocab_size(),
        'device': str(device),
        'all_types': all_types,
        'torch': torch.__version__,
        'started': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(args.out_dir, 'grpo_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    if args.vocab:
        tok.save_vocab(os.path.join(args.out_dir, 'tokenizer_vocab.json'))

    set_seed(args.seed)
    print(f"GRPO: {args.steps} steps x {args.prompts} prompts x {args.group} rollouts "
          f"= {args.steps * args.prompts * args.group} chains, T={args.temperature}, "
          f"reward={args.reward}, beta={args.kl_beta}, lr={args.lr}")

    fixed_prompts: Optional[List[Prompt]] = None
    if args.fixed_prompts:
        fixed_prompts = draw_prompt_batch(shape_gen, rationale_gen, tok, args.prompts,
                                          args.hard_frac, hard_types, all_types)
        types = sorted({p.qtype for p in fixed_prompts})
        print(f"--fixed_prompts: {len(fixed_prompts)} frozen prompts, types {types}")

    def save(path):
        torch.save(getattr(model, '_orig_mod', model).state_dict(), path)

    for step in tqdm(range(1, args.steps + 1), desc="grpo"):
        t0 = time.time()
        prompts = fixed_prompts if fixed_prompts is not None else draw_prompt_batch(
            shape_gen, rationale_gen, tok, args.prompts, args.hard_frac,
            hard_types, all_types)

        batch = rollout_batch(model, prompts, args.group, args.temperature,
                              args.rollout_batch, args.max_gen_len)
        t_roll = time.time() - t0

        decoded = decode_rollouts(tok, batch['ids'], batch['slices'])
        rewards = torch.tensor(
            [compute_reward(rat, ans, batch['row_prompts'][i], args.reward)
             for i, (rat, ans) in enumerate(decoded)], dtype=torch.float32)
        advantages = grouped_advantages(rewards, args.group, args.std_norm)

        stats = update_step(model, ref_model, optimizer, batch, advantages,
                            args, ban_table, device)
        t_step = time.time() - t0

        r = rewards.view(len(prompts), args.group)
        per_type: Dict[str, List[float]] = {}
        for p_idx, prompt in enumerate(prompts):
            per_type.setdefault(prompt.qtype, []).extend(r[p_idx].tolist())
        type_stats = {
            k: {'n': len(v), 'mean': statistics.fmean(v),
                'std': statistics.pstdev(v) if len(v) > 1 else 0.0}
            for k, v in sorted(per_type.items())
        }

        record = {
            'step': step,
            'phase': 'grpo',
            'reward_mean': float(rewards.mean()),
            'reward_std': float(rewards.std(unbiased=False)),
            'reward_max': float(rewards.max()),
            'reward_by_type': type_stats,
            'nonzero_var_groups': float((r.std(dim=1, unbiased=False) > 0).float().mean()),
            'adv_abs_mean': float(advantages.abs().mean()),
            'pg_loss': stats['pg_loss'],
            'kl': stats['kl'],
            'entropy': stats['entropy'],
            'grad_norm': stats['grad_norm'],
            'gen_tokens': stats['gen_tokens'],
            'lr': optimizer.param_groups[0]['lr'],
            'rollout_seconds': t_roll,
            'seconds': t_step,
        }

        if args.eval_every and (step % args.eval_every == 0 or step == args.steps):
            record['hard_eval'] = hard_type_eval(model, text_processor, hard_types,
                                                 args.eval_n, args.max_gen_len,
                                                 args.seed + 10_000)
            save(os.path.join(args.out_dir, f'grpo_step_{step}.pth'))

        with open(args.log, 'a') as f:
            f.write(json.dumps(record) + '\n')

        if step % args.print_every == 0 or step == 1:
            print(f"  step {step:>5} reward {record['reward_mean']:.3f} "
                  f"(std {record['reward_std']:.3f}, live groups "
                  f"{record['nonzero_var_groups']:.2f}) kl {record['kl']:.4f} "
                  f"ent {record['entropy']:.3f} |g| {record['grad_norm']:.3f} "
                  f"{record['seconds']:.1f}s")

    save(os.path.join(args.out_dir, 'grpo_final.pth'))
    meta['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    meta['final_checkpoint'] = os.path.abspath(os.path.join(args.out_dir, 'grpo_final.pth'))
    with open(os.path.join(args.out_dir, 'grpo_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nGRPO complete -> {meta['final_checkpoint']}")


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


def selftest(args) -> None:
    """Four checks that the update is wired to what it claims to be wired to."""
    device = best_device()
    print(f"selftest on {device}: {args.checkpoint}")
    model, text_processor = load_model(args.checkpoint, args.vocab, device)
    tok = text_processor.tokenizer
    ban_table = build_ban_table(tok, tok.get_vocab_size(), device)
    shape_gen = ShapeGenerator()
    rationale_gen = RationaleGenerator()
    all_types = list(rationale_gen.generators.keys())
    failures: List[str] = []

    # --- 1. sampler equivalence with the verified reference decoder ----------
    set_seed(args.seed)
    n = max(32, args.selftest_prompts)
    # A type whose current question phrasings are not spellable in the loaded
    # vocabulary is reported and skipped rather than aborting the run: that is a
    # questions.py/checkpoint mismatch, not a fault in the code under test, and
    # the remaining types still exercise every path. train() keeps the strict
    # behavior -- a training run must never quietly drop a question type.
    prompts, unspellable = [], {}
    i = 0
    while len(prompts) < n and i < n * 8:
        qtype = all_types[i % len(all_types)]
        i += 1
        if qtype in unspellable:
            continue
        try:
            prompts.append(draw_prompt(shape_gen, rationale_gen, tok, qtype))
        except ValueError as e:
            unspellable[qtype] = str(e).split('.')[0]
    n = len(prompts)
    assert n >= 8, f"[grpo] only {n} prompts were spellable; vocabulary mismatch"
    if unspellable:
        print("    !! types skipped (question phrasing outside the loaded vocabulary): "
              + "; ".join(f"{k}: {v}" for k, v in unspellable.items()))
    images = torch.stack([p.image_u8 for p in prompts]).to(torch.float32) / 255.0
    q_ids = [p.q_ids for p in prompts]

    reference = generate_chains_batched(model, images, q_ids, temperature=0.0,
                                        max_length=args.max_gen_len)
    mine = sample_rollouts(model, images, q_ids, temperature=0.0,
                           max_length=args.max_gen_len)
    mine_dec = decode_rollouts(tok, mine['ids'], mine['slices'])
    ids_cpu = mine['ids'].cpu()

    str_mismatch = tok_mismatch = 0
    for b, (rat_ids, ans_ids) in enumerate(reference):
        ref_rat = tok.decode(rat_ids, skip_special_tokens=True)
        ref_ans = tok.decode(ans_ids, skip_special_tokens=True)
        if (ref_rat, ref_ans) != mine_dec[b]:
            str_mismatch += 1
        p_len, r_end, a_start, a_end = mine['slices'][b]
        if (ids_cpu[b, p_len:r_end].tolist() != list(rat_ids)
                or ids_cpu[b, a_start:a_end].tolist() != list(ans_ids)):
            tok_mismatch += 1
    ok = str_mismatch == 0 and tok_mismatch == 0
    print(f"[1] sampler equivalence @T=0 over {n} mixed-type prompts: "
          f"{n - str_mismatch}/{n} string-identical, {n - tok_mismatch}/{n} "
          f"token-identical -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append('sampler equivalence')

    # gold round-trip fidelity: the reward's normalization must be a no-op here
    roundtrip_changed = sum(
        1 for p in prompts
        if p.gold_rationale != tok.decode(tok.tokenize(p.gold_rationale),
                                          skip_special_tokens=True))
    print(f"    gold rationale tokenizer round-trip changed {roundtrip_changed}/{n} prompts")

    # --- 2. mask / log-prob alignment ---------------------------------------
    stage = mine['stage']
    with torch.no_grad():
        logits = model(images.to(device), mine['ids'])
        logp, tok_logp, gen = policy_log_probs(logits, mine['ids'], stage, ban_table)
        argmax = logp.argmax(dim=-1)
        sampled = mine['ids'][:, 1:]
        agree = int(((argmax == sampled) | ~gen).all(dim=1).sum())
        n_bad = int((gen & (argmax != sampled)).sum())
        finite = bool(torch.isfinite(tok_logp[gen]).all())
        n_gen = int(gen.sum())
        # the forced </THINK> and structural <FINAL> must never be credited
        forced = int(((sampled == tok.final_start_id) & gen).sum())
    ok = n_bad == 0 and finite and forced == 0
    print(f"[2] mask/logprob alignment: {n_gen} generated positions, "
          f"{n_bad} argmax disagreements, {n} /{n} rows clean ({agree}), "
          f"logprobs finite={finite}, structural <FINAL> credited={forced} "
          f"-> {'PASS' if ok else 'FAIL'}")
    print(f"    mean generated-token logprob {float(tok_logp[gen].mean()):.4f}, "
          f"min {float(tok_logp[gen].min()):.4f}")
    if not ok:
        failures.append('mask/logprob alignment')

    # --- 3. reward oracle ----------------------------------------------------
    probe = prompts[0]
    for p in prompts:
        if len(p.gold_steps) >= 3:
            probe = p
            break
    r_exact_gold = compute_reward(probe.gold_rationale, probe.gold_answer, probe, 'exact')
    r_dense_gold = compute_reward(probe.gold_rationale, probe.gold_answer, probe, 'dense')

    steps = list(probe.gold_steps)
    k = min(1, len(steps) - 1)
    steps[k] = 'red circle at row 9 col 9'
    corrupt = ' . '.join(steps)
    r_exact_bad = compute_reward(corrupt, probe.gold_answer, probe, 'exact')
    r_dense_bad = compute_reward(corrupt, probe.gold_answer, probe, 'dense')
    expect_dense_bad = 0.5 * (k / len(probe.gold_steps)) + 0.5
    r_exact_ans = compute_reward(probe.gold_rationale, 'definitely not the answer',
                                 probe, 'exact')
    ok = (r_exact_gold == 1.0 and abs(r_dense_gold - 1.0) < 1e-9
          and r_exact_bad == 0.0 and abs(r_dense_bad - expect_dense_bad) < 1e-9
          and r_exact_ans == 0.0)
    print(f"[3] reward oracle ({probe.qtype}, {len(probe.gold_steps)} gold steps): "
          f"gold exact={r_exact_gold} dense={r_dense_gold}; "
          f"step-{k}-corrupted exact={r_exact_bad} dense={r_dense_bad} "
          f"(expected {expect_dense_bad}); wrong-answer exact={r_exact_ans} "
          f"-> {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append('reward oracle')

    # --- 4. zero-advantage no-op --------------------------------------------
    group = 4
    n_groups = 4
    sub = prompts[:n_groups * group]
    batch = {
        'ids': mine['ids'][:len(sub)],
        'stage': mine['stage'][:len(sub)],
        'images': images[:len(sub)],
    }
    # Rewards constant WITHIN each group but different ACROSS groups: the mean
    # baseline must cancel them exactly, no matter what the group scored.
    rewards = torch.tensor([float(i // group) for i in range(len(sub))])
    advantages = grouped_advantages(rewards, group, std_norm=False)
    adv_max = float(advantages.abs().max())

    class _A:
        grad_batch = 16
        kl_beta = 0.0
        grad_clip = 1e9

    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    stats = update_step(model, model, opt, batch, advantages, _A(), ban_table, device)
    ok = adv_max == 0.0 and stats['pg_loss'] == 0.0 and stats['grad_norm'] < 1e-6
    print(f"[4] zero-advantage no-op ({n_groups} groups x {group}): "
          f"max|A|={adv_max:.3e}, pg_loss={stats['pg_loss']:.3e}, "
          f"grad_norm={stats['grad_norm']:.3e} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append('zero-advantage no-op')

    print(f"\nselftest: {4 - len(failures)}/4 passed"
          + (f"  FAILURES: {failures}" if failures else ""))
    if failures:
        raise SystemExit(1)


# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO fine-tuning for the Toy VLM")
    p.add_argument("--checkpoint", type=str, default="toy_vlm_cot.pth")
    p.add_argument("--vocab", type=str, default="tokenizer_vocab.json",
                   help="tokenizer vocab json; empty rebuilds it deterministically")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--prompts", type=int, default=16, help="P: prompts per step")
    p.add_argument("--group", type=int, default=8, help="G: rollouts per prompt")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="rollout temperature, applied to BOTH decode stages")
    p.add_argument("--reward", type=str, default="exact", choices=REWARD_MODES)
    p.add_argument("--std_norm", action="store_true",
                   help="divide the advantage by the group std (default: mean-only baseline)")
    p.add_argument("--kl_beta", type=float, default=0.02)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--hard_frac", type=float, default=0.8,
                   help="probability a prompt is drawn from --hard_types")
    p.add_argument("--hard_types", type=str, nargs='+', default=list(HARD_TYPES))
    p.add_argument("--fixed_prompts", action="store_true",
                   help="draw the P prompts once and reuse them every step "
                        "(overfit diagnostics)")
    p.add_argument("--rollout_batch", type=int, default=64,
                   help="max rows decoded in parallel")
    p.add_argument("--grad_batch", type=int, default=32,
                   help="rows per accumulation micro-chunk in the update")
    p.add_argument("--max_gen_len", type=int, default=MAX_GEN_LEN)
    p.add_argument("--eval_every", type=int, default=100, help="0 disables")
    p.add_argument("--eval_n", type=int, default=60, help="eval samples per hard type")
    p.add_argument("--print_every", type=int, default=10)
    p.add_argument("--out_dir", type=str, default="grpo_checkpoints")
    p.add_argument("--log", type=str, default="grpo_log.jsonl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--selftest", action="store_true",
                   help="run the sampler/mask/reward/no-op checks and exit")
    p.add_argument("--selftest_prompts", type=int, default=32)
    args = p.parse_args()
    assert 0.0 <= args.hard_frac <= 1.0, "--hard_frac must be a probability"
    assert args.group >= 2, "--group must be >= 2 for a group baseline to exist"
    if not args.vocab:
        args.vocab = None
    return args


def main() -> None:
    args = parse_args()
    if args.selftest:
        selftest(args)
        return
    train(args)


if __name__ == "__main__":
    main()
