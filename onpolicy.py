"""
On-policy correction data collection (DAgger-style) for the Toy VLM.

The model reaches ~100% teacher-forced rationale token accuracy but far less
free-running answer accuracy: pure exposure bias. Every trace it has ever been
trained on was drawn from the gold generator, so it has never been taught what
to do once its own enumeration goes wrong. This module closes that loop:

  1. sample the model's OWN chain for a fresh scene/question,
  2. find the first step that disagrees with the gold rationale,
  3. splice `wait` + the gold correction after that step,
  4. save the trace with the wrong step marked *unsupervised*.

Alignment is positional. Every gold rationale enumerates in canonical raster
order, so the model's chain can be compared to the gold one ' . '-step by
' . '-step; the first mismatching index is the first error. The corrected
trajectory is exactly the shape questions.inject_correction produces for random
corruptions:

    gold[0..i-1] . {model's wrong step i} . wait . gold[i..end]
    `- supervised -' `---- unsupervised ---'  `--- supervised ---'

so the same segment construction, the same supervision policy, and the same
budget accounting apply -- the only difference is that the "corruption" is the
model's own emitted step rather than a synthetic one.

Special cases
-------------
* chain identical to gold      -> fully supervised positive example
* unparseable / hallucinated   -> ordinary mismatch (it becomes the wrong step)
* model emitted </THINK> early -> no wrong step exists to condition on, so the
  sample is kept as a plain gold trace (counted separately as `early_stop`).
  Splicing `wait` after a *correct* prefix would teach the model to interject
  the correction marker when nothing is wrong, which is the one thing the
  correction curriculum must never do.
* budget overflow / empty step -> skipped (counted)

STaR / rejection-sampling filtering
------------------------------------
Every kept sample carries an int8 `disposition` tensor recording which of the
categories above produced it (see DISPOSITION below; mirrors KIND_ID/KINDS).
--star restricts the saved tensors to the fully gold-supervised dispositions --
'exact' (the model's own chain matched gold exactly) and 'early_stop' (no
wrong step exists to condition on, so training falls back to the plain gold
trace) -- and discards 'corrected' samples, whose saved trace still carries an
unsupervised, model-authored wrong step. This is the STaR / rejection-sampling
variant of the correction curriculum: fine-tune only on chains that were
already right, rather than teaching recovery from an error.

Usage:
    uv run python onpolicy.py --checkpoint run5/checkpoints/best.pth \
        --vocab run5/tokenizer_vocab.json --n 50000 --batch_size 96 \
        --out onpolicy_data.pt --seed 0
"""

import argparse
import json
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import ToyVLM
from questions import (CORRECTION_MARKER, DIFFICULTY_MAP, RationaleGenerator,
                       STEP_SEP, WORD_BUDGET, join_steps, split_steps)
from shapes import MAX_OBJECTS, MIN_OBJECTS, ShapeGenerator
from text import MAX_SEQ_LEN, NUM_IMG_TOKENS, TextProcessor
from train_model import DIFFICULTIES

# Per-stage decode cap, matching model.generate_response's default. Raised
# 80 -> 160 for the run-11 rank-grouped ordinal trace, which runs to 141
# tokens on a 12-object scene; 80 truncated 52% of those chains.
MAX_GEN_LEN = 160

# Sample dispositions, in the order the summary prints them.
KINDS = ['exact', 'corrected', 'early_stop', 'skip_budget', 'skip_unusable']
KIND_ID = {k: i for i, k in enumerate(KINDS)}

# id -> name for the per-sample `disposition` tensor (int8) collect() saves.
# Mirrors KIND_ID/KINDS above; kept as an explicit dict per the on-disk
# contract so downstream loaders (finetune_onpolicy.py) can filter by
# disposition without re-deriving the mapping.
DISPOSITION = dict(enumerate(KINDS))

# Dispositions eligible for --star (rejection-sampling / STaR filtering): the
# saved rationale for both is the gold trace verbatim, fully supervised end to
# end -- 'exact' because the model's own chain matched it, 'early_stop'
# because there was no wrong step to condition a correction on, so
# build_correction falls back to plain gold. Neither carries the unsupervised,
# model-authored wrong step that 'corrected' samples do.
STAR_KEEP_KINDS = ('exact', 'early_stop')


# ---------------------------------------------------------------------------
# Batched two-stage generation
# ---------------------------------------------------------------------------
# Sequential decoding of 50k chains would take hours: every step is a full
# forward over MAX_SEQ_LEN positions, and the model has no KV cache. Batching B
# rows is exact rather than approximate here -- the prefix-LM mask is
# bidirectional only inside the fixed image block and causal after it, so a
# row's logits at its own last real position cannot see its padding, and rows
# never see each other. A row that has already emitted its stage-ending token is
# frozen in place while its neighbours keep decoding.


def _special_ids(tok) -> set:
    """The special-token ids banned from free generation (see generate_response)."""
    return {
        tok.pad_token_id, tok.bos_token_id, tok.eos_token_id, tok.unk_token_id,
        tok.user_token_id, tok.assistant_token_id,
        tok.think_start_id, tok.think_end_id, tok.final_start_id, tok.final_end_id,
        tok.img_start_id, tok.img_end_id, tok.img_token_id,
    }


@torch.no_grad()
def generate_chains_batched(model, images: torch.Tensor,
                            question_ids_list: Sequence[Sequence[int]],
                            temperature: float = 0.0,
                            max_length: int = MAX_GEN_LEN
                            ) -> List[Tuple[List[int], List[int]]]:
    """Decode B chains in parallel; returns per row (rationale_ids, answer_ids).

    Equivalent to calling model.generate_response once per row -- same prompt,
    same ban lists, same two length guards, same force-closing of an
    unterminated span -- and verified to match it token for token at
    temperature 0.

    Args:
        model: ToyVLM instance (put in eval mode here).
        images: (B, 3, 64, 64) float32 in [0,1].
        question_ids_list: per-row question token ids (already tokenized).
        temperature: rationale-stage sampling temperature; 0 = greedy.
        max_length: max decode steps per stage.
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

    # Rows are padded to MAX_SEQ_LEN exactly as generate_response pads its single
    # row, so the forward sees the same shapes and the same numerics.
    ids = torch.full((B, MAX_SEQ_LEN), tok.pad_token_id, dtype=torch.long, device=device)
    for b, prompt in enumerate(prompts):
        assert len(prompt) <= MAX_SEQ_LEN - 4, \
            f"prompt of {len(prompt)} tokens leaves no room to decode"
        ids[b, :len(prompt)] = torch.tensor(prompt, dtype=torch.long, device=device)

    prompt_lens = torch.tensor([len(p) for p in prompts], dtype=torch.long, device=device)
    lengths = prompt_lens.clone()
    rows = torch.arange(B, device=device)
    all_special = _special_ids(tok)

    def run_stage(lengths, allowed_special, cap, temp):
        """Decode until every row hits `allowed_special`, the cap, or max_length."""
        banned = torch.tensor(sorted(all_special - {allowed_special}),
                              dtype=torch.long, device=device)
        done = lengths >= cap
        for _ in range(max_length):
            if bool(done.all()):
                break
            logits = model(images, ids)                       # (B, T, V)
            pos = (lengths - 1).clamp(0, MAX_SEQ_LEN - 1)     # each row's own last real token
            next_logits = logits[rows, pos, :].clone()        # (B, V)
            next_logits[:, banned] = float('-inf')
            if temp > 0:
                nxt = torch.multinomial(F.softmax(next_logits / temp, dim=-1), 1).squeeze(1)
            else:
                nxt = next_logits.argmax(dim=-1)

            # Frozen rows write their existing value back, so `done` needs no
            # scatter mask and the tensor stays a faithful transcript.
            write = lengths.clamp(max=MAX_SEQ_LEN - 1)
            keep = ids[rows, write]
            ids[rows, write] = torch.where(done, keep, nxt)
            lengths = lengths + (~done).long()
            done = done | (nxt == allowed_special) | (lengths >= cap)
        return lengths

    # Stage 1: rationale until </THINK> (sampled when temperature > 0).
    # Cap MAX_SEQ_LEN - 4 leaves room for </THINK> <FINAL> answer </FINAL>.
    lengths = run_stage(lengths, tok.think_end_id, MAX_SEQ_LEN - 4, temperature)
    closed = ids[rows, (lengths - 1).clamp(min=0)] == tok.think_end_id
    rat_end = torch.where(closed, lengths - 1, lengths)       # exclusive, sans </THINK>
    ids[rows, rat_end] = tok.think_end_id                     # force-close an open span
    lengths = torch.where(closed, lengths, lengths + 1)
    ids[rows, lengths] = tok.final_start_id
    lengths = lengths + 1

    # Stage 2: answer until </FINAL>, always greedy.
    ans_start = lengths.clone()
    lengths = run_stage(lengths, tok.final_end_id, MAX_SEQ_LEN - 1, 0.0)
    closed = ids[rows, (lengths - 1).clamp(min=0)] == tok.final_end_id
    ans_end = torch.where(closed, lengths - 1, lengths)

    ids_cpu = ids.cpu()
    starts = prompt_lens.cpu().tolist()
    rat_ends = rat_end.cpu().tolist()
    ans_starts = ans_start.cpu().tolist()
    ans_ends = ans_end.cpu().tolist()
    return [
        (ids_cpu[b, starts[b]:rat_ends[b]].tolist(),
         ids_cpu[b, ans_starts[b]:ans_ends[b]].tolist())
        for b in range(B)
    ]


# ---------------------------------------------------------------------------
# Positional alignment against the gold rationale
# ---------------------------------------------------------------------------


def first_mismatch(gold_steps: Sequence[str], model_steps: Sequence[str]) -> Optional[int]:
    """First index where the two step lists differ, or None if identical.

    A step present in one list where the other has ended counts as a mismatch at
    that index, so a chain that stops early and a chain that rambles on are both
    located at their first divergent position.
    """
    for k in range(min(len(gold_steps), len(model_steps))):
        if gold_steps[k] != model_steps[k]:
            return k
    if len(gold_steps) == len(model_steps):
        return None
    return min(len(gold_steps), len(model_steps))


def build_correction(gold_rationale: str, model_rationale: str,
                     reserved_words: int) -> Tuple[str, Any, Optional[int]]:
    """Align a sampled chain against gold and build its training rationale.

    Returns ``(kind, rationale_spec, first_error_index)`` where rationale_spec is
    a plain string (fully supervised) or the ``(text, supervised)`` segment list
    questions.inject_correction returns, and kind is one of KINDS.
    """
    gold_steps = split_steps(gold_rationale) if gold_rationale else []
    model_steps = split_steps(model_rationale) if model_rationale.strip() else []

    index = first_mismatch(gold_steps, model_steps)
    if index is None:
        return 'exact', gold_rationale, None

    if index >= len(model_steps):
        # The model closed </THINK> before the gold trace was done: there is no
        # wrong step to condition the correction on. Train on the gold trace so
        # the supervision at that state is "keep going", not "say wait".
        return 'early_stop', gold_rationale, index

    wrong_text = model_steps[index].strip()
    if not wrong_text:
        return 'skip_unusable', None, index

    # Same splice as questions.inject_correction: each piece carries the
    # separators it owns, the wrong step alone is unsupervised, and `wait` plus
    # everything after it is supervised.
    prefix = ''.join(step + STEP_SEP for step in gold_steps[:index])
    tail = join_steps([CORRECTION_MARKER] + list(gold_steps[index:]))
    segments = [(prefix, True), (wrong_text, False), (STEP_SEP + tail, True)]
    segments = [(text, supervised) for text, supervised in segments if text]

    expected = join_steps(list(gold_steps[:index]) + [wrong_text, CORRECTION_MARKER]
                          + list(gold_steps[index:]))
    assert ''.join(text for text, _ in segments) == expected, (
        f"[onpolicy] segment reassembly does not reproduce {expected!r}")

    # questions.inject_correction declines when the gold draw sits within
    # CORRECTION_RESERVE words of the budget -- a *bound* on the 11 words its
    # fixed-form insertion can add. A model's wrong step carries no such bound
    # (it can ramble for the whole decode cap), so the same check is applied with
    # the insertion measured instead of bounded.
    if len(expected.split()) + reserved_words > WORD_BUDGET:
        return 'skip_budget', None, index

    return 'corrected', segments, index


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def _draw_sample(shape_gen: ShapeGenerator, rationale_gen: RationaleGenerator,
                 difficulties: Sequence[str], hard_frac: float = 0.0,
                 hard_types: Sequence[str] = ()):
    """One scene + gold QA triple (with its question type), redrawn until a
    generator commits.

    With probability `hard_frac` the type is drawn uniformly from `hard_types`
    -- the failure-focused collection the STaR/DAgger arms share with grpo.py's
    prompt mix; otherwise a difficulty is drawn as before and a type uniformly
    within it. Either way the concrete type is drawn here (not inside
    generate_qa_with_rationale), so every sample carries the label the
    per-type stats aggregate by.
    """
    while True:
        num_shapes = random.randint(MIN_OBJECTS, MAX_OBJECTS)
        image, metadata = shape_gen.generate_multi_shape_image(num_shapes, True)
        if hard_types and random.random() < hard_frac:
            qtype = random.choice(list(hard_types))
        else:
            difficulty = random.choice(list(difficulties))
            qtype = random.choice(DIFFICULTY_MAP[difficulty])
        question, answer, rationale = rationale_gen.generators[qtype](metadata)
        if question is not None and answer is not None and rationale:
            return image, metadata, question, answer, rationale, qtype


def collect(model, text_processor: TextProcessor, n_samples: int, batch_size: int,
            out_path: str, temperature_mix: Sequence[float] = (0.0, 0.7),
            seed: int = 0, max_gen_len: int = MAX_GEN_LEN,
            num_examples: int = 3, star: bool = False,
            hard_frac: float = 0.0, hard_types: Sequence[str] = ()) -> Dict[str, Any]:
    """Sample chains, splice corrections, and save the padded training tensors.

    `star`: when True, keep only STAR_KEEP_KINDS dispositions ('exact' and
    'early_stop') in the saved tensors, discarding 'corrected' samples -- the
    STaR / rejection-sampling variant (see module docstring).

    `hard_frac` / `hard_types`: failure-focused collection (see _draw_sample).
    Per-type disposition and answer-EM stats are always recorded; under --star
    a type's 'exact' fraction is its pass@1 at the collection temperature.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tok = text_processor.tokenizer
    shape_gen = ShapeGenerator()
    rationale_gen = RationaleGenerator()

    images_out: List[torch.Tensor] = []
    inp_out: List[torch.Tensor] = []
    tgt_out: List[torch.Tensor] = []
    rat_out: List[torch.Tensor] = []
    ans_out: List[torch.Tensor] = []
    kinds_out: List[int] = []

    counts = {k: 0 for k in KINDS}
    # qtype -> {'n': int, 'answer_hits': int, <kind>: int, ...}
    by_type: Dict[str, Dict[str, int]] = {}
    error_indices: List[int] = []
    answer_hits = 0
    examples: List[Dict[str, Any]] = []
    n_batches = (n_samples + batch_size - 1) // batch_size

    t0 = time.time()
    for bi in tqdm(range(n_batches), desc="collect"):
        take = min(batch_size, n_samples - bi * batch_size)
        temperature = float(temperature_mix[bi % len(temperature_mix)])

        drawn = [_draw_sample(shape_gen, rationale_gen, DIFFICULTIES,
                              hard_frac=hard_frac, hard_types=hard_types)
                 for _ in range(take)]
        raw_images = np.stack([d[0] for d in drawn])                    # (B,H,W,3) uint8
        img_u8 = torch.from_numpy(raw_images).permute(0, 3, 1, 2).contiguous()
        images = img_u8.to(torch.float32) / 255.0
        q_ids = [tok.tokenize(d[2]) for d in drawn]

        chains = generate_chains_batched(model, images, q_ids,
                                         temperature=temperature,
                                         max_length=max_gen_len)

        for b, (_, _, question, answer, gold_rationale, qtype) in enumerate(drawn):
            rat_ids, ans_ids = chains[b]
            model_rationale = tok.decode(rat_ids, skip_special_tokens=True)
            model_answer = tok.decode(ans_ids, skip_special_tokens=True)
            hit = int(model_answer.strip() == answer.strip())
            answer_hits += hit

            reserved = len(question.split()) + len(answer.split())
            kind, spec, index = build_correction(gold_rationale, model_rationale, reserved)
            counts[kind] += 1
            trow = by_type.setdefault(qtype, {'n': 0, 'answer_hits': 0,
                                              **{k: 0 for k in KINDS}})
            trow['n'] += 1
            trow['answer_hits'] += hit
            trow[kind] += 1
            if kind == 'corrected':
                error_indices.append(index)

            if spec is None:
                continue

            inp, tgt, rat_mask, ans_mask = text_processor.prepare_input_sequence(
                question, answer, spec)
            images_out.append(img_u8[b].clone())
            inp_out.append(torch.tensor(text_processor.pad_sequence(inp), dtype=torch.long))
            tgt_out.append(torch.tensor(text_processor.pad_sequence(tgt), dtype=torch.long))
            rat_out.append(torch.tensor(text_processor.pad_sequence(rat_mask), dtype=torch.float32))
            ans_out.append(torch.tensor(text_processor.pad_sequence(ans_mask), dtype=torch.float32))
            kinds_out.append(KIND_ID[kind])

            if kind == 'corrected' and len(examples) < num_examples:
                examples.append({
                    'qtype': qtype,
                    'question': question,
                    'answer': answer,
                    'temperature': temperature,
                    'gold_rationale': gold_rationale,
                    'model_rationale': model_rationale,
                    'model_answer': model_answer,
                    'first_error_step': index,
                    'segments': [[text, supervised] for text, supervised in spec],
                })

        del images, img_u8, chains

    kept = len(inp_out)
    assert kept > 0, "[onpolicy] no usable samples collected"
    stats = {
        'n_requested': n_samples,
        'n_kept': kept,
        'counts': counts,
        'frac': {k: counts[k] / max(1, n_samples) for k in KINDS},
        'mean_first_error_step': (sum(error_indices) / len(error_indices))
                                 if error_indices else None,
        'free_running_answer_em': answer_hits / max(1, n_samples),
        'hard_frac': hard_frac,
        'hard_types': list(hard_types),
        'by_type': {t: dict(row) for t, row in sorted(by_type.items())},
        'temperature_mix': list(temperature_mix),
        'seed': seed,
        'batch_size': batch_size,
        'seconds': time.time() - t0,
    }

    if star:
        # Filter every parallel per-sample list to STAR_KEEP_KINDS, computed
        # against the full kinds_out before any reassignment below so the
        # discard breakdown is exact.
        keep_ids = {KIND_ID[k] for k in STAR_KEEP_KINDS}
        keep_mask = [kid in keep_ids for kid in kinds_out]
        discarded_by_kind = {
            KINDS[kid]: sum(1 for k in kinds_out if k == kid)
            for kid in sorted(set(kinds_out) - keep_ids)
        }
        star_kept = sum(keep_mask)
        star_discarded = kept - star_kept
        assert star_kept > 0, "[onpolicy] --star discarded every collected sample"

        images_out = [t for t, keep in zip(images_out, keep_mask) if keep]
        inp_out = [t for t, keep in zip(inp_out, keep_mask) if keep]
        tgt_out = [t for t, keep in zip(tgt_out, keep_mask) if keep]
        rat_out = [t for t, keep in zip(rat_out, keep_mask) if keep]
        ans_out = [t for t, keep in zip(ans_out, keep_mask) if keep]
        kinds_out = [kid for kid in kinds_out if kid in keep_ids]

        stats['star'] = {
            'enabled': True,
            'keep_kinds': list(STAR_KEEP_KINDS),
            'kept': star_kept,
            'discarded': star_discarded,
            'discarded_by_kind': discarded_by_kind,
        }
    else:
        stats['star'] = {'enabled': False}

    payload = {
        'images': torch.stack(images_out),              # (N,3,64,64) uint8
        'input_tokens': torch.stack(inp_out),
        'target_tokens': torch.stack(tgt_out),
        'rat_mask': torch.stack(rat_out),
        'ans_mask': torch.stack(ans_out),
        'kind': torch.tensor(kinds_out, dtype=torch.long),
        'disposition': torch.tensor(kinds_out, dtype=torch.int8),
        'kind_names': KINDS,
        'stats': stats,
        'examples': examples,
    }
    torch.save(payload, out_path)
    return {'stats': stats, 'examples': examples, 'out_path': out_path}


def print_report(result: Dict[str, Any]) -> None:
    stats = result['stats']
    n = stats['n_requested']
    print(f"\nCollected {stats['n_kept']}/{n} usable samples in {stats['seconds']:.1f}s "
          f"-> {result['out_path']}")
    for kind in KINDS:
        print(f"  {kind:<14} {stats['counts'][kind]:>6}  {100 * stats['frac'][kind]:5.1f}%")
    mfe = stats['mean_first_error_step']
    print(f"  mean first-error step index: {mfe:.2f}" if mfe is not None
          else "  mean first-error step index: n/a")
    print(f"  free-running answer EM: {stats['free_running_answer_em']:.3f}")

    by_type = stats.get('by_type', {})
    if len(by_type) > 1:
        # 'exact' fraction per type is that type's pass@1 at the collection
        # temperature -- the number that decides whether rejection sampling
        # has anything to keep on the questions the model usually misses.
        print(f"  {'type':<20} {'n':>6} {'ans EM':>7} {'exact':>7} "
              f"{'early':>7} {'corr':>7}")
        for t, row in by_type.items():
            n = max(1, row['n'])
            print(f"  {t:<20} {row['n']:>6} {row['answer_hits'] / n:>7.3f} "
                  f"{row['exact'] / n:>7.3f} {row['early_stop'] / n:>7.3f} "
                  f"{row['corrected'] / n:>7.3f}")

    star = stats.get('star', {})
    if star.get('enabled'):
        total = star['kept'] + star['discarded']
        rate = star['kept'] / max(1, total)
        print(f"  --star: kept {star['kept']}/{total} ({100 * rate:5.1f}%) "
              f"[{'+'.join(star['keep_kinds'])}], discarded {star['discarded']} "
              f"{star['discarded_by_kind']}")

    for i, ex in enumerate(result['examples']):
        print(f"\n--- corrected example {i + 1} ({ex['qtype']}, T={ex['temperature']}) ---")
        print(f"  Q: {ex['question']}   gold A: {ex['answer']}   model A: {ex['model_answer']}")
        print(f"  gold : {ex['gold_rationale']}")
        print(f"  model: {ex['model_rationale']}")
        print(f"  first error at step {ex['first_error_step']}; spliced trace:")
        for text, supervised in ex['segments']:
            print(f"    [{'sup  ' if supervised else 'UNSUP'}] {text!r}")


def load_model(checkpoint: str, vocab: Optional[str], device: torch.device):
    """Build the tokenizer, assert it matches the checkpoint, and load weights."""
    text_processor = TextProcessor()
    if vocab:
        text_processor.tokenizer.load_vocab(vocab)
    else:
        text_processor.tokenizer.build_vocab_from_rationales(RationaleGenerator())

    state = torch.load(checkpoint, map_location='cpu')
    ckpt_vocab = state['token_embedding.weight'].shape[0]
    assert ckpt_vocab == text_processor.tokenizer.get_vocab_size(), (
        f"[onpolicy] vocab size {text_processor.tokenizer.get_vocab_size()} does not match "
        f"checkpoint embedding rows {ckpt_vocab}")

    model = ToyVLM(text_processor)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, text_processor


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vocab", type=str, default=None,
                        help="tokenizer vocab json; default rebuilds it deterministically")
    parser.add_argument("--n", type=int, default=50_000)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--out", type=str, default="onpolicy_data.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperatures", type=float, nargs='+', default=[0.0, 0.7],
                        help="cycled per batch (default: half greedy, half T=0.7)")
    parser.add_argument("--max_gen_len", type=int, default=MAX_GEN_LEN)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--stats_json", type=str, default=None,
                        help="also write the stats dict to this path")
    parser.add_argument("--star", action="store_true",
                        help="STaR / rejection-sampling: keep only fully "
                             "gold-supervised samples (exact + early_stop) in "
                             "the saved tensors, discarding corrected traces")
    parser.add_argument("--hard_frac", type=float, default=0.0,
                        help="probability a draw comes from --hard_types "
                             "instead of the difficulty mixture (0 = off; "
                             "0.8 matches grpo.py's prompt mix)")
    parser.add_argument("--hard_types", type=str, nargs='+',
                        default=['ordinal', 'compositional_h3',
                                 'compositional_h4', 'relative_count'],
                        help="failure-focused question types for --hard_frac")
    args = parser.parse_args()

    rg_types = RationaleGenerator().generators
    unknown = [t for t in args.hard_types if t not in rg_types]
    assert not unknown, (
        f"[onpolicy] unknown --hard_types {unknown}; have {sorted(rg_types)}")

    device = best_device()
    print(f"Loading {args.checkpoint} on {device}")
    model, text_processor = load_model(args.checkpoint, args.vocab, device)

    result = collect(model, text_processor, n_samples=args.n, batch_size=args.batch_size,
                     out_path=args.out, temperature_mix=tuple(args.temperatures),
                     seed=args.seed, max_gen_len=args.max_gen_len,
                     num_examples=args.examples, star=args.star,
                     hard_frac=args.hard_frac, hard_types=args.hard_types)
    print_report(result)

    if args.stats_json:
        with open(args.stats_json, 'w') as f:
            json.dump(result['stats'], f, indent=2)


if __name__ == "__main__":
    main()
