"""Ordinal error decomposition: where do a checkpoint's ordinal chains break?

An example of the cheapest useful encoder probe: the model's own chain IS an
encoder readout. Every enumeration step states a perceived (color, shape, row,
col); comparing the chain step-by-step against the deterministic gold trace and
classifying the FIRST divergence separates failure hypotheses that a plain
accuracy number cannot:

  coord_off1   enum fact, same color+shape, row or col off by exactly one
               -> per-object localization error (the "crowding" signature)
  identity     enum fact, same cell, wrong shape and/or color
               -> per-object recognition error (the "pixels-per-shape" signature)
  order        enum fact names a REAL object with correct attributes, but not
               the axis-next one -> the sort-during-generation failed, not
               perception
  enum_other   any other enumeration divergence (missing / extra / truncated)
  rank_read    enumeration EXACTLY matches gold, final rank step wrong -> pure
               symbolic read-off failure with perfect perception
  answer_copy  chain exactly gold, answer token wrong

Run against run 10 (2026-08-18, n=300, greedy): coord_off1 0.3%, identity 0%,
order 29.7%, rank_read 15.0%; per-step enumeration error rate rises with scene
density (0.037 at N<=3 -> 0.128 at N>=10) and tie-group target ranks score 29%
vs 69% for singletons. Conclusion: the encoder's per-object readout is clean;
dense-rank ordinal is bounded by the axis-sort computation and by tie-group
read-off, not by perception. (This refined the three-arm result, where no
update rule moved ordinal more than +0.9 points.)

Usage:
    uv run python probe_ordinal.py --checkpoint toy_vlm_cot.pth \
        --vocab tokenizer_vocab.json --n 300 --seed 7
"""

import argparse
import random
from collections import Counter, defaultdict

import torch

from model import generate_response
from onpolicy import best_device, load_model
from questions import (ORDINALS, RationaleGenerator, axis_groups, cell_of,
                       parse_fact, split_steps)
from shapes import MAX_OBJECTS, MIN_OBJECTS, ShapeGenerator

# Scene-density buckets: 1-3, 4-6, 7-9, 10-12 objects.
BUCKET_WIDTH = 3


def bucket_of(n_objects: int) -> str:
    lo = ((n_objects - 1) // BUCKET_WIDTH) * BUCKET_WIDTH + 1
    return f"{lo}-{lo + BUCKET_WIDTH - 1}"


def classify(gold_steps, model_steps, metadata):
    """(class, first-divergence index) for one chain vs its gold trace."""
    n_enum = len(gold_steps) - 1          # the last gold step is the rank read
    k = min(len(gold_steps), len(model_steps))
    idx = next((i for i in range(k) if gold_steps[i] != model_steps[i]),
               None if len(gold_steps) == len(model_steps) else k)
    if idx is None:
        return 'correct', None
    if idx >= n_enum:
        return 'rank_read', idx
    gold = parse_fact(gold_steps[idx])
    fact = parse_fact(model_steps[idx]) if idx < len(model_steps) else None
    if fact is None:
        return 'enum_other', idx
    _, g_color, g_shape, g_row, g_col = gold
    _, m_color, m_shape, m_row, m_col = fact
    if (m_color, m_shape) == (g_color, g_shape) \
            and abs(m_row - g_row) + abs(m_col - g_col) == 1:
        return 'coord_off1', idx
    if (m_row, m_col) == (g_row, g_col) and (m_color, m_shape) != (g_color, g_shape):
        return 'identity', idx
    if any(m['color'] == m_color and m['shape'] == m_shape
           and cell_of(m) == (m_row, m_col) for m in metadata):
        return 'order', idx
    return 'enum_other', idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="toy_vlm_cot.pth")
    parser.add_argument("--vocab", type=str, default="tokenizer_vocab.json")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    device = best_device()
    print(f"Loading {args.checkpoint} on {device}")
    model, text_processor = load_model(args.checkpoint, args.vocab, device)

    shape_gen = ShapeGenerator()
    rationale_gen = RationaleGenerator()

    kinds = Counter()
    by_n = defaultdict(Counter)
    tie_split = defaultdict(Counter)
    step_errs = Counter()   # samples whose enumeration diverged, per bucket
    step_tot = Counter()    # enumeration steps generated before divergence

    got = 0
    while got < args.n:
        num = random.randint(MIN_OBJECTS, MAX_OBJECTS)
        image, metadata = shape_gen.generate_multi_shape_image(num, False)
        question, answer, gold = rationale_gen.generate_ordinal_qa(metadata)
        if question is None:
            continue
        got += 1

        img = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        rationale, model_answer = generate_response(model, img, question)
        gold_steps = split_steps(gold)
        model_steps = split_steps(rationale.strip())

        kind, idx = classify(gold_steps, model_steps, metadata)
        hit = model_answer.strip().lower() == answer
        if kind == 'correct' and not hit:
            kind = 'answer_copy'
        kinds[kind] += 1

        bucket = bucket_of(len(metadata))
        by_n[bucket]['n'] += 1
        by_n[bucket]['acc'] += hit
        if kind in ('coord_off1', 'identity', 'order', 'enum_other'):
            by_n[bucket]['enum_err'] += 1

        # Per-step enumeration error rate: how often does the NEXT enumeration
        # step go wrong, given the chain was right so far? Rising with N means
        # the sort itself gets harder in dense scenes; flat means plain
        # compounding over longer chains.
        n_enum = len(gold_steps) - 1
        first_err = idx if (idx is not None and idx < n_enum) else n_enum
        step_errs[bucket] += int(first_err < n_enum)
        step_tot[bucket] += first_err + int(first_err < n_enum)

        # Tie-group vs singleton target rank (every question surface form ends
        # 'from the {side}' and contains exactly one ordinal word).
        side = question.rsplit(' ', 1)[-1]
        rank = next(i for i, w in enumerate(ORDINALS) if w in question.split())
        group = axis_groups(metadata, side)[rank]
        key = 'tie' if len(group) > 1 else 'singleton'
        tie_split[key]['n'] += 1
        tie_split[key]['acc'] += hit

    total_acc = sum(c['acc'] for c in by_n.values()) / got
    print(f"\nn={got}  answer EM overall: {total_acc:.3f}\n")
    print("first-divergence class:")
    for kind, count in kinds.most_common():
        print(f"  {kind:<12} {count:>4}  ({100 * count / got:.1f}%)")

    print("\nby N: accuracy | enum-error share | per-step enum error rate")
    for bucket in sorted(by_n, key=lambda b: int(b.split('-')[0])):
        c = by_n[bucket]
        rate = step_errs[bucket] / max(1, step_tot[bucket])
        print(f"  {bucket:>5}: acc {c['acc'] / c['n']:.2f}"
              f"  enum-err {c['enum_err'] / c['n']:.2f}"
              f"  per-step {rate:.3f}  (n={c['n']})")

    print("\ntarget rank: " + '  '.join(
        f"{k}: {v['acc'] / v['n']:.2f} (n={v['n']})"
        for k, v in sorted(tie_split.items())))


if __name__ == "__main__":
    main()
