"""Ordinal error decomposition: where do a checkpoint's ordinal chains break?

An example of the cheapest useful encoder probe: the model's own chain IS an
encoder readout. Every enumeration step states a perceived (color, shape, row,
col); comparing the chain step-by-step against the deterministic gold trace and
classifying the FIRST divergence separates failure hypotheses that a plain
accuracy number cannot.

The classifier targets the run-11 rank-grouped trace format (questions.
generate_ordinal_qa): an ascending sweep whose lines carry a dense-rank label
('rank 3 blue cross at row 2 col 3'), then '{R} ranks', then '{ordinal} from
{side} is rank {j}', then 'rank {j} is {shapes}'. Four divergence classes over
the enumeration and one per closing step:

  coord_off1     enum fact, same color+shape, row or col off by exactly one
                 -> per-object localization error (the "crowding" signature)
  identity       enum fact, same cell, wrong shape and/or color
                 -> per-object recognition error (the "pixels-per-shape" signature)
  order          enum fact names a REAL object with correct attributes, but not
                 the sweep-next one -> the sweep-during-generation failed, not
                 perception
  label_err      enum fact exactly right, its 'rank {n}' label wrong (or absent)
                 -> the grouping, not the sweep and not perception
  enum_other     any other enumeration divergence (missing / extra / truncated)
  rank_count_err enumeration EXACTLY matches gold, '{R} ranks' wrong
  conversion_err right through the rank count, '{ordinal} from {side} is rank
                 {j}' names the wrong j -> the side-to-sweep mapping
  readoff_err    right through the conversion, 'rank {j} is {shapes}' wrong
                 -> pure retrieval failure with a perfect enumeration
  answer_copy    chain exactly gold, answer token wrong

History -- run 10 (2026-08-18, n=300, greedy) under the PREVIOUS format, whose
enumeration was sorted from the queried side and which closed with a single
'{ordinal} from {side} is {shapes}' step: coord_off1 0.3%, identity 0%, order
29.7%, rank_read (that format's single closing step) 15.0%; per-step
enumeration error rate rose with scene density (0.037 at N<=3 -> 0.128 at
N>=10) and tie-group target ranks scored 29% vs 69% for singletons. Split by
side, the order-error rate was 0.00 on the ascending 'top' sweep (n=74) against
0.53 right / 0.34 bottom / 0.28 left. Conclusion: the encoder's per-object
readout is clean; that format was bounded by the reversed axis sweeps and by
tie-group read-off, not by perception. (It also refined the three-arm result,
where no update rule moved ordinal more than +0.9 points.)

Run 11 changes the format on exactly those two findings -- one shared ascending
sweep, and the grouping/read-off written out as text -- so the run-10 numbers
above are history, not a baseline this classifier reproduces: 'rank_read' no
longer exists and splits into rank_count_err / conversion_err / readoff_err.
The by-side split stays worth reading even though the sweep is now shared:
conversion and read-off are still per-side work.

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

# Decode steps per stage. A run-11 ordinal chain runs to 141 tokens on a
# 12-object scene (9 words per enumeration line plus its separator), so
# generate_response's 80-step default would cut half of them off mid-enumeration
# and this probe would read its own truncation as an enumeration failure. The
# real bound is the sequence budget, which leaves ~180 rationale tokens after
# the 67-token prefix; this is that with room to spare. evaluate.py and
# onpolicy.py default to the same 160, for the same reason.
MAX_GEN_LEN = 160


def bucket_of(n_objects: int) -> str:
    lo = ((n_objects - 1) // BUCKET_WIDTH) * BUCKET_WIDTH + 1
    return f"{lo}-{lo + BUCKET_WIDTH - 1}"


# Steps that close an ordinal trace after the enumeration: '{R} ranks',
# '{ordinal} from {side} is rank {j}', 'rank {j} is {shapes}'.
CLOSING_STEPS = 3


def classify(gold_steps, model_steps, metadata):
    """(class, first-divergence index) for one chain vs its gold trace.

    The three closing steps are classified by position, so a divergence there is
    attributed to the step that produced it and not to the answer. A model that
    runs past the read-off (extra steps after a complete gold chain) lands in
    readoff_err as well: the read-off segment is what it got wrong.

    Inside the enumeration the fact is compared without its label and the label
    separately, so label_err means "the right object, mis-grouped"; a line that
    gets both the fact and the label wrong is attributed to the fact, which is
    the earlier failure of the two.
    """
    n_enum = len(gold_steps) - CLOSING_STEPS
    k = min(len(gold_steps), len(model_steps))
    idx = next((i for i in range(k) if gold_steps[i] != model_steps[i]),
               None if len(gold_steps) == len(model_steps) else k)
    if idx is None:
        return 'correct', None
    if idx >= n_enum + 2:
        return 'readoff_err', idx
    if idx == n_enum + 1:
        return 'conversion_err', idx
    if idx == n_enum:
        return 'rank_count_err', idx
    gold = parse_fact(gold_steps[idx])
    step = model_steps[idx] if idx < len(model_steps) else None
    fact = parse_fact(step) if step is not None else None
    if fact is None:
        return 'enum_other', idx
    _, g_color, g_shape, g_row, g_col = gold
    _, m_color, m_shape, m_row, m_col = fact
    if (m_color, m_shape, m_row, m_col) == (g_color, g_shape, g_row, g_col):
        # The fact is right, so the lines can only differ in what stands in
        # front of it: a wrong 'rank {n}', or no label at all.
        return 'label_err', idx
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
    parser.add_argument("--max_gen_len", type=int, default=MAX_GEN_LEN,
                        help="decode steps per stage; the run-11 ordinal trace "
                             "needs up to 141 for a 12-object scene, so the "
                             "80-step default of generate_response would report "
                             "truncation as enum_other on half the draws")
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
    by_side = defaultdict(Counter)
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
        rationale, model_answer = generate_response(model, img, question,
                                                    max_length=args.max_gen_len)
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
        if kind in ('coord_off1', 'identity', 'order', 'label_err', 'enum_other'):
            by_n[bucket]['enum_err'] += 1

        # Per-step enumeration error rate: how often does the NEXT enumeration
        # step go wrong, given the chain was right so far? Rising with N means
        # the sort itself gets harder in dense scenes; flat means plain
        # compounding over longer chains.
        n_enum = len(gold_steps) - CLOSING_STEPS
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

        # Per-side split. The sweep is now shared -- every side enumerates
        # ascending -- so a residual order-error gap between sides would mean
        # the sweep is being conditioned on the question after all. What may
        # legitimately differ by side is the work after it: 'left'/'top'
        # convert with j = k, 'right'/'bottom' have to mirror.
        by_side[side]['n'] += 1
        by_side[side]['acc'] += hit
        by_side[side]['order_err'] += int(kind == 'order')
        by_side[side]['conv_err'] += int(kind == 'conversion_err')
        by_side[side]['readoff_err'] += int(kind == 'readoff_err')

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

    print("\nby side: accuracy | order / conversion / read-off error shares")
    for side in ('top', 'bottom', 'left', 'right'):
        c = by_side[side]
        if c['n']:
            print(f"  {side:>6}: acc {c['acc'] / c['n']:.2f}"
                  f"  order-err {c['order_err'] / c['n']:.2f}"
                  f"  conv-err {c['conv_err'] / c['n']:.2f}"
                  f"  readoff-err {c['readoff_err'] / c['n']:.2f}  (n={c['n']})")


if __name__ == "__main__":
    main()
