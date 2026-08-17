# toy-cot: Project History

A toy vision-language model with chain-of-thought reasoning over 2D shape
scenes, built as a small-scale parallel to frontier ideas. This document
records the full arc — including the failures, which taught more than the
successes. Compressed into one intensive working session (2026-08-15/16):
nine training runs, five probe studies, one project-defining bug, ~$15 of
spot GPU.

Companion project: [toy-vlm](https://github.com/alexlinde/toy-vlm) — the
single-shape demonstrator this project extends.

---

## Phase 0 — Review: why the original attempt failed

The starting codebase could not succeed, and could not know it was failing:

- **It was unmeasurable.** `evaluate.py` had a dead import (broken for the
  last 3 commits); no seeds, no validation, no logs.
- **It crashed.** Aux count heads had 5 classes against scenes of up to 8
  objects (guaranteed mid-training IndexError); a shadowed `device_type`
  requested fused AdamW on MPS.
- **The CoT supervision was broken where it mattered.** Relational questions
  had constant-string rationales carrying zero information; one template's
  rationale was factually wrong ~50% of the time (an in-code
  `todo: fix this, it's wrong` admitted it).
- **The training regime fought CoT**: exclusive easy→medium→hard phases
  (catastrophic forgetting) and a loss schedule that decayed rationale
  weight while boosting answer weight — teaching the model to ignore its
  own chain.

## Phase 1 — Rebuild

- **Data**: RGB scenes (4 shapes × 4 colors × 3 sizes, up to 12 objects),
  uniform *enumerate-then-reason* traces ("red circle at row 1 col 5 .
  count is 2") with all ground truth on a quantized grid so traces are
  consistent by construction; ~50/50 answer balancing; deterministic
  enumerated vocabulary (OOV impossible).
- **Model**: image-first token layout with prefix-LM attention
  (bidirectional over the image block), ~5M params total.
- **Training**: mixed-difficulty mixture curriculum, equal rationale/answer
  loss weights, perception warm-up, seeding, per-epoch teacher-forced
  validation, JSONL logs, checkpoints.
- **Verification culture** (the phase's real product): `validate_traces.py`
  independently re-derives ground truth and checks every stated fact in
  every rationale; later extended with **mutation testing** (deliberately
  corrupted traces must be rejected — including "consistent-but-wrong"
  mutants only detectable by recomputing ground truth).

## Phase 2 — The false-plateau era (every conclusion here was later overturned)

Recorded honestly because the reasoning was sound and the experiments well
run — on contaminated measurements.

| Experiment | Apparent result |
|---|---|
| CoT vs direct ablation | **Direct wins 85.5% vs 70.1%** — chains compound perception errors |
| Self-consistency (k=1/5/9 voting) | +1.4 pts only → "errors are systematic, not stochastic" |
| Crossover (12 objects, 13 types) | No crossover; direct still wins 68.5 vs 58.6 |
| Correction training (oracle-corrupted traces) | +2 pts; model never says "wait" at inference |
| Encoder probes | "CoT starves perception" — CoT encoder appears near shape-blind (30%) |
| Encoder transplant (frozen & unfrozen) | Good eyes don't transfer; ~59% plateau persists |
| Diagnosis | "Pure exposure bias" — teacher-forced 99.9%, free-running 59% |

Four different CoT arms landed on the same ~59% plateau. Three successive
theories (systematic misperception → starved encoder → exposure bias)
each fit the data. All three were wrong.

## Phase 3 — The checkpoint bug

**Mechanism**: best-checkpoint selection used strict `>` on validation
answer exact-match. That metric saturates at exactly 1.0 within 1–2 epochs
(the answer is a near-copy given a gold rationale), and `1.0 > 1.0` is
false — so `best.pth` silently froze at ~epoch 1 while training continued
for 19 more. Every evaluation of every CoT arm scored a barely-trained
model. The direct baseline was unaffected (it genuinely converges early),
which made the comparison look credible.

**Discovery**: an agent building the next pipeline measured teacher-forced
accuracy of its input checkpoint as routine hygiene — 0.80, against 99.9%
in the training log for "the same" model. Two numbers about one file can't
disagree; therefore two files.

**Corrected results (final checkpoints, 240/type)**:

| Arm | Overall | Rationale exact |
|---|---|---|
| direct (no-CoT) | 68.3% | — |
| **plain CoT** | **96.1%** | 85.7% |
| CoT + corrections | 91.4% | 80.1% |
| frozen transplant | 92.6% | 77.2% |
| unfrozen transplant | 94.9% | 84.9% |

Plain CoT beats direct by **+27.8 points** — the original thesis — and
every clever intervention scored *below* plain CoT. The encoder finding
**inverted** on re-probe: the converged CoT encoder (87% per-cell shape)
*beats* the direct model's (80%). CoT trains perception slower but to a
better endpoint: per-object grounded enumeration is richer spatial
supervision than a single answer token.

**Fixes**: `>=` with a non-saturating tiebreak (rationale token accuracy),
provenance written to `checkpoints/best_epoch.txt`, and a
final-checkpoint eval protocol.

## Phase 4 — Consolidation and the semantics experiments

**Run 7** (94.2%, 14 types): added `relative_count`
("how many circles are below the red triangle") with **clarification
behavior** — on an ambiguous referent the model answers "which triangle"
(85.8% vs 20.8% baseline, verified live). Slimmed the cross (small crosses
were near-identical to circles: IoU vs circle 0.81→0.65; counting hit 100%
with 100% faithful chains). Gap-banded sizes (contiguous bands had made
boundary sizes pixel-identical but differently labeled).

**Run 8** (95.1%, the score champion): +4 epochs ≈ flat (recipe converged
at 20); chains improved (h3 92.9%).

**Ordinal walk experiment (falsified hypothesis)**: aligning ordinal's
trace to raster order with an explicit sorted walk made it *worse*
(79.6→74.2). Mechanism: the old axis-sorted enumeration let the model sort
*perceptually* (reading the image in sorted order while generating); the
walk demanded a *symbolic* sort over its own written tokens — harder.
Reasoning is easiest when generation order matches computation order.

**Dense-rank ordinal (run 9, 92.4%, the honest champion)**: "fourth from
the right" now means the 4th occupied column, and ties are answered as
groups ("circle and cross") instead of resolved by an arbitrary unstated
tie-break. Result: ordinal *dropped* to 62.5% — the human-honest semantics
are genuinely harder (group enumeration + multi-shape answers) than the
learnable-but-arbitrary convention. A values call: run 9 ships in the GUI
because a demo shouldn't defend "the higher circle counts as fourth."

**STaR round 1 (null result)**: rejection-sampling fine-tune on the
model's own fully-fact-verified chains kept 89.8% of 50k — meaning the
signal was dominated by what the model already does right, and the
failures it needed were exactly the discarded 10%. Overall flat
(92.4→92.6); h3/h4 faithfulness +1.7 within noise.

## Perception science (all checkpoint-independent, all stand)

- **Factorial (canvas × count × size)**: total counting is *never* the
  constraint (100% everywhere); canvas size and crowding barely matter;
  **pixels-per-shape governs shape identity** — small shapes give superb
  cell attribution but identity confusions wreck per-shape tallies at
  high N; large shapes invert that (counts ~99%, cell attribution drops as
  shapes straddle cells).
- **Rotation** halves shape discrimination at 8–15px (toy-vlm's rotation
  benefit doesn't transfer down in scale). Kept off.
- **Training noise** leaves shape features untouched but adds +21 pts to
  count aggregation. Kept on.

## Lessons (presentation candy)

1. **Measurement infrastructure is the product.** The eval/validator suite
   both enabled every finding and eventually caught its own biggest error.
2. **A saturated metric cannot select checkpoints.** Strict `>` + exact
   ties froze "best" at epoch 1; three sophisticated wrong theories grew in
   the gap between the log (final model) and the eval (frozen model).
3. **Artifacts cascade.** One bug manufactured a plateau, which motivated
   five interventions, each "failing" against the same artifact.
4. **CoT supervision must carry computation** (enumerate-then-reason), and
   at this scale it also trains *better perception* than direct answering.
5. **Generation order should match computation order** — forcing "show
   your sorting work" symbolically made sorting worse.
6. **Honest semantics ≠ easier semantics.** Removing an arbitrary
   convention (tie-breaks) lowered scores while raising answer quality.
7. **Single-round STaR reinforces confidence, not competence.** Rejection
   sampling must be iterated and hard-focused, or it re-teaches the known.
8. **Verify subagent work, and let subagents verify yours** — the bug fell
   to routine hygiene, not brilliance.

## Current state

- **GUI champion**: run 9 (92.4%, honest semantics, clarification skill),
  `uv run python test_model.py`; scenes carry shareable seeds.
- **Score champion**: run 8 (95.1%) archived alongside all runs in
  `gs://toy-cot-models/` (per-run prefixes, essentials tarballs).
- Both spot instances (L4, A100) stopped; disks retained.
- Residual gaps: ordinal under dense-rank semantics (62.5%), deep-chain
  faithfulness (h4 rationale exact ~41%), h3/h4 accuracy ~84%.

## Proposed next step: iterated, hard-focused STaR

One round of gentle rejection sampling was a null; the frontier version of
the recipe differs in exactly the ways round 1 lacked:

1. **Collect on the failure distribution**: hard-bucket questions only
   (h3/h4, ordinal, relative_count), higher temperature (1.0), larger k
   per prompt — hunting for the rare verified chains on questions the
   model usually misses.
2. **Iterate**: fine-tune → re-collect with the improved policy → repeat
   2–3 rounds (each round's newly-solvable questions feed the next).
3. **Keep the DAgger arm as a comparison**: corrected trajectories
   (`--star` off) on the same failure distribution — corrections vs
   rejection as the update rule, same budget.
4. **Success metric**: h3/h4 rationale exact-match closing toward answer
   accuracy (chains as true as they are fluent), and dense-rank ordinal
   recovering toward 80%+ without semantics changes.

Infrastructure required: none — `onpolicy.py --star`, disposition
filtering, and `finetune_onpolicy.py` are built and verified; only a
per-type collection filter (~20 lines) is new.
