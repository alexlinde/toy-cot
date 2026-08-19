# toy-cot: Project History

A toy vision-language model with chain-of-thought reasoning over 2D shape
scenes, built as a small-scale parallel to frontier ideas. This document
records the full arc — including the failures, which taught more than the
successes. Two intensive working sessions (2026-08-15/16 and 08-17/19):
eleven training runs, two three-arm fine-tuning comparisons, seven probe
studies, one project-defining bug, well under $30 of spot GPU, ending at
98.4% overall with ordinal — the stubbornest type — at 99.2%.

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

## Phase 5 — Robustness to how a person asks (run 10, second session,
2026-08-17)

A user-typed question exposed the gap: "is there a red square above the
yellow triangle" — a phrasing the generator never produced, though every
word of it occurred in training — made the model parse it as an existence
question about the *yellow triangle* and answer a confident wrong "yes"
55% of the time on scenes with no red square. On-template phrasings of
the same question: 0–8% wrong. Fixes, all data-layer:

- **Surface variants**: every type asks in 2–4 phrasings (`phrase()`
  helper, canonical ~50%), validator grammar extended with alternations.
- **Polarity variants**: parity asks even/odd, comparisons ask
  more/fewer/equal — the trace never changes, only the answer read off it,
  so the model must read the asked polarity against the stated fact
  instead of copying the trace's conclusion. One new vocab word (`fewer`).
- **Run 10** (from scratch, 93-token vocab): 91.7% overall on the much
  wider distribution. Polarity inversion learned to ~100% per polarity;
  the missing-shape bug fell 55%→2%. Ordinal reproduced at 60.4% — the
  gap survived, cleanly, as the next target.

## Phase 6 — Three update rules, round one (STaR vs DAgger vs GRPO)

Same base (run 10), same failure distribution (80% hard types), same
~150k-rollout budget per arm; only the sample→gradient rule differed
(PROTOCOL.md). Result: **STaR won, nothing cracked ordinal.**

| Arm | Overall | Ordinal | Verdict |
|---|---|---|---|
| STaR ×3 (rejection-SFT) | 92.6 (+0.9) | 61.3 (+0.9) | only net-positive arm; h4 rationale-exact +6.2 |
| DAgger ×3 (corrections) | 90.3 (−1.4) | 52.9 (−7.5) | net-negative: instilled spontaneous `wait` (23% of ordinal chains) with no error-detector — false alarms broke correct chains |
| GRPO (policy gradient) | 91.5 (−0.2) | 59.6 (−0.8) | flat, but underpowered: final KL to reference 0.003 — the policy barely moved |

The three-way null on ordinal (best +0.9) was the pre-registered
signature that the bottleneck was not the training signal.

## Phase 7 — The probe: chains are an encoder readout

`probe_ordinal.py` (kept in the repo as the worked example): classify each
sampled chain's *first divergence* from the deterministic gold trace, and
failure hypotheses separate that a plain accuracy number cannot.
Run 10, n=300: coordinate off-by-ones 0.3%, identity confusions 0% —
**perception exonerated**. The real failures: *order* (29.7% — a real
object, right attributes, wrong sweep position) and *rank read-off* (15%
— wrong final step after a PERFECT enumeration; tie-group targets 29% vs
69% singleton). Split by side: the ascending 'top' sweep (= raster order,
the direction every other type trains) had a **0.00** order-error rate;
the reversed sweeps carried it all (right 0.53, bottom 0.34, left 0.28).

## Phase 8 — Run 11: the trace format was the bottleneck

The rank-grouped raster-sweep ordinal trace carries both failing
computations as text: one ascending sweep per axis whatever side is asked
(`rank 3 blue cross at row 2 col 3` — dense-rank labels, local
copy-or-increment), then `4 ranks`, an explicit conversion (`fourth from
right is rank 1`, j = R−k+1), and a read-off that retrieves the exact
bigram the enumeration wrote. Questions and answers unchanged
(equivalence: 3000 draws, 0 mismatches); vocab +`rank`/`ranks`; decode
cap 80→160 (the new traces run to 141 tokens).

**Ordinal 60.4 → 99.2%** — 100% at every density bucket from N=3 up —
and the whole model rose to **97.9%** (h3 +8.7, relative_count +15.9,
h4 rationale-exact 47→78). Two confounds checked and dismissed: the
decode-cap raise explains ~none of it (run-10 re-eval at cap 160: 91.4 vs
91.7), so the spillover is real training dynamics — removing ordinal's
~40%-noise gradient from the hard bucket helped every other hard type.
Post-probe: conversion, rank-count and read-off errors **all zero** in
300 chains; ties 29→98%; order 29.7→3.3%.

## Phase 9 — Three update rules, round two (from run 11): the 2×2

Same protocol from the 97.9% base; GRPO granted its fair dose (5× lr).

- **STaR wins again** — 98.4% overall, the project's best model
  (`a11_r3`); relative_count +2.9, faithfulness up everywhere, zero
  forgetting. The ranking replicates across bases.
- **DAgger's damage was dose**: from a strong base the collections carry
  few corrections, spontaneous `wait` collapsed 23%→≤3%, and the arm
  turned neutral (98.1). Mechanism confirmed, and retired.
- **GRPO at fair dose went negative** (96.7): reward flat at 0.889,
  faithfulness degraded (ordinal rationale-exact 96→90). From a
  near-solved base, groups are unanimous — no advantage signal, so the
  gradient is exploration noise. RL never paid on this toy at any dose.

DAgger and GRPO were then removed from the codebase; this file and the
git history are their record.

## The GUI grew an interpretability panel

Inner state the forward pass already computed, surfaced instead of
discarded: the aux count heads' scene readout (encoder belief before any
reasoning), per-token confidence tinting of the chain, top-3 answer
alternatives, and a click-any-token attention overlay on the 8×8 grid
(grounding verified: object-naming tokens' attention hits the object's
cell 34.5% exactly, 72% within one cell; row-major mapping confirmed
against the transposed alternative scoring exactly chance).

## Lessons (second session)

9. **A trace must carry every computation the answer needs.** The ranking
   was the one computation the old ordinal trace left implicit — and no
   update rule could compensate: three of them moved it ≤ +0.9 points,
   while writing the computation into the trace moved it +38.
10. **Update rules amplify what exists; they do not create.** Rejection
    sampling needs successes to keep, corrections need a detector to
    gate them, policy gradients need disagreement to grade. Feed any of
    them a format that withholds the computation and they polish the
    error.
11. **Probe by decomposition before intervening.** The chain itself is an
    encoder readout; classifying first divergences separated perception
    from sweep from read-off in one afternoon and redirected the whole
    project — away from the encoder we were about to blame.
12. **Cheap oracle, cheap science.** Every finding above leaned on the
    deterministic gold trace: exact verification, dense partial rewards,
    positional error attribution. Building the verifier first (phase 1)
    kept paying to the last day.

## Current state (end of second session, 2026-08-19)

- **Champion**: `a11_r3` (98.4% overall / 95.7% rationale-exact; STaR
  round 3 from run 11), archived in
  `gs://toy-cot-models/run-20260818-a100-arms11/`; run 11 (97.9%,
  ordinal 99.2%) is the installed GUI pair.
- **Pipeline**: STaR-only — `onpolicy.py` (rejection-sampling collection,
  hard-focused, per-type pass@1 stats) + `finetune_onpolicy.py`;
  `probe_ordinal.py` as the probe example; GUI with the
  interpretability panel.
- **Ops**: `toy-cot-ops` service account (no interactive auth), both spot
  instances stopped, instance scopes fixed for direct GCS archiving,
  linger enabled against unattended-upgrade session kills.
- Residual gaps, for whoever picks this up: h4 rationale-exact 79%
  against 94.6% answers, relative_count 97, and the N=1–2 ordinal
  oddity (95% on trivial scenes vs 100% everywhere else).
