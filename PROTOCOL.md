# Three-arm comparison: STaR vs DAgger vs GRPO

Which update rule best closes a verified reasoning gap at fixed sampling
budget? Same base model, same failure distribution, same rollout budget;
only the way the model's own samples become gradients differs.

## Base: run 10

Full retrain from scratch with the question surface variants AND the
polarity variants (odd/fewer/less-than forms whose answer inverts against
an unchanged trace) — run-9 recipe otherwise: 20 epochs, mixed
curriculum, dense-rank ordinal semantics, equal loss weights. All arms
start here, so no arm spends its budget learning surface forms instead of
reasoning. Since run 10 trains from scratch, the vocabulary may grow
(expected: +'fewer'), and tokenizer_vocab.json is regenerated in-repo
ahead of training. Run 9 is never fine-tuned and its GUI pairing is
intentionally retired: the vocab assert in test_model.py fails loudly
against toy_vlm_cot.pth until the run-10 checkpoint replaces it (the old
pair lives in git history if ever needed).

Baseline eval at 240/type establishes the pre-arm numbers (expect ordinal
near run 9's 62.5%) and doubles as a first read on whether a 5M model
learns answer-polarity inversion — reading the asked polarity against the
trace's stated fact — from scratch, before any arm touches it. Report
parity/comparison/side_count accuracy split by asked polarity.

## Arms (equal rollout budget: ~150k sampled chains each)

Shared collection distribution: 80% hard types (ordinal,
compositional_h3, compositional_h4, relative_count), 20% all types;
rationale stage at T=1.0. Implemented: onpolicy.py --hard_frac 0.8 (arms
A/B) mirrors grpo.py's prompt mix (arm C), and collection now reports
per-type disposition stats. A 96-draw smoke on run 9 at T=1.0 already
shows nonzero exact-chain fractions on every hard type (ordinal 0.52,
h3 0.38, relative_count 0.65, h4 0.14), so arm A is not starved.

**A — iterated STaR** (rejection-sampling SFT): 3 rounds x 50k rollouts.
Each round: collect with `--star` (keep only fully-verified chains),
fine-tune via finetune_onpolicy.py with gold-data replay, re-collect with
the improved policy. Round-1 disposition stats double as the per-type
pass@1 measurement at T=1.0 — if the kept fraction on ordinal failures is
~0, this arm is starved by construction and that is itself a result.

**B — DAgger** (corrections): identical collection without `--star`
(corrected trajectories kept: gold prefix + model's wrong step,
unsupervised + `wait` + gold tail), same rounds, same fine-tune recipe.

**C — GRPO** (policy gradient): grpo.py, ~1150 steps x (16 prompts x 8
rollouts) ≈ 147k rollouts. Reward `dense` (0.5 x gold-prefix fraction +
0.5 x answer EM): the overfit smoke showed the binary `exact` reward
(chain == gold AND answer correct, complete oracle verification) starves
on this base model — its policy entropy is so low (0.03-0.05 nats/token)
that groups go all-right or all-wrong and the gradient dies (0% live
groups at plateau vs 88-100% under dense). Run `exact` as a same-budget
ablation if time allows. Group-mean advantage, no critic, exact
token-level KL to the run-10 reference, beta 0.02, lr 1e-5. Rollout
temperature stays exactly 1.0 — the sampler tempers logits but the
gradient scores the untempered policy, so any other value is an
uncorrected off-policy update (grpo.py warns).

Known asymmetry, reported not hidden: rollout budget is the controlled
variable; gradient-step counts differ by arm (SFT epochs vs RL steps) and
are reported alongside results. Forgetting guard differs in kind but
matches in intent: arms A/B replay fresh gold data during fine-tuning;
arm C keeps 20% all-type prompts in the RL mix plus the KL anchor.

## Evaluation (identical for all arms)

Final-checkpoint protocol (no best-checkpoint selection — see HISTORY.md
phase 3), 240/type, seed 0:

- headline: dense-rank ordinal answer EM (base ~62.5%; target 80%+)
- faithfulness: h3/h4 rationale exact-match closing toward answer EM
- guard: overall EM and easy/medium types must not regress
  (catastrophic-forgetting check)
- provenance: every checkpoint carries its meta file (grpo_meta.json /
  best_epoch.txt convention)

## Compute

A100 spot (lambda.ai or GCS), archive per run-prefix convention to
gs://toy-cot-models/. Rough budget: run-10 retrain ~1-2 GPU-h, each arm
~1-2 GPU-h; total well under $15.

## Order of operations

1. DONE — surface variants (71bab07) + polarity variants (4f04ad7),
   validator-verified, committed.
2. DONE — grpo.py (f004de5), five-part verification on MPS, committed.
3. DONE — per-type collection filter in onpolicy.py (61a14f2).
4. Cloud: run 10 retrain -> baseline eval (incl. per-polarity split) ->
   arms A, B, C -> final evals.
