# Toy VLM with Chain-of-Thought Reasoning

A small, from-scratch vision-language model that answers questions about
2D shape scenes by first writing out a step-by-step rationale and then a
final answer -- a frontier-parallel design (inline image tokens, prefix-LM
attention, enumerate-then-reason CoT, mixed-difficulty curriculum, auxiliary
perception heads) shrunk down to a toy scale that trains in minutes to hours
on a single GPU.

## Architecture

- **~5.0M parameters total** (~140K in the vision encoder, ~86K in the
  auxiliary count heads, the rest in the transformer decoder and tied
  embeddings).
- **6-layer, 4-head, 256-dim** pre-LN transformer decoder (`model.py`).
- **Vision encoder**: a small CNN with always-on (x, y) coordinate channels
  and a squeeze-and-excitation block, producing exactly 64 image tokens on
  an 8x8 grid with learned 2D positional embeddings.
- **Image-first, fixed-position layout**: the 64 image tokens are spliced
  into the token sequence at fixed positions 2..65 (`IMG_POS_START=2`), so
  the model never has to locate them dynamically:
  ```
  [BOS] <IMG_START> <IMG>x64 <IMG_END> <|user|> question <|assistant|>
  <THINK> rationale </THINK> <FINAL> answer </FINAL> [EOS]
  ```
- **Prefix-LM attention**: bidirectional self-attention within the image
  prefix (positions 0..66), causal thereafter.
- **`MAX_SEQ_LEN = 192`**; a 73-token deterministic vocabulary built from
  the question generator's declared word set (`text.py`,
  `tokenizer.build_vocab_from_rationales`) -- training can never see an
  out-of-vocabulary token.
- **Weight-tied output head**: the LM head shares weights with the token
  embedding.
- **Auxiliary heads**: per-shape, per-size, and per-color count classifiers
  (0..`MAX_OBJECTS` classes each) over pooled vision tokens, used only as an
  auxiliary loss to help the vision encoder disentangle features.

## Data

Scenes are synthesized on the fly (`shapes.py`): 64x64 RGB images, 1-6
objects per scene (`MIN_OBJECTS`/`MAX_OBJECTS`), each object one of
**4 shapes** (square, circle, cross, triangle) x **4 colors** (red, green,
blue, yellow) x **3 sizes** (small, medium, large), placed without overlap.
All spatial ground truth is derived from a quantized 8x8 grid (`grid_row`/
`grid_col`), never from raw pixel coordinates.

`questions.py` generates 8 question types across 3 difficulty tiers
(`DIFFICULTY_MAP`), each with an *enumerate-then-reason* CoT rationale:
first enumerate the objects the question needs (in raster order, with
quantized grid coordinates), then perform the reasoning steps that lead to
the answer. Yes/no questions are balanced to a roughly 50/50 answer prior.

| Difficulty | Question types |
|---|---|
| easy | existence, positional_existence |
| medium | counting, size, relative_position, side_count_comparison |
| hard | comparison, compositional |

Example full training sequences (`<THINK>...</THINK> <FINAL>...</FINAL>`):

```
q: is there a red shape
<THINK> red circle at row 2 col 5 . count is 1 </THINK> <FINAL> yes </FINAL>

q: how many blue triangles are there
<THINK> blue triangle at row 6 col 5 . count is 1 </THINK> <FINAL> 1 </FINAL>

q: is a circle left of a red circle
<THINK> red circle at row 3 col 5 . blue circle at row 5 col 2 . red circle
at row 3 col 5 . col 2 is left of col 5 . found </THINK> <FINAL> yes </FINAL>

q: are there more yellow shapes than yellow squares
<THINK> yellow square at row 3 col 5 . count of yellow shapes is 1 . yellow
square at row 3 col 5 . count of yellow squares is 1 . 1 equal to 1
</THINK> <FINAL> no </FINAL>
```

Every generated sample is independently re-verified by `validate_traces.py`:
every enumerated object, every stated count, and every cited witness is
checked against freshly recomputed ground truth, with zero tolerance for
out-of-vocabulary tokens or sequences that overflow `MAX_SEQ_LEN`.

## Training

`train_model.py` implements the regime described in its own docstring:

- **Mixed-difficulty curriculum**: every epoch samples all three
  difficulties, with the mixture shifting toward harder questions over
  training (`get_difficulty_mixture`) -- never an exclusive phase.
- **Constant rationale/answer loss weights** (1.0 / 1.0); the auxiliary
  count-head weight decays 0.3 -> 0.1 over training (`get_loss_weights`).
- Optional **1-epoch perception warm-up** (aux loss only, disable with
  `--no_perception_warmup`).
- **`--no_cot`** ablation: trains with empty `<THINK>` spans (no rationale
  supervision) to isolate the effect of chain-of-thought.
- Full **seeding** (`random`/`numpy`/`torch`) for reproducibility.
- **Per-epoch teacher-forced validation** on fixed, seeded validation sets
  per difficulty (answer exact-match + rationale token accuracy).
- **JSONL logging** (`train_log.jsonl` by default) and per-epoch
  checkpoints in `checkpoints/`, plus a `checkpoints/best.pth` tracked by
  validation answer EM.

Default hyperparameters (`--epochs 20 --samples_per_epoch 100000
--batch_size 128 --lr 5e-4`) target a single cloud GPU and run fine on a
GCS or lambda.ai instance. On Apple Silicon (MPS), reduce scale, e.g.:

```bash
.venv/bin/python train_model.py --samples_per_epoch 10000 --batch_size 64 --no_compile
```

Device selection, autocast/AMP, and dataloader settings are all handled
automatically by `runtime.py` (CUDA -> MPS -> CPU).

## Usage

```bash
# Set up the environment
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Validate the data layer (no model, no torch-heavy work required)
.venv/bin/python validate_traces.py --samples 1000

# Fast end-to-end smoke test: can the model overfit 64 fixed samples?
.venv/bin/python test_force.py --steps 500 --samples 64

# Train
.venv/bin/python train_model.py

# Evaluate a trained checkpoint (per-question-type exact match + majority baselines)
.venv/bin/python evaluate.py --checkpoint toy_vlm_cot.pth --vocab tokenizer_vocab.json

# Interactive GUI: generate scenes, ask questions, see rationale + answer
.venv/bin/python test_model.py --checkpoint toy_vlm_cot.pth --vocab tokenizer_vocab.json
```

Trained artifacts (`toy_vlm_cot.pth`, `tokenizer_vocab.json`,
`checkpoints/`, `train_log.jsonl`) are gitignored -- train (or run
`test_force.py`) before running `evaluate.py` or `test_model.py`.
