"""Overfit smoke test for the toy VLM's real training pipeline.

Trains on a small, fixed set of mixed-difficulty samples (drawn once at
startup) through the ACTUAL ShapeDataset -> ToyVLM -> compute_weighted_loss
pipeline used by train_model.py, then checks that the model can memorize
them: teacher-forced answer exact-match and rationale token accuracy must
both cross a high threshold within a bounded number of optimizer steps.

This is a fast end-to-end sanity check that data generation, tokenization,
masking, the model's forward/backward pass, and the loss are all wired up
correctly. It is not a measure of generalization -- see evaluate.py for that.

Usage:
    python test_force.py [--steps 500] [--samples 64] [--seed 1] [--cpu]
"""

import argparse
import random
import sys
import time

import numpy as np
import torch

from questions import RationaleGenerator
from text import TextProcessor
from model import ToyVLM, generate_response
from train_model import ShapeDataset, collate_fn
from utils_loss import compute_weighted_loss

ANSWER_EM_THRESHOLD = 0.98
RATIONALE_ACC_THRESHOLD = 0.95


def best_device(force_cpu: bool) -> torch.device:
    """Pick the best available device: cuda -> mps -> cpu (or cpu if forced)."""
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500, help="optimizer steps")
    parser.add_argument("--samples", type=int, default=64, help="fixed samples to memorize")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cpu", action="store_true", help="force CPU even if CUDA/MPS is available")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = best_device(args.cpu)

    rg = RationaleGenerator()
    tp = TextProcessor()
    tp.tokenizer.build_vocab_from_rationales(rg)

    # A fixed batch of mixed-difficulty samples, generated once via the real
    # dataset/collate path so this exercises exactly what train_model.py does.
    ds = ShapeDataset(args.samples, tp, mixture={'easy': 1 / 3, 'medium': 1 / 3, 'hard': 1 / 3})
    items = [ds[i] for i in range(args.samples)]
    batch = collate_fn(items)
    images = batch['image'].to(device)
    inp = batch['input_tokens'].to(device)
    tgt = batch['target_tokens'].to(device)
    rat_m = batch['rat_mask'].to(device)
    ans_m = batch['ans_mask'].to(device)
    aux_labels = batch['aux_labels']

    model = ToyVLM(tp).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
    weights = {'rationale': 1.0, 'answer': 1.0, 'aux': 0.3}

    ans_em = rat_acc = 0.0
    t0 = time.time()
    for step in range(args.steps):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits, aux = model(images, inp, return_aux=True)
        loss, rl, al, xl = compute_weighted_loss(logits, tgt, rat_m, ans_m, aux, aux_labels, tp.tokenizer, weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 100 == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                match = model(images, inp).argmax(-1) == tgt
                am, rm = ans_m.bool(), rat_m.bool()
                ans_em = ((match | ~am).all(1) & am.any(1)).float().mean().item()
                rat_acc = ((match & rm).sum() / rm.sum().clamp_min(1)).item()
            print(f"step {step:3d} loss {loss.item():.4f} (rat {rl.item():.3f} ans {al.item():.3f} "
                  f"aux {xl.item():.3f}) TF answer EM {ans_em:.3f} rat tok acc {rat_acc:.3f}")

    print(f"({time.time() - t0:.0f}s on {device})")

    # Free-running generation on a few memorized samples (diagnostic, not a gate).
    for i in range(min(3, args.samples)):
        rat, ans = generate_response(model, batch['image'][i].to(device), batch['question'][i])
        print(f"[gen] q={batch['question'][i]!r}\n  gold  rat={batch['rationale'][i]!r} ans={batch['answer'][i]!r}"
              f"\n  model rat={rat!r} ans={ans!r}")

    passed = ans_em >= ANSWER_EM_THRESHOLD and rat_acc >= RATIONALE_ACC_THRESHOLD
    if passed:
        print(f"\nPASS: overfit smoke test (answer EM {ans_em:.3f} >= {ANSWER_EM_THRESHOLD}, "
              f"rationale token acc {rat_acc:.3f} >= {RATIONALE_ACC_THRESHOLD})")
        return 0

    print(f"\nFAIL: overfit smoke test (answer EM {ans_em:.3f}, rationale token acc {rat_acc:.3f}; "
          f"need >= {ANSWER_EM_THRESHOLD} and >= {RATIONALE_ACC_THRESHOLD})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
