"""
On-policy (STaR) fine-tuning for the Toy VLM.

Starts from a fully trained checkpoint and continues training on a mixture of

  * the rejection-sampled traces collected by onpolicy.py -- scenes and
    questions on which the model's OWN chain agreed with gold, trained on the
    gold trace and supervised end to end, and
  * fresh synthetic gold data from the standard pipeline, so the fine-tune does
    not drift away from the distribution it already fits.

Only the data mixture is new: the loss, the masks, the validation and the
logging are train_model's, imported rather than restated. On-policy batches
carry no scene metadata, so they contribute no auxiliary count loss (the model
is called without return_aux and compute_weighted_loss is handed aux_outputs
None); the synthetic half keeps the perception heads alive.

Usage:
    uv run python finetune_onpolicy.py --init_from run5/checkpoints/best.pth \
        --onpolicy_data onpolicy_data.pt --onpolicy_frac 0.5 --epochs 4 \
        --samples_per_epoch 50000 --batch_size 128 --lr 1e-4 \
        --seed 0 --log train_log_run6.jsonl
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import ToyVLM
from onpolicy import KEEP_KINDS
from questions import RationaleGenerator
from runtime import setup_runtime
from text import TextProcessor
from train_model import (ShapeDataset, build_param_groups, build_val_batches,
                         collate_fn, cosine_warmup_scheduler,
                         evaluate_teacher_forced, worker_init_fn)
from utils_loss import compute_weighted_loss

# The answer should be a copy from a correct rationale, so both spans keep
# weight 1.0; aux only steadies the (already trained) perception heads.
LOSS_WEIGHTS = {'rationale': 1.0, 'answer': 1.0, 'aux': 0.1}

# Fine-tuning sees the full question mixture from step one -- the curriculum
# already happened in the run that produced --init_from.
FINETUNE_MIXTURE = {'easy': 0.3, 'medium': 0.4, 'hard': 0.3}


class OnPolicyBatches:
    """Cycling batch sampler over the tensors saved by onpolicy.collect.

    Draws a fresh random permutation each pass; the tail shorter than
    batch_size is dropped and picked up by the next permutation, so every epoch
    of the underlying data is seen in a different order without ever emitting a
    ragged batch.
    """

    def __init__(self, path: str, batch_size: int, rng: random.Random):
        data = torch.load(path, map_location='cpu', weights_only=False)
        self.images = data['images']                     # (N,3,64,64) uint8
        self.input_tokens = data['input_tokens']
        self.target_tokens = data['target_tokens']
        self.rat_mask = data['rat_mask']
        self.ans_mask = data['ans_mask']
        self.disposition = data['disposition']            # (N,) int8
        self.stats = data.get('stats', {})
        self.n = self.input_tokens.shape[0]

        # The file carries its own int8 -> name legend, so the breakdown is read
        # off the data rather than assuming this process's KINDS ordering -- a
        # file written by a different vintage of onpolicy.py decodes correctly
        # and then fails the contract check below instead of silently training
        # on the wrong dispositions.
        kind_names = data['kind_names']
        counts = torch.bincount(self.disposition.to(torch.long),
                                minlength=len(kind_names))
        assert counts.numel() == len(kind_names), (
            f"[finetune] {path} holds disposition ids past the end of its own "
            f"{len(kind_names)}-name legend {kind_names}")
        breakdown = {name: int(counts[i]) for i, name in enumerate(kind_names)
                     if counts[i]}
        print(f"On-policy disposition breakdown ({self.n} samples): {breakdown}")

        # Every trace this trainer touches is supervised end to end: the loss
        # below weights the whole rationale span, so a partially supervised
        # trace would be trained as if its unsupervised part were gold.
        unexpected = sorted(set(breakdown) - set(KEEP_KINDS))
        assert not unexpected, (
            f"[finetune] {path} contains {unexpected} samples; this trainer "
            f"only accepts fully gold-supervised dispositions {list(KEEP_KINDS)}. "
            f"Re-collect with the current onpolicy.py.")

        assert self.n >= batch_size, (
            f"[finetune] {self.n} on-policy samples cannot fill a batch of {batch_size}")
        self.batch_size = batch_size
        self.rng = rng
        self.order: list = []

    def next_batch(self) -> dict:
        if len(self.order) < self.batch_size:
            self.order = list(range(self.n))
            self.rng.shuffle(self.order)
        idx = torch.tensor(self.order[:self.batch_size], dtype=torch.long)
        self.order = self.order[self.batch_size:]
        return {
            # uint8 -> float in [0,1], the exact conversion ShapeDataset applies
            'image': self.images[idx].to(torch.float32) / 255.0,
            'input_tokens': self.input_tokens[idx],
            'target_tokens': self.target_tokens[idx],
            'rat_mask': self.rat_mask[idx],
            'ans_mask': self.ans_mask[idx],
        }


def finetune(model, text_processor, runtime, onpolicy, args):
    device = runtime["device"]
    model = runtime["to_device"](model)
    model = runtime["maybe_compile"](model)

    param_groups = build_param_groups(model, wd=0.01)
    optimizer = runtime["make_optimizer_from_groups"](param_groups, lr=args.lr)

    steps_per_epoch = math.ceil(args.samples_per_epoch / args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    scheduler = cosine_warmup_scheduler(optimizer, int(0.03 * total_steps), total_steps)

    print(f"Fine-tuning on {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Total steps: {total_steps} ({args.epochs} epochs x {steps_per_epoch})")
    print(f"On-policy pool: {onpolicy.n} samples, drawn for {args.onpolicy_frac:.0%} of batches")

    print("Building fixed validation sets...")
    val_batches = build_val_batches(text_processor, use_cot=True,
                                    samples_per_difficulty=args.val_samples,
                                    batch_size=args.batch_size, seed=args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Batch-source coin flips live on their own stream, so the mixture is
    # reproducible regardless of how much randomness the data pipeline burns.
    mix_rng = random.Random(args.seed + 1)

    os.makedirs('checkpoints', exist_ok=True)
    best_val = -1.0

    def save_model(path):
        raw = getattr(model, '_orig_mod', model)  # unwrap torch.compile
        torch.save(raw.state_dict(), path)

    for epoch in range(args.epochs):
        dataset = ShapeDataset(args.samples_per_epoch, text_processor,
                               mixture=FINETUNE_MIXTURE, use_cot=True)
        dl_kwargs = {k: v for k, v in runtime["dataloader_kwargs"].items() if v is not None}
        if args.num_workers is not None:
            dl_kwargs['num_workers'] = args.num_workers
            dl_kwargs['persistent_workers'] = args.num_workers > 0
            if args.num_workers == 0:
                dl_kwargs.pop('prefetch_factor', None)
        synthetic_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
            collate_fn=collate_fn, worker_init_fn=worker_init_fn, **dl_kwargs)
        synthetic_iter = iter(synthetic_loader)

        model.train()
        totals = {'loss': 0.0, 'rat': 0.0, 'ans': 0.0, 'aux': 0.0}
        onp_loss, onp_steps = 0.0, 0
        syn_loss, syn_steps = 0.0, 0

        label = f"Epoch {epoch + 1}/{args.epochs}"
        print(f"\n{label} - mixture: {FINETUNE_MIXTURE} - loss weights: {LOSS_WEIGHTS}")
        progress_bar = tqdm(range(steps_per_epoch), desc=label)

        t0 = time.time()
        for _ in progress_bar:
            use_onpolicy = mix_rng.random() < args.onpolicy_frac
            if use_onpolicy:
                batch = onpolicy.next_batch()
                aux_labels = None
            else:
                try:
                    batch = next(synthetic_iter)
                except StopIteration:
                    synthetic_iter = iter(synthetic_loader)
                    batch = next(synthetic_iter)
                aux_labels = batch['aux_labels']

            images = batch['image'].to(device, non_blocking=True)
            inp = batch['input_tokens'].to(device, non_blocking=True)
            tgt = batch['target_tokens'].to(device, non_blocking=True)
            rat_mask = batch['rat_mask'].to(device, non_blocking=True)
            ans_mask = batch['ans_mask'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with runtime["autocast"]:
                if aux_labels is None:
                    # No scene metadata travels with an on-policy trace, so the
                    # count heads simply sit this step out.
                    logits = model(images, inp)
                    aux_outputs = None
                else:
                    logits, aux_outputs = model(images, inp, return_aux=True)
                loss, rat_loss, ans_loss, aux_loss = compute_weighted_loss(
                    logits, tgt, rat_mask, ans_mask,
                    aux_outputs, aux_labels, text_processor.tokenizer, LOSS_WEIGHTS
                )

            if runtime["scaler"] is not None:
                runtime["scaler"].scale(loss).backward()
                runtime["scaler"].unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                runtime["scaler"].step(optimizer)
                runtime["scaler"].update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            step_loss = loss.item()
            totals['loss'] += step_loss
            totals['rat'] += rat_loss.item()
            totals['ans'] += ans_loss.item()
            totals['aux'] += aux_loss.item()
            if use_onpolicy:
                onp_loss += step_loss
                onp_steps += 1
            else:
                syn_loss += step_loss
                syn_steps += 1

            progress_bar.set_postfix({
                'src': 'onp' if use_onpolicy else 'syn',
                'loss': f'{step_loss:.4f}',
                'rat': f'{rat_loss.item():.3f}',
                'ans': f'{ans_loss.item():.3f}',
                'aux': f'{aux_loss.item():.3f}'
            })

        del synthetic_iter, synthetic_loader, dataset

        val = evaluate_teacher_forced(model, val_batches, device, runtime["autocast"])
        overall_em = (sum(v['answer_em'] * v['n'] for v in val.values())
                      / max(1, sum(v['n'] for v in val.values())))

        record = {
            'epoch': epoch,
            'phase': 'finetune_onpolicy',
            'mixture': FINETUNE_MIXTURE,
            'loss_weights': LOSS_WEIGHTS,
            'onpolicy_frac': args.onpolicy_frac,
            'onpolicy_steps': onp_steps,
            'synthetic_steps': syn_steps,
            'train_loss': totals['loss'] / steps_per_epoch,
            'rationale_loss': totals['rat'] / steps_per_epoch,
            'answer_loss': totals['ans'] / steps_per_epoch,
            'aux_loss': totals['aux'] / steps_per_epoch,
            'onpolicy_loss': onp_loss / max(1, onp_steps),
            'synthetic_loss': syn_loss / max(1, syn_steps),
            'lr': optimizer.param_groups[0]['lr'],
            'val': val,
            'val_answer_em_overall': overall_em,
            'epoch_seconds': time.time() - t0,
        }
        with open(args.log, 'a') as f:
            f.write(json.dumps(record) + '\n')

        print(f"{label} summary: loss {record['train_loss']:.4f} "
              f"(rat {record['rationale_loss']:.3f} ans {record['answer_loss']:.3f} "
              f"aux {record['aux_loss']:.3f}) "
              f"[on-policy {record['onpolicy_loss']:.4f} x{onp_steps} | "
              f"synthetic {record['synthetic_loss']:.4f} x{syn_steps}] "
              f"lr {record['lr']:.6f}")
        for diff, v in val.items():
            print(f"  val[{diff}]: answer EM {v['answer_em']:.3f}  "
                  f"rationale token acc {v['rationale_token_acc']:.3f}")

        save_model(f'checkpoints/epoch_{epoch + 1}.pth')
        if overall_em >= best_val:  # >= so ties advance to the later checkpoint
            best_val = overall_em
            save_model('checkpoints/best.pth')
            print(f"  new best val answer EM: {best_val:.3f}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_from", type=str, required=True,
                        help="full checkpoint to fine-tune from")
    parser.add_argument("--onpolicy_data", type=str, required=True,
                        help=".pt file written by onpolicy.py")
    parser.add_argument("--onpolicy_frac", type=float, default=0.5,
                        help="probability a training batch comes from the on-policy pool")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--samples_per_epoch", type=int, default=50_000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_samples", type=int, default=256,
                        help="validation samples per difficulty")
    parser.add_argument("--log", type=str, default="train_log_run6.jsonl")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="override the runtime's dataloader worker count")
    parser.add_argument("--no_compile", action="store_true")
    args = parser.parse_args()

    assert 0.0 <= args.onpolicy_frac <= 1.0, "--onpolicy_frac must be a probability"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Same deterministic vocabulary the checkpoint was trained with; a mismatch
    # would silently permute token ids under the loaded embedding table.
    rationale_gen = RationaleGenerator()
    text_processor = TextProcessor()
    text_processor.tokenizer.build_vocab_from_rationales(rationale_gen)
    vocab_size = text_processor.tokenizer.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")

    state = torch.load(args.init_from, map_location='cpu')
    ckpt_vocab = state['token_embedding.weight'].shape[0]
    assert ckpt_vocab == vocab_size, (
        f"[finetune] vocabulary has {vocab_size} tokens but {args.init_from} was "
        f"trained with {ckpt_vocab}; the embedding rows would not line up")

    model = ToyVLM(text_processor)
    model.load_state_dict(state)
    print(f"Initialized from {args.init_from}")

    onpolicy = OnPolicyBatches(args.onpolicy_data, args.batch_size,
                               random.Random(args.seed + 2))
    if onpolicy.stats:
        print(f"On-policy data: {onpolicy.n} samples, counts {onpolicy.stats.get('counts')}")

    runtime = setup_runtime(prefer_compile=not args.no_compile)
    model = finetune(model, text_processor, runtime, onpolicy, args)

    torch.save(getattr(model, '_orig_mod', model).state_dict(), 'toy_vlm_cot.pth')
    print("\nFine-tuning complete. Model saved as 'toy_vlm_cot.pth' "
          "(per-epoch + best in checkpoints/)")


if __name__ == "__main__":
    main()
