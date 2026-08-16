"""
Toy Vision Language Model (VLM) in PyTorch
A simple VLM that answers questions about multi-shape scenes with
chain-of-thought reasoning.

Training regime:
  - Mixed-difficulty sampling every epoch (curriculum = mixture weights only,
    never exclusive phases).
  - Constant rationale/answer loss weights (the answer should be a copy from a
    correct rationale); aux count-head weight decays over training.
  - Optional 1-epoch perception warm-up (aux loss only) before full training.
  - Per-epoch teacher-forced validation on fixed seeded val sets, JSONL logging,
    and per-epoch checkpointing with best-by-val tracking.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Dict, List, Any
import argparse
import json
import os
import time
from shapes import ShapeGenerator, ObjType, ObjSize, COLORS, MIN_OBJECTS, MAX_OBJECTS
from questions import RationaleGenerator
from text import TextProcessor, MAX_SEQ_LEN
from model import ToyVLM
from runtime import setup_runtime
from utils_loss import compute_weighted_loss
import random
import math

# Training constants (defaults tuned for a single cloud GPU; override via CLI)
BATCH_SIZE = 128
NUM_EPOCHS = 20
LEARNING_RATE = 5e-4
SAMPLES_PER_EPOCH = 100_000

DIFFICULTIES = ['easy', 'medium', 'hard']


class ShapeDataset(Dataset):
    """On-the-fly dataset: multi-shape RGB scenes with QA pairs and CoT rationales.

    Difficulty is drawn per sample from `mixture` (dict difficulty -> weight),
    so every epoch sees all question families.
    """

    def __init__(self, num_samples: int, text_processor: TextProcessor,
                 mixture: Dict[str, float], use_cot: bool = True):
        self.num_samples = num_samples
        self.shape_generator = ShapeGenerator()
        self.rationale_generator = RationaleGenerator()
        self.text_processor = text_processor
        self.difficulties = list(mixture.keys())
        self.weights = [mixture[d] for d in self.difficulties]
        self.use_cot = use_cot

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        difficulty = random.choices(self.difficulties, weights=self.weights, k=1)[0]

        num_shapes = random.randint(MIN_OBJECTS, MAX_OBJECTS)
        image, metadata_list = self.shape_generator.generate_multi_shape_image(num_shapes, True)

        question, answer, rationale = self.rationale_generator.generate_qa_with_rationale(
            metadata_list, difficulty=difficulty
        )
        if not self.use_cot:
            rationale = ""  # ablation: empty <THINK> span, no rationale supervision

        inp_ids, tgt_ids, rat_mask, ans_mask = self.text_processor.prepare_input_sequence(
            question, answer, rationale
        )

        aux_labels = self._generate_aux_labels(metadata_list)

        # ---- pad everything (prepare_input_sequence asserts len <= MAX_SEQ_LEN) ----
        inp_ids  = self.text_processor.pad_sequence(inp_ids)
        tgt_ids  = self.text_processor.pad_sequence(tgt_ids)
        rat_mask = self.text_processor.pad_sequence(rat_mask)
        ans_mask = self.text_processor.pad_sequence(ans_mask)

        # ---- image: (H,W,3) uint8 -> (3,H,W) float in [0,1] ----
        img = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return {
            'image': img,
            'input_tokens': torch.tensor(inp_ids, dtype=torch.long),
            'target_tokens': torch.tensor(tgt_ids, dtype=torch.long),
            'rat_mask': torch.tensor(rat_mask, dtype=torch.float32),
            'ans_mask': torch.tensor(ans_mask, dtype=torch.float32),
            'question': question, 'answer': answer, 'rationale': rationale,
            'aux_labels': aux_labels, 'metadata': metadata_list
        }

    def _generate_aux_labels(self, metadata_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        """Ground-truth count labels for the auxiliary heads (0..MAX_OBJECTS each)."""
        shape_counts = {obj_type.value: 0 for obj_type in ObjType}
        size_counts = {size.value: 0 for size in ObjSize}
        color_counts = {color: 0 for color in sorted(COLORS.keys())}

        for m in metadata_list:
            shape_counts[m['shape']] += 1
            size_counts[m['size_category']] += 1
            color_counts[m['color']] += 1

        return {
            'counts': shape_counts,
            'size_counts': size_counts,
            'color_counts': color_counts,
        }


def get_difficulty_mixture(epoch: int) -> Dict[str, float]:
    """Curriculum as mixture weights only - all difficulties present every epoch."""
    if epoch < 2:
        return {'easy': 0.6, 'medium': 0.3, 'hard': 0.1}
    elif epoch < 4:
        return {'easy': 0.4, 'medium': 0.4, 'hard': 0.2}
    return {'easy': 0.3, 'medium': 0.4, 'hard': 0.3}


def get_loss_weights(epoch: int, num_epochs: int) -> Dict[str, float]:
    """Constant rationale/answer weights; aux decays 0.3 -> 0.1 over training."""
    frac = epoch / max(1, num_epochs - 1)
    return {'rationale': 1.0, 'answer': 1.0, 'aux': 0.3 + frac * (0.1 - 0.3)}


PERCEPTION_WARMUP_WEIGHTS = {'rationale': 0.0, 'answer': 0.0, 'aux': 1.0}


def collate_fn(batch):
    return {
        'image':         torch.stack([b['image'] for b in batch]),
        'input_tokens':  torch.stack([b['input_tokens'] for b in batch]),
        'target_tokens': torch.stack([b['target_tokens'] for b in batch]),
        'rat_mask':      torch.stack([b['rat_mask'] for b in batch]),
        'ans_mask':      torch.stack([b['ans_mask'] for b in batch]),
        'aux_labels':    [b['aux_labels'] for b in batch],
        'metadata':      [b['metadata'] for b in batch],
        'question':      [b['question'] for b in batch],
        'answer':        [b['answer'] for b in batch],
        'rationale':     [b['rationale'] for b in batch],
    }


def worker_init_fn(worker_id):
    # On fork-start workers, python/numpy RNG state is copied identically into
    # every worker; since __getitem__ generates data from global RNGs (not idx),
    # unseeded workers would produce duplicate sample streams.
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def build_param_groups(model, wd=0.01):
    no_wd = {
        id(model.token_embedding.weight),
        id(model.position_embedding.weight),
        id(model.text_type), id(model.vision_type),
        id(model.vision_token_encoder.pos_embed_2d),
        id(model.vision_token_encoder.pos_scale),
    }
    decay, nodecay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (nodecay if id(p) in no_wd or p.ndim == 1 else decay).append(p)
    return [
        {'params': decay,   'weight_decay': wd},
        {'params': nodecay, 'weight_decay': 0.0},
    ]


def cosine_warmup_scheduler(optimizer, warmup_steps, total_steps):
    def fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def build_val_batches(text_processor, use_cot: bool, samples_per_difficulty: int,
                      batch_size: int, seed: int) -> Dict[str, List[dict]]:
    """Fixed, seeded validation batches per difficulty (CPU tensors)."""
    val = {}
    for i, diff in enumerate(DIFFICULTIES):
        random.seed(seed * 1000 + i)
        np.random.seed(seed * 1000 + i)
        ds = ShapeDataset(samples_per_difficulty, text_processor,
                          mixture={diff: 1.0}, use_cot=use_cot)
        items = [ds[j] for j in range(samples_per_difficulty)]
        val[diff] = [collate_fn(items[k:k + batch_size])
                     for k in range(0, len(items), batch_size)]
    return val


@torch.no_grad()
def evaluate_teacher_forced(model, val_batches, device, autocast_ctx):
    """Teacher-forced argmax metrics per difficulty.

    answer_em: fraction of samples whose ENTIRE answer span is argmax-correct.
    rationale_token_acc: fraction of supervised rationale tokens argmax-correct.
    """
    model.eval()
    results = {}
    for diff, batches in val_batches.items():
        n = ans_correct = 0
        rat_correct = rat_total = 0
        for batch in batches:
            images = batch['image'].to(device)
            inp = batch['input_tokens'].to(device)
            tgt = batch['target_tokens'].to(device)
            rat_m = batch['rat_mask'].to(device).bool()
            ans_m = batch['ans_mask'].to(device).bool()

            with autocast_ctx:
                logits = model(images, inp)
            match = logits.argmax(dim=-1) == tgt

            has_ans = ans_m.any(dim=1)
            span_ok = (match | ~ans_m).all(dim=1) & has_ans
            ans_correct += int(span_ok.sum())
            n += int(has_ans.sum())

            rat_correct += int((match & rat_m).sum())
            rat_total += int(rat_m.sum())

        results[diff] = {
            'answer_em': ans_correct / max(1, n),
            'rationale_token_acc': rat_correct / max(1, rat_total),
            'n': n,
        }
    model.train()
    return results


def train(model, text_processor, runtime, args):
    device = runtime["device"]
    model = runtime["to_device"](model)
    model = runtime["maybe_compile"](model)

    param_groups = build_param_groups(model, wd=0.01)
    optimizer = runtime["make_optimizer_from_groups"](param_groups, lr=args.lr)

    warmup_epochs = 0 if args.no_perception_warmup else 1
    steps_per_epoch = math.ceil(args.samples_per_epoch / args.batch_size)
    total_steps = (args.epochs + warmup_epochs) * steps_per_epoch
    scheduler = cosine_warmup_scheduler(optimizer, int(0.03 * total_steps), total_steps)

    print(f"Training on {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Total steps: {total_steps} ({args.epochs} epochs + {warmup_epochs} perception warm-up)")

    print("Building fixed validation sets...")
    val_batches = build_val_batches(text_processor, use_cot=not args.no_cot,
                                    samples_per_difficulty=args.val_samples,
                                    batch_size=args.batch_size, seed=args.seed)
    # Restore the training seed after seeded val-set generation
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs('checkpoints', exist_ok=True)
    log_path = args.log
    best_val = -1.0

    def save_model(path):
        raw = getattr(model, '_orig_mod', model)  # unwrap torch.compile
        torch.save(raw.state_dict(), path)

    # epoch -1 = perception warm-up (aux loss only), then epochs 0..N-1
    for epoch in range(-warmup_epochs, args.epochs):
        is_warmup = epoch < 0
        if is_warmup:
            mixture = {'easy': 1/3, 'medium': 1/3, 'hard': 1/3}
            loss_weights = PERCEPTION_WARMUP_WEIGHTS
            label = "perception warm-up"
        else:
            mixture = get_difficulty_mixture(epoch)
            loss_weights = get_loss_weights(epoch, args.epochs)
            label = f"Epoch {epoch + 1}/{args.epochs}"

        dataset = ShapeDataset(args.samples_per_epoch, text_processor,
                               mixture=mixture, use_cot=not args.no_cot)
        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
            **{k: v for k, v in runtime["dataloader_kwargs"].items() if v is not None}
        )

        model.train()
        total_loss = total_rationale_loss = total_answer_loss = total_aux_loss = 0.0

        print(f"\n{label} - mixture: {mixture} - loss weights: {loss_weights}")
        progress_bar = tqdm(train_loader, desc=label)

        t0 = time.time()
        for batch in progress_bar:
            images = batch['image'].to(device, non_blocking=True)
            inp = batch['input_tokens'].to(device, non_blocking=True)
            tgt = batch['target_tokens'].to(device, non_blocking=True)
            rat_mask = batch['rat_mask'].to(device, non_blocking=True)
            ans_mask = batch['ans_mask'].to(device, non_blocking=True)
            aux_labels = batch['aux_labels']

            optimizer.zero_grad(set_to_none=True)
            with runtime["autocast"]:
                logits, aux_outputs = model(images, inp, return_aux=True)
                loss, rat_loss, ans_loss, aux_loss = compute_weighted_loss(
                    logits, tgt, rat_mask, ans_mask,
                    aux_outputs, aux_labels, text_processor.tokenizer, loss_weights
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

            scheduler.step()  # step per batch

            total_loss += loss.item()
            total_rationale_loss += rat_loss.item()
            total_answer_loss += ans_loss.item()
            total_aux_loss += aux_loss.item()

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'rat': f'{rat_loss.item():.3f}',
                'ans': f'{ans_loss.item():.3f}',
                'aux': f'{aux_loss.item():.3f}'
            })

        nb = len(train_loader)
        val = evaluate_teacher_forced(model, val_batches, device, runtime["autocast"])
        overall_em = sum(v['answer_em'] * v['n'] for v in val.values()) / max(1, sum(v['n'] for v in val.values()))

        record = {
            'epoch': epoch,
            'phase': 'warmup' if is_warmup else 'train',
            'mixture': mixture,
            'loss_weights': loss_weights,
            'train_loss': total_loss / nb,
            'rationale_loss': total_rationale_loss / nb,
            'answer_loss': total_answer_loss / nb,
            'aux_loss': total_aux_loss / nb,
            'lr': optimizer.param_groups[0]['lr'],
            'val': val,
            'val_answer_em_overall': overall_em,
            'epoch_seconds': time.time() - t0,
        }
        with open(log_path, 'a') as f:
            f.write(json.dumps(record) + '\n')

        print(f"{label} summary: loss {record['train_loss']:.4f} "
              f"(rat {record['rationale_loss']:.3f} ans {record['answer_loss']:.3f} "
              f"aux {record['aux_loss']:.3f}) lr {record['lr']:.6f}")
        for diff, v in val.items():
            print(f"  val[{diff}]: answer EM {v['answer_em']:.3f}  "
                  f"rationale token acc {v['rationale_token_acc']:.3f}")

        if not is_warmup:
            save_model(f'checkpoints/epoch_{epoch + 1}.pth')
            if overall_em > best_val:
                best_val = overall_em
                save_model('checkpoints/best.pth')
                print(f"  new best val answer EM: {best_val:.3f}")

    return model


def main():
    """Main training function with chain-of-thought reasoning."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--samples_per_epoch", type=int, default=SAMPLES_PER_EPOCH)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_samples", type=int, default=256,
                        help="validation samples per difficulty")
    parser.add_argument("--log", type=str, default="train_log.jsonl")
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--try_fp8", action="store_true")
    parser.add_argument("--no_cot", action="store_true",
                        help="ablation: train with empty <THINK> spans")
    parser.add_argument("--no_perception_warmup", action="store_true",
                        help="skip the 1-epoch aux-only perception warm-up")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Initializing Toy VLM with Chain-of-Thought Reasoning...")

    # Deterministic tokenizer vocabulary enumerated from the question generator
    rationale_gen = RationaleGenerator()
    text_processor = TextProcessor()
    text_processor.tokenizer.build_vocab_from_rationales(rationale_gen)
    text_processor.tokenizer.save_vocab('tokenizer_vocab.json')
    print(f"Vocabulary size: {text_processor.tokenizer.get_vocab_size()}")

    model = ToyVLM(text_processor)

    runtime = setup_runtime(prefer_compile=not args.no_compile, try_fp8=args.try_fp8)

    model = train(model, text_processor, runtime, args)

    torch.save(getattr(model, '_orig_mod', model).state_dict(), 'toy_vlm_cot.pth')
    print("\nTraining complete. Model saved as 'toy_vlm_cot.pth' (per-epoch + best in checkpoints/)")


if __name__ == "__main__":
    main()
