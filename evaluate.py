"""
Evaluation suite for the Toy VLM with Chain-of-Thought reasoning.
Tests the model on various question types and measures exact match accuracy,
aggregated by difficulty, with majority-class baselines for context.
"""

import argparse
import json
import os
import random
import sys
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from shapes import ShapeGenerator, MIN_OBJECTS, MAX_OBJECTS
from questions import RationaleGenerator, DIFFICULTY_MAP
from text import TextProcessor, SimpleTokenizer
from model import ToyVLM, generate_response, generate_response_self_consistent


def best_device() -> torch.device:
    """Pick the best available device: cuda -> mps -> cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class VLMEvaluator:
    """Evaluates VLM on generated test sets, per question type and difficulty."""

    def __init__(self, model: ToyVLM, text_processor: TextProcessor,
                 self_consistency_k: int = 1, temperature: float = 0.0,
                 max_gen_len: int = 160):
        self.model = model
        self.text_processor = text_processor
        self.shape_gen = ShapeGenerator()
        self.rationale_gen = RationaleGenerator()
        self.self_consistency_k = self_consistency_k
        self.temperature = temperature
        self.max_gen_len = max_gen_len
        self.device = next(model.parameters()).device

    @staticmethod
    def normalize_answer(answer: str) -> str:
        """Normalize answer for comparison."""
        if answer is None:
            return ""
        return answer.lower().strip().replace('.', '').replace(',', '')

    def exact_match(self, predicted: str, ground_truth: str) -> bool:
        return self.normalize_answer(predicted) == self.normalize_answer(ground_truth)

    def _make_scene(self, target_n: int = None) -> Tuple[Any, List[Dict[str, Any]]]:
        """Generate a scene; with target_n, retry toward that ACTUAL object count.

        Rejection sampling under-delivers on dense scenes, so unstratified draws
        would leave high-N accuracy buckets nearly empty. Best effort: return the
        closest-achieved scene after a bounded number of tries.
        """
        if target_n is None:
            num_shapes = random.randint(MIN_OBJECTS, MAX_OBJECTS)
            return self.shape_gen.generate_multi_shape_image(num_shapes, add_noise=False)
        best = None
        for _ in range(30):
            image, metadata = self.shape_gen.generate_multi_shape_image(target_n, add_noise=False)
            if len(metadata) == target_n:
                return image, metadata
            if best is None or abs(len(metadata) - target_n) < abs(len(best[1]) - target_n):
                best = (image, metadata)
        return best

    def generate_test_set(self, qtype: str, generator_fn, num_samples: int) -> List[Dict[str, Any]]:
        """Generate a test set of `num_samples` for a single question type,
        resampling scenes whenever the generator declines (None, None, None)."""
        test_samples = []
        # Guard against pathological generators that can never produce a question
        # for the shapes on offer; bail out after a generous number of attempts.
        max_attempts = num_samples * 50 + 100
        attempts = 0

        while len(test_samples) < num_samples and attempts < max_attempts:
            attempts += 1
            # Stratify scene density: cycle target N over 1..MAX_OBJECTS so
            # accuracy-vs-N buckets all get samples (per-N accuracy is
            # conditional on N, so the changed marginal doesn't bias it).
            target_n = MIN_OBJECTS + (len(test_samples) % (MAX_OBJECTS - MIN_OBJECTS + 1))
            image, metadata = self._make_scene(target_n)
            question, answer, rationale = generator_fn(metadata)
            if question is None or answer is None or rationale is None:
                continue
            test_samples.append({
                'image': image,
                'metadata': metadata,
                'question': question,
                'ground_truth_answer': answer,
                'ground_truth_rationale': rationale,
            })

        if len(test_samples) < num_samples:
            print(
                f"  Warning: only generated {len(test_samples)}/{num_samples} samples "
                f"for question type '{qtype}' after {attempts} attempts"
            )

        return test_samples

    def _to_model_image(self, img) -> torch.Tensor:
        """Convert an (H, W, 3) uint8 RGB image into the (3, 64, 64) float32 [0,1]
        tensor expected by generate_response."""
        return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

    def evaluate_test_set(
        self, qtype: str, test_samples: List[Dict[str, Any]], show_examples: int
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Evaluate model on a single question type's test set and return metrics."""
        self.model.eval()

        correct = 0
        rationale_correct = 0
        total = len(test_samples)
        empty_predictions = 0
        generation_errors = 0

        results = []
        gt_answers = []

        for i, sample in enumerate(tqdm(test_samples, desc=qtype, leave=False)):
            question = sample['question']
            gt_answer = sample['ground_truth_answer']
            gt_rationale = sample['ground_truth_rationale']
            gt_answers.append(gt_answer)

            img = self._to_model_image(sample['image'])

            try:
                if self.self_consistency_k > 1:
                    pred_rationale, pred_answer, _ = generate_response_self_consistent(
                        self.model, img, question, k=self.self_consistency_k,
                        temperature=self.temperature, max_length=self.max_gen_len
                    )
                else:
                    pred_rationale, pred_answer = generate_response(
                        self.model, img, question, max_length=self.max_gen_len,
                        return_rationale=True, temperature=self.temperature
                    )
                if not pred_answer.strip():
                    empty_predictions += 1
            except Exception as e:
                print(f"\nError generating response for question: {question}")
                print(f"Error: {e}")
                pred_rationale, pred_answer = "", ""
                generation_errors += 1

            is_correct = self.exact_match(pred_answer, gt_answer)
            if is_correct:
                correct += 1

            is_rationale_correct = self.exact_match(pred_rationale, gt_rationale)
            if is_rationale_correct:
                rationale_correct += 1

            results.append({
                'question': question,
                'ground_truth_answer': gt_answer,
                'predicted_answer': pred_answer,
                'ground_truth_rationale': gt_rationale,
                'predicted_rationale': pred_rationale,
                'correct': is_correct,
                'rationale_correct': is_rationale_correct,
                'n_objects': len(sample['metadata']),
            })

            if i < show_examples:
                print(f"\n--- {qtype} example {i + 1} ---")
                print(f"Question: {question}")
                print(f"GT Answer: {gt_answer}")
                print(f"Pred Answer: {pred_answer}")
                print(f"GT Rationale: {gt_rationale}")
                print(f"Pred Rationale: {pred_rationale}")
                print(f"Correct: {is_correct}")

        accuracy = correct / total if total > 0 else 0.0
        rationale_exact = rationale_correct / total if total > 0 else 0.0

        majority_count = Counter(self.normalize_answer(a) for a in gt_answers).most_common(1)
        majority_baseline = (majority_count[0][1] / total) if total > 0 and majority_count else 0.0

        # Self-correction telemetry: how often does the model emit 'wait'?
        wait_rate = sum(1 for r in results
                        if ' wait ' in f" {r['predicted_rationale']} ") / total if total else 0.0

        # Accuracy vs scene object count (the crossover-curve data): 2-object bins
        by_n: Dict[str, Dict[str, Any]] = {}
        for r in results:
            lo = ((r['n_objects'] - 1) // 2) * 2 + 1
            key = f"{lo}-{lo + 1}"
            b = by_n.setdefault(key, {'n': 0, 'correct': 0})
            b['n'] += 1
            b['correct'] += int(r['correct'])
        for b in by_n.values():
            b['accuracy'] = b['correct'] / b['n']

        metrics = {
            'type': qtype,
            'n': total,
            'accuracy': accuracy,
            'majority_baseline': majority_baseline,
            'rationale_exact': rationale_exact,
            'empty': empty_predictions,
            'errors': generation_errors,
            'wait_rate': wait_rate,
            'by_n': {k: {'n': v['n'], 'accuracy': v['accuracy']}
                     for k, v in sorted(by_n.items(), key=lambda kv: int(kv[0].split('-')[0]))},
        }

        return metrics, results

    def evaluate_all_types(
        self, num_samples_per_type: int, show_examples: int
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate on every question type registered on the rationale generator."""
        per_type_metrics: Dict[str, Dict[str, Any]] = {}

        for qtype, generator_fn in self.rationale_gen.generators.items():
            print(f"\n{'=' * 60}")
            print(f"Evaluating question type: {qtype}")
            print('=' * 60)

            test_set = self.generate_test_set(qtype, generator_fn, num_samples_per_type)
            if not test_set:
                print(f"  Skipping '{qtype}': no valid samples could be generated.")
                continue

            metrics, _ = self.evaluate_test_set(qtype, test_set, show_examples)
            per_type_metrics[qtype] = metrics

            print(
                f"\n{qtype}: acc {metrics['accuracy']:.1%}  "
                f"majority {metrics['majority_baseline']:.1%}  "
                f"rationale_exact {metrics['rationale_exact']:.1%}  "
                f"(n={metrics['n']}, empty={metrics['empty']}, errors={metrics['errors']})"
            )

        return per_type_metrics

    @staticmethod
    def aggregate_by_difficulty(
        per_type_metrics: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Roll per-type metrics up into per-difficulty metrics using DIFFICULTY_MAP."""
        by_difficulty: Dict[str, Dict[str, Any]] = {}

        for difficulty, qtypes in DIFFICULTY_MAP.items():
            n = 0
            correct = 0
            rationale_correct = 0
            majority_correct = 0
            empty = 0
            errors = 0

            for qtype in qtypes:
                m = per_type_metrics.get(qtype)
                if m is None:
                    continue
                n += m['n']
                correct += m['accuracy'] * m['n']
                rationale_correct += m['rationale_exact'] * m['n']
                majority_correct += m['majority_baseline'] * m['n']
                empty += m['empty']
                errors += m['errors']

            if n == 0:
                continue

            by_difficulty[difficulty] = {
                'type': difficulty,
                'n': n,
                'accuracy': correct / n,
                'majority_baseline': majority_correct / n,
                'rationale_exact': rationale_correct / n,
                'empty': empty,
                'errors': errors,
            }

        return by_difficulty

    @staticmethod
    def aggregate_overall(per_type_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        n = sum(m['n'] for m in per_type_metrics.values())
        if n == 0:
            return {
                'type': 'overall', 'n': 0, 'accuracy': 0.0,
                'majority_baseline': 0.0, 'rationale_exact': 0.0,
                'empty': 0, 'errors': 0,
            }

        correct = sum(m['accuracy'] * m['n'] for m in per_type_metrics.values())
        rationale_correct = sum(m['rationale_exact'] * m['n'] for m in per_type_metrics.values())
        majority_correct = sum(m['majority_baseline'] * m['n'] for m in per_type_metrics.values())
        empty = sum(m['empty'] for m in per_type_metrics.values())
        errors = sum(m['errors'] for m in per_type_metrics.values())

        return {
            'type': 'overall',
            'n': n,
            'accuracy': correct / n,
            'majority_baseline': majority_correct / n,
            'rationale_exact': rationale_correct / n,
            'empty': empty,
            'errors': errors,
        }


def print_summary(
    per_type_metrics: Dict[str, Dict[str, Any]],
    per_difficulty_metrics: Dict[str, Dict[str, Any]],
    overall_metrics: Dict[str, Any],
) -> None:
    print(f"\n{'=' * 72}")
    print("SUMMARY BY QUESTION TYPE")
    print('=' * 72)
    print(f"{'type':28s} {'n':>5s} {'acc':>8s} {'majority':>10s} {'rat.exact':>10s} {'empty':>7s} {'errors':>7s}")
    for qtype, m in per_type_metrics.items():
        print(
            f"{qtype:28s} {m['n']:5d} {m['accuracy']:7.1%} {m['majority_baseline']:9.1%} "
            f"{m['rationale_exact']:9.1%} {m['empty']:7d} {m['errors']:7d}"
        )

    print(f"\n{'=' * 72}")
    print("ACCURACY BY SCENE OBJECT COUNT (crossover-curve data)")
    print('=' * 72)
    bins = sorted({b for m in per_type_metrics.values() for b in m.get('by_n', {})},
                  key=lambda k: int(k.split('-')[0]))
    print(f"{'type':28s} " + " ".join(f"{b:>8s}" for b in bins))
    for qtype, m in per_type_metrics.items():
        cells = []
        for b in bins:
            v = m.get('by_n', {}).get(b)
            cells.append(f"{v['accuracy']:7.1%} " if v else f"{'-':>8s}")
        print(f"{qtype:28s} " + " ".join(c.strip().rjust(8) for c in cells))

    print(f"\n{'=' * 72}")
    print("SUMMARY BY DIFFICULTY")
    print('=' * 72)
    print(f"{'difficulty':28s} {'n':>5s} {'acc':>8s} {'majority':>10s} {'rat.exact':>10s}")
    for difficulty, m in per_difficulty_metrics.items():
        print(
            f"{difficulty:28s} {m['n']:5d} {m['accuracy']:7.1%} {m['majority_baseline']:9.1%} "
            f"{m['rationale_exact']:9.1%}"
        )

    print(f"\n{'=' * 72}")
    print("OVERALL")
    print('=' * 72)
    print(
        f"n={overall_metrics['n']}  accuracy={overall_metrics['accuracy']:.1%}  "
        f"majority_baseline={overall_metrics['majority_baseline']:.1%}  "
        f"rationale_exact={overall_metrics['rationale_exact']:.1%}  "
        f"empty={overall_metrics['empty']}  errors={overall_metrics['errors']}"
    )


def write_results(
    out_path: str,
    per_type_metrics: Dict[str, Dict[str, Any]],
    overall_metrics: Dict[str, Any],
) -> None:
    fields = ['type', 'n', 'accuracy', 'majority_baseline', 'rationale_exact', 'empty', 'errors']
    with open(out_path, 'w') as f:
        for m in per_type_metrics.values():
            row = {k: m[k] for k in fields}
            row['by_n'] = m.get('by_n', {})
            row['wait_rate'] = m.get('wait_rate', 0.0)
            f.write(json.dumps(row) + '\n')
        f.write(json.dumps({k: overall_metrics[k] for k in fields}) + '\n')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Toy VLM with chain-of-thought reasoning.")
    parser.add_argument('--checkpoint', type=str, default='toy_vlm_cot.pth')
    parser.add_argument('--vocab', type=str, default='tokenizer_vocab.json')
    parser.add_argument('--samples', type=int, default=500, help='Samples per question type')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default='eval_results.jsonl')
    parser.add_argument('--show-examples', type=int, default=3)
    parser.add_argument('--self_consistency', type=int, default=1,
                        help='k chains with majority vote (1 = single chain)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='rationale sampling temperature; only used when k > 1')
    parser.add_argument('--max_gen_len', type=int, default=160,
                        help='max tokens per decode stage (old default 35 truncated long chains)')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not os.path.isfile(args.checkpoint):
        sys.exit(
            f"Checkpoint not found: '{args.checkpoint}'. "
            f"Train a model first (artifacts are gitignored)."
        )
    if not os.path.isfile(args.vocab):
        sys.exit(
            f"Vocab file not found: '{args.vocab}'. "
            f"Train a model first (artifacts are gitignored)."
        )

    print("Loading model for evaluation...")

    tokenizer = SimpleTokenizer()
    tokenizer.load_vocab(args.vocab)
    text_processor = TextProcessor()
    text_processor.tokenizer = tokenizer

    model = ToyVLM(text_processor)
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(state_dict)

    device = best_device()
    model = model.to(device)
    model.eval()
    print(f"Loaded '{args.checkpoint}' onto device: {device}")

    k = args.self_consistency
    evaluator = VLMEvaluator(
        model, text_processor,
        self_consistency_k=k,
        temperature=args.temperature if k > 1 else 0.0,
        max_gen_len=args.max_gen_len,
    )
    if k > 1:
        print(f"Self-consistency: {k} chains at temperature {args.temperature}, majority vote")

    per_type_metrics = evaluator.evaluate_all_types(
        num_samples_per_type=args.samples, show_examples=args.show_examples
    )
    per_difficulty_metrics = evaluator.aggregate_by_difficulty(per_type_metrics)
    overall_metrics = evaluator.aggregate_overall(per_type_metrics)

    print_summary(per_type_metrics, per_difficulty_metrics, overall_metrics)
    write_results(args.out, per_type_metrics, overall_metrics)
    print(f"\nWrote results to '{args.out}'")


if __name__ == '__main__':
    main()
