#!/usr/bin/env python3
"""Standalone validator for the toy-cot data layer (no torch required).

For every question type it draws fresh scenes, generates the question / answer /
rationale, and then *independently* re-derives the ground truth from the
quantized grid coordinates. Every fact stated in a rationale is re-parsed and
checked:

* each enumerated ``{color} {shape} at row r col c`` names a real object that
  really sits in that quantized cell, and the enumeration is complete and in
  raster order;
* every stated count equals the true count of the filtered set it claims to
  describe (including per-side counts, which is where the old generator lied);
* every cited witness really satisfies the relation it is cited for;
* the final answer equals the independently recomputed ground truth and is
  consistent with the trace's found / none found step.

It also reports the yes/no answer balance per type and the maximum tokenized
sequence length, tokenizing with allow_unk=False so any out-of-vocabulary word
is a hard failure.

Usage:
    python validate_traces.py [--samples 1000] [--seed 0]
"""

import argparse
import random
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from shapes import (
    COLORS,
    MAX_OBJECTS,
    MIN_OBJECTS,
    ObjSize,
    ObjType,
    ShapeGenerator,
    grid_col,
    grid_row,
)
from questions import DIFFICULTY_MAP, RationaleGenerator
from text import MAX_SEQ_LEN, PREFIX_LEN, SimpleTokenizer, TextProcessor

# ---------------------------------------------------------------------------
# Independent re-implementation of the ground-truth semantics
# ---------------------------------------------------------------------------

SHAPES = [t.value for t in ObjType]
COLOR_NAMES = list(COLORS.keys())
SIZES = [s.value for s in ObjSize]
SIDES = ['left', 'right', 'top', 'bottom']
OPPOSITE = {'left': 'right', 'right': 'left', 'top': 'bottom', 'bottom': 'top'}
RELATIONS = ['above', 'below', 'left of', 'right of']

Descriptor = Tuple[Optional[str], Optional[str]]  # (color, shape)


def plural_of(word: str) -> str:
    return 'crosses' if word == 'cross' else word + 's'


SINGULAR = {}
for _w in SHAPES + ['shape']:
    SINGULAR[_w] = _w
    SINGULAR[plural_of(_w)] = _w


def cell(m: Dict[str, Any]) -> Tuple[int, int]:
    return grid_row(m['cy']), grid_col(m['cx'])


def matches(m: Dict[str, Any], desc: Descriptor) -> bool:
    color, shape = desc
    if color is not None and m['color'] != color:
        return False
    if shape is not None and m['shape'] != shape:
        return False
    return True


def select(meta: Sequence[Dict[str, Any]], desc: Descriptor) -> List[Dict[str, Any]]:
    return sorted([m for m in meta if matches(m, desc)], key=cell)


def relation(a: Dict[str, Any], b: Dict[str, Any], rel: str) -> bool:
    ra, ca = cell(a)
    rb, cb = cell(b)
    if rel == 'above':
        return ra < rb
    if rel == 'below':
        return ra > rb
    if rel == 'left of':
        return ca < cb
    if rel == 'right of':
        return ca > cb
    raise ValueError(rel)


def side_holds(m: Dict[str, Any], side: str) -> bool:
    row, col = cell(m)
    if side == 'left':
        return col <= 3
    if side == 'right':
        return col >= 4
    if side == 'top':
        return row <= 3
    if side == 'bottom':
        return row >= 4
    raise ValueError(side)


def phrase_for(a: int, b: int) -> str:
    if a > b:
        return 'greater than'
    if a < b:
        return 'less than'
    return 'equal to'


def desc_text(desc: Descriptor, plural: bool = False) -> str:
    color, shape = desc
    head = shape if shape is not None else 'shape'
    if plural:
        head = plural_of(head)
    return f"{color} {head}" if color is not None else head


def parse_desc(text: str) -> Descriptor:
    words = text.split()
    color = None
    if words[0] in COLOR_NAMES:
        color = words[0]
        words = words[1:]
    head = SINGULAR[words[0]]
    shape = None if head == 'shape' else head
    return (color, shape)


# ---------------------------------------------------------------------------
# Question grammars
# ---------------------------------------------------------------------------

_COLOR_ALT = '|'.join(COLOR_NAMES)
_NOUNS = sorted({w for s in SHAPES for w in (s, plural_of(s))} | {'shape', 'shapes'},
                key=len, reverse=True)
_NOUN_ALT = '|'.join(_NOUNS)
DESC = rf"(?:(?:{_COLOR_ALT}) )?(?:{_NOUN_ALT})"
REL = '|'.join(RELATIONS)
SIDE = '|'.join(SIDES)
SIZE = '|'.join(SIZES)

Q_PATTERNS = {
    'existence': re.compile(rf"is there a ({DESC})$"),
    'positional_existence': re.compile(
        rf"(?:is there a|are there any) ({DESC}) on the ({SIDE})$"),
    'counting': re.compile(rf"how many ({DESC}) are there$"),
    'size': re.compile(rf"are there any ({SIZE}) shapes$"),
    'relative_position': re.compile(rf"is a ({DESC}) ({REL}) a ({DESC})$"),
    'side_count_comparison': re.compile(rf"are there more ({DESC}) on the ({SIDE})$"),
    'comparison': re.compile(rf"are there more ({DESC}) than ({DESC})$"),
    'compositional': re.compile(
        rf"is a ({DESC}) ({REL}) the ({DESC}) that is ({REL}) the ({DESC})$"),
}

ENUM_STEP = re.compile(
    rf"(?:({SIZE}) )?({_COLOR_ALT}) ({'|'.join(SHAPES)}) at row (\d) col (\d)$")
QUALIFY_STEP = re.compile(
    rf"qualifying (?:({_COLOR_ALT}) )?({_NOUN_ALT}) at row (\d) col (\d)$")
WITNESS_STEP = re.compile(rf"(row|col) (\d) is ({REL}) (row|col) (\d)$")
COUNT_STEP = re.compile(r"count is (\d+)$")
COUNT_OF_STEP = re.compile(rf"count of ({DESC}) is (\d+)$")
COUNT_SIDE_STEP = re.compile(rf"count on ({SIDE}) is (\d+)$")
EMPTY_STEP = re.compile(rf"no (?:({SIZE}) )?(?:({_COLOR_ALT}) )?({_NOUN_ALT}) found$")
CMP_STEP = re.compile(r"(\d+) (greater than|less than|equal to) (\d+)$")
PLAIN_STEPS = {'found', 'none found'}


# ---------------------------------------------------------------------------
# Sample checking
# ---------------------------------------------------------------------------

class SampleError(Exception):
    """A fact stated by a rationale did not survive verification."""


class Cursor:
    """Walks the steps of a rationale, checking each one against the truth."""

    def __init__(self, rationale: str):
        self.steps = rationale.split(' . ')
        self.i = 0

    def peek(self) -> str:
        if self.i >= len(self.steps):
            raise SampleError(f"trace ended early after {self.i} steps")
        return self.steps[self.i]

    def take(self) -> str:
        step = self.peek()
        self.i += 1
        return step

    def expect(self, want: str):
        got = self.take()
        if got != want:
            raise SampleError(f"step {self.i - 1}: expected {want!r}, got {got!r}")

    def expect_end(self):
        if self.i != len(self.steps):
            raise SampleError(
                f"trailing steps after {self.i}: {self.steps[self.i:]!r}")

    def enumerate_set(self, objs: Sequence[Dict[str, Any]], desc_or_empty: str,
                      with_size: bool = False):
        """Consume the enumeration of `objs` (raster order) or the empty step."""
        if not objs:
            self.expect(f"no {desc_or_empty} found")
            return
        for m in objs:
            row, col = cell(m)
            prefix = f"{m['size_category']} " if with_size else ''
            self.expect(f"{prefix}{m['color']} {m['shape']} at row {row} col {col}")

    def take_count(self, expected: int):
        step = self.take()
        m = COUNT_STEP.fullmatch(step)
        if not m:
            raise SampleError(f"expected 'count is N', got {step!r}")
        if int(m.group(1)) != expected:
            raise SampleError(f"stated count {m.group(1)} != true count {expected} "
                              f"in step {step!r}")

    def take_count_of(self, desc: Descriptor, expected: int):
        step = self.take()
        m = COUNT_OF_STEP.fullmatch(step)
        if not m:
            raise SampleError(f"expected 'count of X is N', got {step!r}")
        if parse_desc(m.group(1)) != desc:
            raise SampleError(f"count scoped to {m.group(1)!r}, expected "
                              f"{desc_text(desc, True)!r} in step {step!r}")
        if int(m.group(2)) != expected:
            raise SampleError(f"stated count {m.group(2)} != true count {expected} "
                              f"in step {step!r}")

    def take_count_on(self, side: str, expected: int):
        step = self.take()
        m = COUNT_SIDE_STEP.fullmatch(step)
        if not m:
            raise SampleError(f"expected 'count on SIDE is N', got {step!r}")
        if m.group(1) != side:
            raise SampleError(f"count attributed to side {m.group(1)!r}, "
                              f"expected {side!r} in step {step!r}")
        if int(m.group(2)) != expected:
            raise SampleError(f"stated count {m.group(2)} for side {side} != true "
                              f"count {expected} in step {step!r}")

    def take_comparison(self, a: int, b: int):
        step = self.take()
        m = CMP_STEP.fullmatch(step)
        if not m:
            raise SampleError(f"expected comparison step, got {step!r}")
        if int(m.group(1)) != a or int(m.group(3)) != b:
            raise SampleError(f"comparison operands {m.group(1)},{m.group(3)} != "
                              f"{a},{b} in step {step!r}")
        if m.group(2) != phrase_for(a, b):
            raise SampleError(f"comparison phrase {m.group(2)!r} wrong for {a} vs {b}")

    def take_verdict(self, truth: bool, answer: str, set_a: Sequence[Dict[str, Any]],
                     set_b: Sequence[Dict[str, Any]], rel: str):
        """Consume either 'none found' or a witness step followed by 'found'."""
        step = self.take()
        if step == 'none found':
            if truth:
                raise SampleError(f"trace says 'none found' but a {rel} pair exists")
            if answer != 'no':
                raise SampleError(f"answer {answer!r} contradicts 'none found'")
            return
        m = WITNESS_STEP.fullmatch(step)
        if not m:
            raise SampleError(f"expected witness or 'none found', got {step!r}")
        axis_a, val_a, cited_rel, axis_b, val_b = (
            m.group(1), int(m.group(2)), m.group(3), m.group(4), int(m.group(5)))
        if cited_rel != rel:
            raise SampleError(f"witness cites relation {cited_rel!r}, question asks {rel!r}")
        want_axis = 'row' if rel in ('above', 'below') else 'col'
        if axis_a != want_axis or axis_b != want_axis:
            raise SampleError(f"witness cites axis {axis_a}/{axis_b}, expected {want_axis} "
                              f"for relation {rel!r}")
        idx = 0 if want_axis == 'row' else 1
        if not any(cell(m_)[idx] == val_a for m_ in set_a):
            raise SampleError(f"witness {step!r} cites a first object at {want_axis} "
                              f"{val_a} that is not in the first set")
        if not any(cell(m_)[idx] == val_b for m_ in set_b):
            raise SampleError(f"witness {step!r} cites a second object at {want_axis} "
                              f"{val_b} that is not in the second set")
        holds = val_a < val_b if rel in ('above', 'left of') else val_a > val_b
        if not holds:
            raise SampleError(f"witness {step!r} does not satisfy relation {rel!r}")
        if not truth:
            raise SampleError(f"trace cites a witness but no {rel} pair exists")
        self.expect('found')
        if answer != 'yes':
            raise SampleError(f"answer {answer!r} contradicts a found witness")


def check_step_facts(rationale: str, meta: Sequence[Dict[str, Any]]):
    """Every step must be well-formed; object steps must name a real object."""
    for step in rationale.split(' . '):
        if step in PLAIN_STEPS:
            continue
        m = ENUM_STEP.fullmatch(step)
        if m:
            size, color, shape, row, col = m.groups()
            hit = [o for o in meta
                   if o['color'] == color and o['shape'] == shape
                   and cell(o) == (int(row), int(col))
                   and (size is None or o['size_category'] == size)]
            if not hit:
                raise SampleError(f"step {step!r} describes no real object")
            continue
        m = QUALIFY_STEP.fullmatch(step)
        if m:
            color, noun, row, col = m.groups()
            shape = SINGULAR[noun]
            hit = [o for o in meta
                   if (color is None or o['color'] == color)
                   and (shape == 'shape' or o['shape'] == shape)
                   and cell(o) == (int(row), int(col))]
            if not hit:
                raise SampleError(f"step {step!r} describes no real object")
            continue
        if any(p.fullmatch(step) for p in (WITNESS_STEP, COUNT_STEP, COUNT_OF_STEP,
                                           COUNT_SIDE_STEP, EMPTY_STEP, CMP_STEP)):
            continue
        raise SampleError(f"unrecognized rationale step {step!r}")


def parse_question(qtype: str, question: str):
    m = Q_PATTERNS[qtype].fullmatch(question)
    if not m:
        raise SampleError(f"question {question!r} does not match the {qtype} template")
    return m


def check_existence(q, a, r, meta):
    desc = parse_desc(parse_question('existence', q).group(1))
    objs = select(meta, desc)
    cur = Cursor(r)
    cur.enumerate_set(objs, desc_text(desc, True))
    cur.take_count(len(objs))
    cur.expect_end()
    truth = 'yes' if objs else 'no'
    if a != truth:
        raise SampleError(f"answer {a!r} != ground truth {truth!r}")


def check_positional_existence(q, a, r, meta):
    m = parse_question('positional_existence', q)
    desc, side = parse_desc(m.group(1)), m.group(2)
    objs = select(meta, desc)
    on_side = [o for o in objs if side_holds(o, side)]
    cur = Cursor(r)
    cur.enumerate_set(objs, desc_text(desc, True))
    cur.take_count_on(side, len(on_side))
    cur.expect_end()
    truth = 'yes' if on_side else 'no'
    if a != truth:
        raise SampleError(f"answer {a!r} != ground truth {truth!r}")


def check_counting(q, a, r, meta):
    desc = parse_desc(parse_question('counting', q).group(1))
    objs = select(meta, desc)
    cur = Cursor(r)
    cur.enumerate_set(objs, desc_text(desc, True))
    cur.take_count(len(objs))
    cur.expect_end()
    if a != str(len(objs)):
        raise SampleError(f"answer {a!r} != ground truth {len(objs)}")


def check_size(q, a, r, meta):
    size = parse_question('size', q).group(1)
    objs = sorted([o for o in meta if o['size_category'] == size], key=cell)
    cur = Cursor(r)
    cur.enumerate_set(objs, f"{size} shapes", with_size=True)
    cur.take_count(len(objs))
    cur.expect_end()
    truth = 'yes' if objs else 'no'
    if a != truth:
        raise SampleError(f"answer {a!r} != ground truth {truth!r}")


def check_relative_position(q, a, r, meta):
    m = parse_question('relative_position', q)
    desc_a, rel, desc_b = parse_desc(m.group(1)), m.group(2), parse_desc(m.group(3))
    objs_a, objs_b = select(meta, desc_a), select(meta, desc_b)
    truth = any(relation(x, y, rel) for x in objs_a for y in objs_b)

    cur = Cursor(r)
    cur.enumerate_set(objs_a, desc_text(desc_a, True))
    cur.enumerate_set(objs_b, desc_text(desc_b, True))
    cur.take_verdict(truth, a, objs_a, objs_b, rel)
    cur.expect_end()
    want = 'yes' if truth else 'no'
    if a != want:
        raise SampleError(f"answer {a!r} != ground truth {want!r}")


def check_side_count_comparison(q, a, r, meta):
    m = parse_question('side_count_comparison', q)
    desc, side = parse_desc(m.group(1)), m.group(2)
    other = OPPOSITE[side]
    objs = select(meta, desc)
    here = sum(1 for o in objs if side_holds(o, side))
    there = sum(1 for o in objs if side_holds(o, other))

    cur = Cursor(r)
    cur.enumerate_set(objs, desc_text(desc, True))
    cur.take_count_on(side, here)
    cur.take_count_on(other, there)
    cur.take_comparison(here, there)
    cur.expect_end()
    want = 'yes' if here > there else 'no'
    if a != want:
        raise SampleError(f"answer {a!r} != ground truth {want!r} ({side}={here}, "
                          f"{other}={there})")


def check_comparison(q, a, r, meta):
    m = parse_question('comparison', q)
    desc_a, desc_b = parse_desc(m.group(1)), parse_desc(m.group(2))
    objs_a, objs_b = select(meta, desc_a), select(meta, desc_b)

    cur = Cursor(r)
    cur.enumerate_set(objs_a, desc_text(desc_a, True))
    cur.take_count_of(desc_a, len(objs_a))
    cur.enumerate_set(objs_b, desc_text(desc_b, True))
    cur.take_count_of(desc_b, len(objs_b))
    cur.take_comparison(len(objs_a), len(objs_b))
    cur.expect_end()
    want = 'yes' if len(objs_a) > len(objs_b) else 'no'
    if a != want:
        raise SampleError(f"answer {a!r} != ground truth {want!r}")


def check_compositional(q, a, r, meta):
    m = parse_question('compositional', q)
    desc_a = parse_desc(m.group(1))
    rel1 = m.group(2)
    desc_b = parse_desc(m.group(3))
    rel2 = m.group(4)
    desc_c = parse_desc(m.group(5))

    objs_b, objs_c = select(meta, desc_b), select(meta, desc_c)
    qualifying = [b for b in objs_b if any(relation(b, c, rel2) for c in objs_c)]

    cur = Cursor(r)
    cur.enumerate_set(objs_b, desc_text(desc_b, True))
    cur.enumerate_set(objs_c, desc_text(desc_c, True))

    if not qualifying:
        cur.expect('none found')
        cur.expect_end()
        if a != 'no':
            raise SampleError(f"answer {a!r} but no {desc_text(desc_b)} qualifies")
        return

    for b in qualifying:
        row, col = cell(b)
        cur.expect(f"qualifying {desc_text(desc_b)} at row {row} col {col}")

    objs_a = select(meta, desc_a)
    cur.enumerate_set(objs_a, desc_text(desc_a, True))
    truth = any(relation(x, b, rel1) for x in objs_a for b in qualifying)
    cur.take_verdict(truth, a, objs_a, qualifying, rel1)
    cur.expect_end()
    want = 'yes' if truth else 'no'
    if a != want:
        raise SampleError(f"answer {a!r} != ground truth {want!r}")


CHECKERS = {
    'existence': check_existence,
    'positional_existence': check_positional_existence,
    'counting': check_counting,
    'size': check_size,
    'relative_position': check_relative_position,
    'side_count_comparison': check_side_count_comparison,
    'comparison': check_comparison,
    'compositional': check_compositional,
}

YES_NO_TYPES = {name for name in CHECKERS if name != 'counting'}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def difficulty_of(qtype: str) -> str:
    for difficulty, names in DIFFICULTY_MAP.items():
        if qtype in names:
            return difficulty
    return '?'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=1000,
                        help='fresh scenes per question type (default: 1000)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--max-failures', type=int, default=5,
                        help='failures to print per type before summarizing')
    args = parser.parse_args()

    random.seed(args.seed)

    shape_gen = ShapeGenerator()
    rg = RationaleGenerator()

    vocabulary = rg.vocabulary()
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(vocabulary)
    processor = TextProcessor()
    processor.tokenizer = tokenizer

    missing_types = sorted(set(CHECKERS) ^ set(rg.generators))
    if missing_types:
        print(f"FAIL: generator/checker mismatch: {missing_types}")
        return 1
    mapped = {name for names in DIFFICULTY_MAP.values() for name in names}
    if mapped != set(rg.generators):
        print(f"FAIL: DIFFICULTY_MAP {sorted(mapped)} != generators "
              f"{sorted(rg.generators)}")
        return 1

    stats: Dict[str, Dict[str, Any]] = {}
    failures: List[str] = []
    used_words: set = set()
    rows_seen: Counter = Counter()
    cols_seen: Counter = Counter()
    global_max_len = 0
    longest_sample = None

    for qtype, generator in rg.generators.items():
        answers: Counter = Counter()
        max_len = 0
        skipped = 0
        checked = 0
        type_failures = 0

        for _ in range(args.samples):
            num_shapes = random.randint(MIN_OBJECTS, MAX_OBJECTS)
            _img, meta = shape_gen.generate_multi_shape_image(num_shapes, False)
            for obj in meta:
                row, col = cell(obj)
                rows_seen[row] += 1
                cols_seen[col] += 1
            question, answer, rationale = generator(meta)
            if question is None:
                skipped += 1
                continue

            # 1. facts stated by the trace
            try:
                check_step_facts(rationale, meta)
                CHECKERS[qtype](question, answer, rationale, meta)
                checked += 1
            except SampleError as exc:
                type_failures += 1
                if type_failures <= args.max_failures:
                    failures.append(
                        f"[{qtype}] {exc}\n    scene={[(o['size_category'], o['color'], o['shape'], cell(o)) for o in meta]}"
                        f"\n    q={question!r}\n    a={answer!r}\n    r={rationale!r}")
                continue

            answers[answer] += 1
            used_words.update(rationale.split())
            used_words.update(question.split())
            used_words.update(answer.split())

            # 2. tokenization: length budget and zero OOV (allow_unk=False)
            try:
                input_ids, target_ids, rat_mask, ans_mask = \
                    processor.prepare_input_sequence(question, answer, rationale)
            except (AssertionError, ValueError) as exc:
                type_failures += 1
                if type_failures <= args.max_failures:
                    failures.append(f"[{qtype}] tokenization failed: {exc}\n"
                                    f"    q={question!r}\n    a={answer!r}\n    r={rationale!r}")
                continue

            if not (len(target_ids) == len(rat_mask) == len(ans_mask) == len(input_ids) - 1):
                failures.append(f"[{qtype}] mask/target length mismatch")
                type_failures += 1
                continue
            if sum(ans_mask) < 1 or sum(rat_mask) < 1:
                failures.append(f"[{qtype}] empty supervision mask")
                type_failures += 1
                continue

            if len(input_ids) > max_len:
                max_len = len(input_ids)
            if len(input_ids) > global_max_len:
                global_max_len = len(input_ids)
                longest_sample = (qtype, question, answer, rationale, len(input_ids))

        stats[qtype] = {
            'checked': checked,
            'skipped': skipped,
            'failures': type_failures,
            'answers': answers,
            'max_len': max_len,
        }

    # ---------------------------------------------------------------- report
    print("=" * 78)
    print(f"validate_traces: {args.samples} scenes per type, seed={args.seed}, "
          f"objects {MIN_OBJECTS}..{MAX_OBJECTS}")
    print("=" * 78)
    header = (f"{'question type':<24}{'diff':<8}{'checked':>8}{'skip':>6}{'fail':>6}"
              f"{'yes%':>8}{'no%':>8}{'maxlen':>8}")
    print(header)
    print("-" * len(header))

    for qtype, s in stats.items():
        answers = s['answers']
        total = sum(answers.values())
        if qtype in YES_NO_TYPES and total:
            yes_pct = 100.0 * answers.get('yes', 0) / total
            no_pct = 100.0 * answers.get('no', 0) / total
            balance = f"{yes_pct:7.1f}%{no_pct:7.1f}%"
        else:
            balance = f"{'n/a':>8}{'n/a':>8}"
        print(f"{qtype:<24}{difficulty_of(qtype):<8}{s['checked']:>8}{s['skipped']:>6}"
              f"{s['failures']:>6}{balance}{s['max_len']:>8}")

    print("-" * len(header))
    print(f"vocabulary size (incl. 13 special tokens): {tokenizer.get_vocab_size()} "
          f"({len(vocabulary)} words)")
    print(f"max sequence length observed: {global_max_len} / MAX_SEQ_LEN={MAX_SEQ_LEN} "
          f"(prefix={PREFIX_LEN})")
    if longest_sample:
        qtype, q, a, r, n = longest_sample
        print(f"  longest ({n} tokens, {qtype}): q={q!r}")

    counting_answers = stats['counting']['answers']
    print(f"counting answer distribution: "
          f"{sorted(counting_answers.items(), key=lambda kv: int(kv[0]))}")

    print(f"grid rows occupied: {sorted(rows_seen)}   cols occupied: {sorted(cols_seen)}"
          f"   (shape margins keep objects off the border cells)")

    unused = sorted(vocabulary - used_words)
    if unused:
        print(f"vocabulary words never observed ({len(unused)}): {unused}")

    problems = []
    if failures:
        problems.append(f"{sum(s['failures'] for s in stats.values())} sample failures")
    for qtype, s in stats.items():
        if s['checked'] == 0:
            problems.append(f"{qtype}: no samples were generated")
            continue
        if qtype in YES_NO_TYPES:
            unexpected = set(s['answers']) - {'yes', 'no'}
            if unexpected:
                problems.append(f"{qtype}: non yes/no answers {sorted(unexpected)}")
            yes_share = s['answers'].get('yes', 0) / max(1, sum(s['answers'].values()))
            if not 0.30 <= yes_share <= 0.70:
                problems.append(f"{qtype}: answer prior is skewed "
                                f"(yes={100 * yes_share:.1f}%)")
    if counting_answers.get('0', 0) == 0:
        problems.append("counting: a count of 0 never occurred")
    if global_max_len > MAX_SEQ_LEN:
        problems.append(f"max sequence length {global_max_len} exceeds {MAX_SEQ_LEN}")

    if failures:
        print("\n" + "=" * 78)
        print("FAILURES (truncated)")
        print("=" * 78)
        for line in failures[:40]:
            print(line)

    if problems:
        print("\nFAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("\nOK: every rationale fact verified, answers match recomputed ground "
          "truth, zero OOV tokens, all sequences fit the context.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
