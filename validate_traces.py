#!/usr/bin/env python3
"""Standalone validator for the toy-cot data layer (no torch required).

For every question type it draws fresh scenes, generates the question / answer /
rationale, and then *independently* re-derives the ground truth from the
quantized grid coordinates. Every fact stated in a rationale is re-parsed and
checked:

* each enumerated ``{color} {shape} at row r col c`` names a real object that
  really sits in that quantized cell, and the enumeration is complete and in
  raster order -- for every type, ordinal included. Two objects may share a
  quantized cell, so an enumeration may legitimately repeat a line --
  multiplicity is checked per enumerated set;
* an ``ordinal`` trace's sorted walk is checked against an axis order recomputed
  here: rank by rank, the step must name the object this validator puts at that
  rank and cite that object's true coordinate on the axis the side is measured
  on. A walk that is internally coherent but sorted wrongly fails, however well
  it agrees with the answer it leads to;
* every stated count equals the true count of the filtered set it claims to
  describe (including per-side counts, which is where the old generator lied);
* every stated parity and count difference is recomputed from those counts;
* every cited witness really satisfies the relation it is cited for, and every
  ``qualifying ...`` line of a compositional chain names an object that really
  starts a valid chain from that level onwards (re-derived here by a top-down
  existential search, not by the generator's cascade);
* a ``relative_count`` referent is resolved from the scene here, not read off
  the trace: one match makes the question answerable and the trace must exclude
  the referent from its own candidates, two or more make it unanswerable as
  posed and the trace must stop at ``ambiguous`` with an answer that asks the
  question back ("which red triangle"). A trace that answers the other regime
  than the scene dictates is rejected;
* the final answer equals the independently recomputed ground truth and is
  consistent with the trace's found / none found step.

It also reports the yes/no answer balance per type (per chain depth as well),
the answer spread of the counting / count-difference / ordinal types, the
achieved scene-density distribution, and the maximum tokenized sequence length,
tokenizing with allow_unk=False so any out-of-vocabulary word is a hard failure.
A final stress pass hammers the generators at maximum scene density to prove no
draw can overflow the model's context.

With ``--correction_p P`` each verified sample is additionally run through
questions.inject_correction with probability P, and the resulting
error-injected self-correction trace is verified too: the injected fact must be
unambiguously false, the correction must restore the original (real) fact, the
answer must be untouched, and the segment masks must leave exactly the
corrupted tokens unsupervised.

Usage:
    python validate_traces.py [--samples 1000] [--seed 0] [--stress 65536]
                              [--correction_p 0.0] [--roundtrip 500]
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
from questions import (
    AMBIGUOUS_MARKER,
    AMBIGUOUS_TARGET_PROB,
    CLARIFY_PREFIX,
    CORRECTION_MARKER,
    CORRECTION_RESERVE,
    DIFFICULTY_MAP,
    ENUM_FACT,
    QUESTION_RESERVE,
    STEP_SEP,
    WORD_BUDGET,
    RationaleGenerator,
    inject_correction,
    split_steps,
)
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
ORDINAL_WORDS = ['first', 'second', 'third', 'fourth']
ORDINAL_RANK = {w: i + 1 for i, w in enumerate(ORDINAL_WORDS)}
CHAIN_DEPTHS = [2, 3, 4]

# Independent sort keys for "k-th from the {side}".
AXIS_KEY = {
    'left': lambda rc: (rc[1], rc[0]),
    'right': lambda rc: (-rc[1], rc[0]),
    'top': lambda rc: (rc[0], rc[1]),
    'bottom': lambda rc: (-rc[0], rc[1]),
}
# The coordinate an ordering "from the {side}" is read off, and therefore the
# one an ordinal walk step must cite.
ORDINAL_AXIS = {'left': 'col', 'right': 'col', 'top': 'row', 'bottom': 'row'}

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


def axis_order(meta: Sequence[Dict[str, Any]], side: str) -> List[Dict[str, Any]]:
    """Objects ordered by distance from `side` (total order iff cells distinct)."""
    key = AXIS_KEY[side]
    return sorted(meta, key=lambda m: key(cell(m)))


def chain_oracle(meta: Sequence[Dict[str, Any]], descs: Sequence[Descriptor],
                 rels: Sequence[str]):
    """Top-down existential search over the chain.

    Returns starts(level, obj): True iff `obj` matches descs[level] and objects
    x(level+1)..x(h+1) exist with rels[i](x_i, x_i+1) all the way to the end.
    Nothing here mirrors the generator's tail-first cascade: it is the plain
    reading of "is a D1 r1 the D2 that is r2 the D3 ...".
    """
    last = len(descs) - 1
    cache: Dict[Tuple[int, int], bool] = {}

    def starts(level: int, obj: Dict[str, Any]) -> bool:
        key = (level, id(obj))
        if key in cache:
            return cache[key]
        # `level` strictly increases with the recursion depth, so no cycles.
        if not matches(obj, descs[level]):
            result = False
        elif level == last:
            result = True
        else:
            result = any(relation(obj, nxt, rels[level]) and starts(level + 1, nxt)
                         for nxt in meta)
        cache[key] = result
        return result

    return starts


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
SHAPE_ALT = '|'.join(SHAPES)
ORD = '|'.join(ORDINAL_WORDS)


def chain_question_pattern(hops: int) -> 're.Pattern':
    """'is a D1 r1 the D2 that is r2 the D3 ...' with `hops` relations."""
    pattern = rf"is a ({DESC}) ({REL}) the ({DESC})"
    pattern += rf" that is ({REL}) the ({DESC})" * (hops - 1)
    return re.compile(pattern + "$")


Q_PATTERNS = {
    'existence': re.compile(rf"is there a ({DESC})$"),
    'positional_existence': re.compile(
        rf"(?:is there a|are there any) ({DESC}) on the ({SIDE})$"),
    'counting': re.compile(rf"how many ({DESC}) are there$"),
    'size': re.compile(rf"are there any ({SIZE}) shapes$"),
    'relative_position': re.compile(rf"is a ({DESC}) ({REL}) a ({DESC})$"),
    'side_count_comparison': re.compile(rf"are there more ({DESC}) on the ({SIDE})$"),
    'parity': re.compile(rf"are there an even number of ({DESC})$"),
    'comparison': re.compile(rf"are there more ({DESC}) than ({DESC})$"),
    'count_difference': re.compile(
        rf"what is the difference between the number of ({DESC}) and "
        rf"the number of ({DESC})$"),
    'ordinal': re.compile(rf"what shape is ({ORD}) from the ({SIDE})$"),
    'relative_count': re.compile(rf"how many ({DESC}) are ({REL}) the ({DESC})$"),
}
Q_PATTERNS.update({f'compositional_h{h}': chain_question_pattern(h)
                   for h in CHAIN_DEPTHS})

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
PARITY_STEP = re.compile(r"(\d+) is (even|odd)$")
DIFF_STEP = re.compile(r"difference is (\d+)$")
ORDINAL_STEP = re.compile(
    rf"({ORD}) from ({SIDE}) is ({SHAPE_ALT}) at (row|col) (\d)$")
PLAIN_STEPS = {'found', 'none found', AMBIGUOUS_MARKER}


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
        """Consume the enumeration of `objs` (in the given order) or the empty step.

        One step is consumed per object, so if two objects share a quantized
        cell the trace must repeat the line: multiplicity is enforced here,
        where the set boundaries are known (a whole-trace tally could not tell a
        repeated cell from an object legitimately enumerated in two sets).
        """
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

    def take_parity(self, n: int):
        step = self.take()
        m = PARITY_STEP.fullmatch(step)
        if not m:
            raise SampleError(f"expected 'N is even|odd', got {step!r}")
        if int(m.group(1)) != n:
            raise SampleError(f"parity stated for {m.group(1)}, true count is {n}")
        want = 'even' if n % 2 == 0 else 'odd'
        if m.group(2) != want:
            raise SampleError(f"{n} called {m.group(2)!r}, it is {want!r}")

    def take_difference(self, a: int, b: int):
        step = self.take()
        m = DIFF_STEP.fullmatch(step)
        if not m:
            raise SampleError(f"expected 'difference is N', got {step!r}")
        if int(m.group(1)) != abs(a - b):
            raise SampleError(f"stated difference {m.group(1)} != |{a} - {b}|")

    def take_ordinal_step(self, ordinal: str, side: str, obj: Dict[str, Any]):
        """Consume one step of the ordinal sorted walk; it must name `obj`.

        `obj` is whatever the independently recomputed axis order puts at this
        rank, so a walk whose steps agree with each other -- and with the answer
        they lead to -- still fails here if the order it walks is not the true
        one. The cited coordinate is checked too, against the axis the queried
        side is measured on and the object's own quantized cell.
        """
        step = self.take()
        m = ORDINAL_STEP.fullmatch(step)
        if not m:
            raise SampleError(f"expected '{{ordinal}} from {{side}} is {{shape}} at "
                              f"{{row|col}} {{n}}', got {step!r}")
        if m.group(1) != ordinal or m.group(2) != side:
            raise SampleError(f"step {step!r} answers a different question than "
                              f"{ordinal!r} from {side!r}")
        if m.group(3) != obj['shape']:
            raise SampleError(f"step {step!r} names {m.group(3)!r}, the {ordinal} "
                              f"from the {side} is a {obj['shape']} at {cell(obj)}")
        axis = ORDINAL_AXIS[side]
        if m.group(4) != axis:
            raise SampleError(f"step {step!r} cites a {m.group(4)}, an ordering from "
                              f"the {side} is read off the {axis}")
        want = cell(obj)[0 if axis == 'row' else 1]
        if int(m.group(5)) != want:
            raise SampleError(f"step {step!r} cites {axis} {m.group(5)}, but the "
                              f"object it names sits at {axis} {want}")

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
    """Every step must be well-formed; object steps must name a real object.

    This is the line-level pass: it only asks "does this line describe an object
    that exists?". How many times a line may appear is a property of the set
    being enumerated and is checked by Cursor.enumerate_set.
    """
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
                                           COUNT_SIDE_STEP, EMPTY_STEP, CMP_STEP,
                                           PARITY_STEP, DIFF_STEP, ORDINAL_STEP)):
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


def check_parity(q, a, r, meta):
    desc = parse_desc(parse_question('parity', q).group(1))
    objs = select(meta, desc)
    cur = Cursor(r)
    cur.enumerate_set(objs, desc_text(desc, True))
    cur.take_count(len(objs))
    cur.take_parity(len(objs))
    cur.expect_end()
    truth = 'yes' if len(objs) % 2 == 0 else 'no'
    if a != truth:
        raise SampleError(f"answer {a!r} != ground truth {truth!r} (count {len(objs)})")


def check_count_difference(q, a, r, meta):
    m = parse_question('count_difference', q)
    desc_a, desc_b = parse_desc(m.group(1)), parse_desc(m.group(2))
    if desc_a == desc_b:
        raise SampleError("count difference asked between a descriptor and itself")
    objs_a, objs_b = select(meta, desc_a), select(meta, desc_b)

    cur = Cursor(r)
    cur.enumerate_set(objs_a, desc_text(desc_a, True))
    cur.take_count_of(desc_a, len(objs_a))
    cur.enumerate_set(objs_b, desc_text(desc_b, True))
    cur.take_count_of(desc_b, len(objs_b))
    cur.take_difference(len(objs_a), len(objs_b))
    cur.expect_end()
    want = str(abs(len(objs_a) - len(objs_b)))
    if a != want:
        raise SampleError(f"answer {a!r} != ground truth {want!r} "
                          f"({len(objs_a)} vs {len(objs_b)})")


def check_ordinal(q, a, r, meta):
    """'what shape is third from the left'.

    Two independent passes over the trace. The enumeration must list the whole
    scene in raster order -- the same listing every other type emits, so nothing
    about this type asks the model to read an object list a second way. The
    sorted walk is then checked rank by rank against an axis order recomputed
    here from the quantized cells, not read off the enumeration: a walk that
    skips, reorders or invents a rank fails even when every one of its steps
    names a real object and the answer follows from the last of them.
    """
    m = parse_question('ordinal', q)
    ordinal, side = m.group(1), m.group(2)
    rank = ORDINAL_RANK[ordinal]

    cells = [cell(o) for o in meta]
    if len(set(cells)) != len(cells):
        raise SampleError("ordinal question asked of a scene where two objects "
                          "share a cell: the axis order is not total")
    if rank > len(meta):
        raise SampleError(f"asked for the {ordinal} object of a {len(meta)}-object scene")

    cur = Cursor(r)
    cur.enumerate_set(sorted(meta, key=cell), plural_of('shape'))
    ordered = axis_order(meta, side)
    for i in range(rank):
        cur.take_ordinal_step(ORDINAL_WORDS[i], side, ordered[i])
    cur.expect_end()
    truth = ordered[rank - 1]['shape']
    if a != truth:
        raise SampleError(f"answer {a!r} != ground truth {truth!r}")


def check_relative_count(q, a, r, meta):
    """'how many circles are below the red triangle'.

    Which of the two regimes the sample belongs to is decided here, from the
    scene: the referent descriptor is matched against the metadata and the count
    of matches -- not anything the trace says -- selects the structure the trace
    and the answer must have. A unique-referent trace carrying an ambiguous
    answer, or the reverse, is therefore rejected by construction.
    """
    m = parse_question('relative_count', q)
    desc_x, rel, desc_ref = parse_desc(m.group(1)), m.group(2), parse_desc(m.group(3))
    x_noun, ref_noun = m.group(1).split()[-1], m.group(3).split()[-1]
    if x_noun == SINGULAR[x_noun]:
        raise SampleError(f"the counted descriptor {m.group(1)!r} is not plural")
    if ref_noun != SINGULAR[ref_noun] or desc_ref[1] is None:
        raise SampleError(f"the referent {m.group(3)!r} must name one shape "
                          f"in the singular")
    if desc_x == desc_ref:
        raise SampleError(f"the counted set and the referent are the same "
                          f"descriptor {desc_text(desc_ref)!r}")

    referents = select(meta, desc_ref)
    if not referents:
        raise SampleError(f"referent {desc_text(desc_ref)!r} matches no object: "
                          f"the empty regime must never be drawn")

    cur = Cursor(r)
    cur.enumerate_set(referents, desc_text(desc_ref, True))

    # ---- ambiguous regime: the question has no answer as posed --------------
    if len(referents) > 1:
        cur.take_count_of(desc_ref, len(referents))
        cur.expect(AMBIGUOUS_MARKER)
        cur.expect_end()
        want = f"{CLARIFY_PREFIX} {desc_text(desc_ref)}"
        if a != want:
            raise SampleError(f"answer {a!r} != {want!r}: {len(referents)} objects "
                              f"match the referent, so the answer must ask back "
                              f"with the referent's own descriptor")
        return

    # ---- unique regime: count the candidates on the named side -------------
    # Nothing lies left of, right of, above or below itself, so the referent is
    # not a candidate even when it matches the counted descriptor. Excluding it
    # by identity is exact here: any other object with the referent's color and
    # shape would match the referent descriptor too, and would have put this
    # sample in the ambiguous regime.
    referent = referents[0]
    candidates = [o for o in select(meta, desc_x) if o is not referent]
    qualifying = [o for o in candidates if relation(o, referent, rel)]

    cur.enumerate_set(candidates, desc_text(desc_x, True))
    for o in qualifying:
        row, col = cell(o)
        cur.expect(f"qualifying {desc_text(desc_x)} at row {row} col {col}")
    cur.take_count(len(qualifying))
    cur.expect_end()
    if a != str(len(qualifying)):
        raise SampleError(f"answer {a!r} != ground truth {len(qualifying)} "
                          f"({len(candidates)} candidates, {rel} the referent at "
                          f"{cell(referent)})")


def check_chain(q, a, r, meta, hops: int):
    """Compositional chain of `hops` relation hops.

    Truth is re-derived by an existential search (chain_oracle); the trace's
    per-level 'qualifying' lines must name exactly the objects that really start
    a valid chain at that level, in raster order.
    """
    groups = parse_question(f'compositional_h{hops}', q).groups()
    descs = [parse_desc(groups[i]) for i in range(0, len(groups), 2)]
    rels = [groups[i] for i in range(1, len(groups), 2)]
    if len(descs) != hops + 1 or len(rels) != hops:
        raise SampleError(f"chain question parsed into {len(descs)} descriptors and "
                          f"{len(rels)} relations, expected {hops + 1}/{hops}")
    for j in range(1, hops):
        if descs[j] == descs[j + 1]:
            raise SampleError(f"chain clause {j} and {j + 1} name the same referent "
                              f"{desc_text(descs[j])!r}")

    starts = chain_oracle(meta, descs, rels)
    sets = [select(meta, d) for d in descs]

    cur = Cursor(r)
    for j in range(1, hops + 1):
        cur.enumerate_set(sets[j], desc_text(descs[j], True))

    qualifying: List[Dict[str, Any]] = []
    for j in range(hops - 1, 0, -1):
        qualifying = [o for o in sets[j] if starts(j, o)]
        if not qualifying:
            cur.expect('none found')
            cur.expect_end()
            if a != 'no':
                raise SampleError(f"answer {a!r} but nothing qualifies as the "
                                  f"{desc_text(descs[j])} of clause {j}")
            return
        for o in qualifying:
            row, col = cell(o)
            cur.expect(f"qualifying {desc_text(descs[j])} at row {row} col {col}")

    cur.enumerate_set(sets[0], desc_text(descs[0], True))
    truth = any(starts(0, o) for o in meta)
    cur.take_verdict(truth, a, sets[0], qualifying, rels[0])
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
    'parity': check_parity,
    'comparison': check_comparison,
    'count_difference': check_count_difference,
    'ordinal': check_ordinal,
    'relative_count': check_relative_count,
}
CHECKERS.update({
    f'compositional_h{h}': (lambda hops: lambda q, a, r, meta: check_chain(q, a, r, meta, hops))(h)
    for h in CHAIN_DEPTHS
})

# Types whose answer is not yes/no (their spread is reported instead).
OPEN_ANSWER_TYPES = {'counting', 'count_difference', 'ordinal', 'relative_count'}
YES_NO_TYPES = set(CHECKERS) - OPEN_ANSWER_TYPES


# ---------------------------------------------------------------------------
# Error-injected correction traces (--correction_p)
# ---------------------------------------------------------------------------

def enum_matches(step: str, meta: Sequence[Dict[str, Any]],
                 ignore_size: bool = False) -> List[Dict[str, Any]]:
    """Objects the enumeration `step` could be describing (may be empty)."""
    m = ENUM_STEP.fullmatch(step)
    if not m:
        raise SampleError(f"step {step!r} is not an enumeration fact")
    size, color, shape, row, col = m.groups()
    return [o for o in meta
            if o['color'] == color and o['shape'] == shape
            and cell(o) == (int(row), int(col))
            and (ignore_size or size is None or o['size_category'] == size)]


def fact_fields(step: str) -> Tuple[Optional[str], str, str, int, int]:
    m = ENUM_STEP.fullmatch(step)
    if not m:
        raise SampleError(f"step {step!r} is not an enumeration fact")
    size, color, shape, row, col = m.groups()
    return size, color, shape, int(row), int(col)


def segment_layout(segments: Sequence[Tuple[str, bool]]) -> Tuple[List[str], int, int]:
    """Return (steps of the corrected trace, step index and segment index of the
    corrupted fact).

    Enforces the injector's contract: exactly one unsupervised segment, holding
    exactly one whole step, and the concatenation of the segments splits cleanly
    on the ' . ' separator (no doubled or missing separators).
    """
    if not segments:
        raise SampleError("inject_correction returned an empty segment list")
    wrong = [j for j, (_text, supervised) in enumerate(segments) if not supervised]
    if len(wrong) != 1:
        raise SampleError(f"expected exactly one unsupervised segment, got {len(wrong)}")
    j = wrong[0]
    corrupted = segments[j][0]
    if STEP_SEP in corrupted or corrupted != corrupted.strip():
        raise SampleError(f"unsupervised segment {corrupted!r} is not a single bare step")

    full = ''.join(text for text, _sup in segments)
    if full != full.strip():
        raise SampleError(f"reassembled trace has a dangling separator: {full!r}")
    steps = full.split(STEP_SEP)
    if any(not step or step != step.strip() for step in steps):
        raise SampleError(f"reassembled trace has doubled separators: {full!r}")
    if STEP_SEP.join(steps) != full:
        raise SampleError(f"reassembled trace does not round-trip its steps: {full!r}")

    index = ''.join(text for text, _sup in segments[:j]).count(STEP_SEP)
    if steps[index] != corrupted:
        raise SampleError(f"unsupervised segment {corrupted!r} is not step {index} "
                          f"({steps[index]!r}) of the reassembled trace")
    return steps, index, j


def check_correction(qtype: str, question: str, answer: str, rationale: str,
                     segments: Sequence[Tuple[str, bool]],
                     meta: Sequence[Dict[str, Any]], processor: TextProcessor
                     ) -> Tuple[List[int], List[int], int]:
    """Verify one error-injected correction trace.

    Returns (target_ids, rat_mask, think_start) so the caller can print a spot
    example. Raises SampleError on any violation.
    """
    steps, index, seg_index = segment_layout(segments)

    # ---- structure: {corrupted} . wait . {original} ----------------------
    if index + 2 >= len(steps):
        raise SampleError("correction insertion runs past the end of the trace")
    if steps[index + 1] != CORRECTION_MARKER:
        raise SampleError(f"step after the corrupted fact is {steps[index + 1]!r}, "
                          f"expected {CORRECTION_MARKER!r}")
    corrupted, correction = steps[index], steps[index + 2]

    # ---- (b) the correction restores the original, real fact -------------
    original_steps = steps[:index] + steps[index + 2:]
    original = STEP_SEP.join(original_steps)
    if original != rationale:
        raise SampleError(f"stripping the insertion yields {original!r}, "
                          f"not the gold rationale {rationale!r}")
    if not enum_matches(correction, meta):
        raise SampleError(f"correction step {correction!r} describes no real object")

    # ---- (a) the injected fact is genuinely wrong ------------------------
    if enum_matches(corrupted, meta, ignore_size=True):
        raise SampleError(f"corrupted step {corrupted!r} still matches an object "
                          f"in the scene: it is not an error")
    wrong_fields = fact_fields(corrupted)
    true_fields = fact_fields(correction)
    differing = [name for name, w, t in zip(('size', 'color', 'shape', 'row', 'col'),
                                            wrong_fields, true_fields) if w != t]
    if 'size' in differing:
        raise SampleError(f"corruption changed the size prefix: {corrupted!r}")
    if len(differing) != 1:
        raise SampleError(f"corruption changed {differing} of {correction!r}, "
                          f"expected exactly one attribute")
    if differing[0] in ('row', 'col'):
        w, t = (wrong_fields[3], true_fields[3]) if differing[0] == 'row' \
            else (wrong_fields[4], true_fields[4])
        if abs(w - t) != 1 or not 1 <= w <= 6:
            raise SampleError(f"{differing[0]} corrupted from {t} to {w}: expected "
                              f"+/-1 inside the occupied 1..6 band")

    # ---- (d) the answer is untouched and still correct -------------------
    check_step_facts(original, meta)
    CHECKERS[qtype](question, answer, original, meta)

    # ---- (c) masks: corrupted tokens unsupervised, correction supervised --
    tok = processor.tokenizer
    seg_tokens = [tok.tokenize(text) for text, _sup in segments]
    flags = [1 if supervised else 0
             for (_text, supervised), toks in zip(segments, seg_tokens)
             for _ in toks]
    r_tokens = [t for toks in seg_tokens for t in toks]
    full = ''.join(text for text, _sup in segments)
    if tok.tokenize(full) != r_tokens:
        raise SampleError("segment-wise tokenization differs from the joined trace: "
                          "separator discipline is broken")

    try:
        input_ids, target_ids, rat_mask, ans_mask = \
            processor.prepare_input_sequence(question, answer, segments)
    except AssertionError as exc:
        raise SampleError(f"corrected trace does not fit MAX_SEQ_LEN: {exc}")
    except ValueError as exc:
        raise SampleError(f"corrected trace has an out-of-vocabulary word: {exc}")

    th_s = input_ids.index(tok.think_start_id)
    if input_ids[th_s + 1: th_s + 1 + len(r_tokens)] != r_tokens:
        raise SampleError("THINK span does not carry the segment tokens")
    # rat_mask[k] supervises the prediction of input_ids[k+1], so rationale
    # token `idx` is supervised iff rat_mask[th_s + idx] == 1.
    for idx, flag in enumerate(flags):
        if rat_mask[th_s + idx] != flag:
            raise SampleError(f"rat_mask[{idx}] = {rat_mask[th_s + idx]}, expected "
                              f"{flag} for token {tok.idx_to_word[r_tokens[idx]]!r}")
    if rat_mask[th_s + len(r_tokens)] != 1:
        raise SampleError("</THINK> must stay supervised")

    lo = sum(len(toks) for toks in seg_tokens[:seg_index])
    hi = lo + len(tok.tokenize(corrupted))
    if any(rat_mask[th_s + idx] for idx in range(lo, hi)):
        raise SampleError(f"corrupted step {corrupted!r} has supervised positions")
    # ' . ' then 'wait' then ' . ' then the corrected fact, all supervised.
    marker_span = range(hi, hi + 3 + len(tok.tokenize(correction)))
    if not all(rat_mask[th_s + idx] for idx in marker_span):
        raise SampleError("the wait marker / corrected fact must stay supervised")
    if tok.idx_to_word[r_tokens[hi + 1]] != CORRECTION_MARKER:
        raise SampleError(f"token after the corrupted step is "
                          f"{tok.idx_to_word[r_tokens[hi + 1]]!r}, not {CORRECTION_MARKER!r}")
    if sum(ans_mask) != len(tok.tokenize(answer)) + 1:
        raise SampleError("answer supervision changed under injection")

    return target_ids, rat_mask, th_s


def masked_trace(processor: TextProcessor, target_ids: Sequence[int],
                 rat_mask: Sequence[int], th_s: int) -> str:
    """'word[mask] word[mask] ...' over the THINK span, for a printed example."""
    tok = processor.tokenizer
    words = []
    for k in range(th_s, len(target_ids)):
        word = tok.idx_to_word.get(target_ids[k], '?')
        words.append(f"{word}[{rat_mask[k]}]")
        if target_ids[k] == tok.think_end_id:
            break
    return ' '.join(words)


def roundtrip_pass(shape_gen, rg, processor, draws: int) -> Tuple[int, List[str]]:
    """A plain string and [(string, True)] must prepare to identical outputs."""
    generators = list(rg.generators.items())
    problems: List[str] = []
    checked = 0
    scenes = max(1, (draws + len(generators) - 1) // len(generators))

    for _ in range(scenes):
        num_shapes = random.randint(MIN_OBJECTS, MAX_OBJECTS)
        _img, meta = shape_gen.generate_multi_shape_image(num_shapes, False)
        for qtype, generator in generators:
            if checked >= draws:
                break
            question, answer, rationale = generator(meta)
            if question is None:
                continue
            try:
                plain = processor.prepare_input_sequence(question, answer, rationale)
                segmented = processor.prepare_input_sequence(
                    question, answer, [(rationale, True)])
            except (AssertionError, ValueError):
                continue
            checked += 1
            if plain != segmented:
                problems.append(f"[roundtrip/{qtype}] segmented output differs from "
                                f"the plain-string output: q={question!r}")
    print(f"round-trip pass: {checked} samples, {len(problems)} mismatches between "
          f"rationale='...' and rationale=[('...', True)]")
    return checked, problems


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def difficulty_of(qtype: str) -> str:
    for difficulty, names in DIFFICULTY_MAP.items():
        if qtype in names:
            return difficulty
    return '?'


def numeric_spread(answers: Counter) -> str:
    """'0:12.3% 1:...' for a Counter whose keys are numeric strings."""
    total = max(1, sum(answers.values()))
    return "  ".join(f"{k}:{100.0 * answers[k] / total:.1f}%"
                     for k in sorted(answers, key=int))


def stress_pass(shape_gen, rg, processor, draws: int, max_objects: int,
                correction_p: float = 0.0) -> Tuple[int, int, int, List[str]]:
    """Hammer every generator at maximum scene density.

    Returns (draws made, overflows, declines, problems). A draw that does not
    fit MAX_SEQ_LEN, or that contains an out-of-vocabulary word, is a hard
    failure: training tokenizes with allow_unk=False and asserts on overflow.
    With `correction_p` > 0 each fitting draw is also error-injected at that
    probability and the corrected trace must fit as well -- this is the proof
    that CORRECTION_RESERVE keeps the insertion inside the context.
    """
    generators = list(rg.generators.items())
    scenes = max(1, draws // max(1, len(generators)))
    made = overflow = declined = 0
    injected = 0
    longest = 0
    longest_corrected = 0
    problems: List[str] = []

    for _ in range(scenes):
        _img, meta = shape_gen.generate_multi_shape_image(max_objects, False)
        for qtype, generator in generators:
            question, answer, rationale = generator(meta)
            made += 1
            if question is None:
                declined += 1
                continue
            try:
                input_ids, _t, _r, _a = processor.prepare_input_sequence(
                    question, answer, rationale)
            except AssertionError:
                overflow += 1
                if overflow <= 3:
                    problems.append(f"[stress/{qtype}] overflow: q={question!r} "
                                    f"r={rationale!r}")
                continue
            except ValueError as exc:
                problems.append(f"[stress/{qtype}] out-of-vocabulary word: {exc}")
                continue
            longest = max(longest, len(input_ids))

            if correction_p <= 0 or random.random() >= correction_p:
                continue
            # Alternate the two call shapes: with the exact word count (what the
            # validator can compute) and without it (what train_model's dataset
            # passes, which falls back to QUESTION_RESERVE). Neither may overflow.
            reserved = (len(question.split()) + len(answer.split())
                        if made % 2 == 0 else None)
            segments = inject_correction(rationale, meta, reserved_words=reserved)
            if segments is None:
                continue
            injected += 1
            try:
                corrected_ids, _t, _r, _a = processor.prepare_input_sequence(
                    question, answer, segments)
            except AssertionError:
                overflow += 1
                if overflow <= 3:
                    problems.append(f"[stress/{qtype}] corrected trace overflow: "
                                    f"q={question!r} segments={segments!r}")
                continue
            except ValueError as exc:
                problems.append(f"[stress/{qtype}] corrected trace out-of-vocabulary "
                                f"word: {exc}")
                continue
            longest_corrected = max(longest_corrected, len(corrected_ids))

    print(f"stress pass: {made} draws over {scenes} scenes at N={max_objects}, "
          f"{overflow} overflows, {declined} declines, longest {longest} tokens "
          f"/ MAX_SEQ_LEN={MAX_SEQ_LEN}")
    if correction_p > 0:
        print(f"  correction_p={correction_p}: {injected} injections applied, "
              f"longest corrected {longest_corrected} tokens "
              f"/ MAX_SEQ_LEN={MAX_SEQ_LEN}")
    return made, overflow, declined, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=1000,
                        help='fresh scenes per question type (default: 1000)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--max-failures', type=int, default=5,
                        help='failures to print per type before summarizing')
    parser.add_argument('--stress', type=int, default=65536,
                        help='question draws at maximum scene density in the '
                             'overflow stress pass (0 disables it)')
    parser.add_argument('--correction_p', type=float, default=0.0,
                        help='probability of error-injecting each verified sample '
                             'and validating the resulting correction trace '
                             '(default: 0, i.e. gold traces only)')
    parser.add_argument('--roundtrip', type=int, default=500,
                        help='samples checked for string/segment equivalence of '
                             'prepare_input_sequence (0 disables it)')
    args = parser.parse_args()
    if not 0.0 <= args.correction_p <= 1.0:
        parser.error('--correction_p must lie in [0, 1]')

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
    scene_sizes: Counter = Counter()
    shared_cell_scenes = 0
    global_max_len = 0
    longest_sample = None
    correction_example = None
    max_qa_words = 0

    for qtype, generator in rg.generators.items():
        answers: Counter = Counter()
        max_len = 0
        skipped = 0
        checked = 0
        type_failures = 0
        corr_tried = 0
        corr_applied = 0
        corr_declined_budget = 0
        corr_declined_facts = 0
        corr_declined_true = 0
        corr_default_extra = 0

        for _ in range(args.samples):
            num_shapes = random.randint(MIN_OBJECTS, MAX_OBJECTS)
            _img, meta = shape_gen.generate_multi_shape_image(num_shapes, False)
            scene_sizes[len(meta)] += 1
            cells = [cell(obj) for obj in meta]
            if len(set(cells)) != len(cells):
                shared_cell_scenes += 1
            for row, col in cells:
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
            max_qa_words = max(max_qa_words,
                               len(question.split()) + len(answer.split()))
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

            # 3. error-injected correction trace (opt-in)
            if args.correction_p <= 0 or random.random() >= args.correction_p:
                continue
            corr_tried += 1
            reserved = len(question.split()) + len(answer.split())
            # How often the budget guard costs an injection when the caller does
            # not pass reserved_words (train_model's call shape).
            if (len(rationale.split()) + reserved <= WORD_BUDGET - CORRECTION_RESERVE
                    < len(rationale.split()) + QUESTION_RESERVE):
                corr_default_extra += 1
            segments = inject_correction(rationale, meta, reserved_words=reserved)
            if segments is None:
                # Three reasons to decline, reported separately and in the order
                # the injector applies them: the trace has no enumeration fact to
                # corrupt, it is within CORRECTION_RESERVE words of the budget, or
                # every corruption drawn happened to describe a real object.
                if not any(ENUM_FACT.fullmatch(s) for s in split_steps(rationale)):
                    corr_declined_facts += 1
                elif len(rationale.split()) + reserved > WORD_BUDGET - CORRECTION_RESERVE:
                    corr_declined_budget += 1
                else:
                    corr_declined_true += 1
                continue
            try:
                target_ids, rat_mask, th_s = check_correction(
                    qtype, question, answer, rationale, segments, meta, processor)
                corr_applied += 1
            except SampleError as exc:
                type_failures += 1
                if type_failures <= args.max_failures:
                    failures.append(f"[{qtype}/correction] {exc}\n"
                                    f"    q={question!r}\n    a={answer!r}\n"
                                    f"    r={rationale!r}\n    segments={segments!r}")
                continue
            used_words.update(''.join(text for text, _sup in segments).split())
            if correction_example is None:
                correction_example = (qtype, question, answer, segments,
                                      masked_trace(processor, target_ids, rat_mask, th_s))

        stats[qtype] = {
            'checked': checked,
            'skipped': skipped,
            'failures': type_failures,
            'answers': answers,
            'max_len': max_len,
            'corr_tried': corr_tried,
            'corr_applied': corr_applied,
            'corr_declined_budget': corr_declined_budget,
            'corr_declined_facts': corr_declined_facts,
            'corr_declined_true': corr_declined_true,
            'corr_default_extra': corr_default_extra,
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
    print(f"max question+answer words: {max_qa_words} / QUESTION_RESERVE="
          f"{QUESTION_RESERVE} (the worst case inject_correction charges when the "
          f"caller does not pass reserved_words)")

    counting_answers = stats['counting']['answers']
    print(f"counting answer distribution:  {numeric_spread(counting_answers)}")
    print(f"count_difference distribution: "
          f"{numeric_spread(stats['count_difference']['answers'])}")
    print(f"ordinal answer distribution:   "
          f"{dict(sorted(stats['ordinal']['answers'].items()))}")

    rc_answers = stats['relative_count']['answers']
    rc_total = max(1, sum(rc_answers.values()))
    rc_ambiguous = sum(c for k, c in rc_answers.items()
                       if k.startswith(f"{CLARIFY_PREFIX} "))
    rc_counts = Counter({k: c for k, c in rc_answers.items() if k.isdigit()})
    print(f"relative_count regimes: unique referent "
          f"{100.0 * (rc_total - rc_ambiguous) / rc_total:.1f}%  ambiguous "
          f"{100.0 * rc_ambiguous / rc_total:.1f}%  (target "
          f"{100 * (1 - AMBIGUOUS_TARGET_PROB):.0f}/"
          f"{100 * AMBIGUOUS_TARGET_PROB:.0f}; scenes too sparse for an "
          f"ambiguous referent decline instead)")
    print(f"relative_count unique-regime distribution: {numeric_spread(rc_counts)}")

    total_scenes = sum(scene_sizes.values())
    print(f"achieved scene density over {total_scenes} scenes "
          f"(requested uniform {MIN_OBJECTS}..{MAX_OBJECTS}):")
    print("  " + "  ".join(f"{n}:{100.0 * scene_sizes[n] / total_scenes:.1f}%"
                           for n in sorted(scene_sizes)))
    print(f"  mean actual N = "
          f"{sum(n * c for n, c in scene_sizes.items()) / total_scenes:.2f}"
          f"   scenes with two objects in one cell: {shared_cell_scenes}")

    print(f"grid rows occupied: {sorted(rows_seen)}   cols occupied: {sorted(cols_seen)}"
          f"   (shape margins keep objects off the border cells)")

    unused = sorted(vocabulary - used_words)
    if unused:
        note = (f"  ({CORRECTION_MARKER!r} is only emitted by inject_correction; "
                f"run with --correction_p > 0)"
                if unused == [CORRECTION_MARKER] else "")
        print(f"vocabulary words never observed ({len(unused)}): {unused}{note}")

    total_tried = total_applied = 0
    if args.correction_p > 0:
        print()
        print(f"error-injected correction traces (--correction_p {args.correction_p}, "
              f"reserve {CORRECTION_RESERVE} words):")
        corr_header = (f"{'question type':<24}{'tried':>8}{'injected':>10}{'rate':>8}"
                       f"{'no-fact':>9}{'near-budget':>13}{'no-error':>10}")
        print(corr_header)
        print("-" * len(corr_header))
        for qtype, s in stats.items():
            tried, applied = s['corr_tried'], s['corr_applied']
            total_tried += tried
            total_applied += applied
            rate = f"{100.0 * applied / tried:.1f}%" if tried else "n/a"
            print(f"{qtype:<24}{tried:>8}{applied:>10}{rate:>8}"
                  f"{s['corr_declined_facts']:>9}{s['corr_declined_budget']:>13}"
                  f"{s['corr_declined_true']:>10}")
        print("-" * len(corr_header))
        overall = f"{100.0 * total_applied / total_tried:.1f}%" if total_tried else "n/a"
        print(f"{'all types':<24}{total_tried:>8}{total_applied:>10}{overall:>8}")
        default_extra = sum(s['corr_default_extra'] for s in stats.values())
        print(f"a caller that does not pass reserved_words (train_model's shape) "
              f"declines {default_extra} more of these ({100.0 * default_extra / max(1, total_tried):.1f}%)")
        if correction_example:
            qtype, q, a, segments, masked = correction_example
            print(f"\nexample corrected trace ({qtype}): q={q!r} a={a!r}")
            for text, supervised in segments:
                print(f"  {'sup ' if supervised else 'UNSUP'} {text!r}")
            print(f"  targets[mask]: {masked}")

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
    if rc_ambiguous == 0:
        problems.append("relative_count: the ambiguous regime never occurred")
    if rc_total - rc_ambiguous == 0:
        problems.append("relative_count: the unique-referent regime never occurred")
    if not any(int(k) >= 2 for k in rc_counts):
        problems.append("relative_count: no count above 1 ever occurred, the "
                        "answers collapse onto 0/1")
    if args.correction_p > 0 and total_applied == 0:
        problems.append("correction injection never applied to a single sample")
    if global_max_len > MAX_SEQ_LEN:
        problems.append(f"max sequence length {global_max_len} exceeds {MAX_SEQ_LEN}")
    if max_qa_words > QUESTION_RESERVE:
        problems.append(f"a question+answer takes {max_qa_words} words, more than "
                        f"QUESTION_RESERVE={QUESTION_RESERVE}: inject_correction's "
                        f"default budget guard is no longer safe")

    if args.roundtrip > 0:
        _rt_checked, rt_problems = roundtrip_pass(shape_gen, rg, processor,
                                                  args.roundtrip)
        failures.extend(rt_problems)
        if rt_problems:
            problems.append(f"{len(rt_problems)} string/segment round-trip mismatches")

    if args.stress > 0:
        made, overflow, declined, stress_problems = stress_pass(
            shape_gen, rg, processor, args.stress, MAX_OBJECTS, args.correction_p)
        failures.extend(stress_problems)
        if overflow:
            problems.append(f"{overflow}/{made} stress draws overflow MAX_SEQ_LEN")
        if any('out-of-vocabulary' in p for p in stress_problems):
            problems.append("stress pass produced out-of-vocabulary words")
        if declined > made // 4:
            problems.append(f"stress pass: {declined}/{made} draws declined at "
                            f"maximum density")

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
