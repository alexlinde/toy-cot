"""
Question generation module for the Toy VLM.

Generates questions, answers, and *enumerate-then-reason* chain-of-thought
rationales. Every rationale first enumerates the objects it needs (in raster
order, with quantized grid coordinates), then performs the reasoning steps that
lead to the answer. Every stated fact is checkable against the scene metadata
(see validate_traces.py).

Trace conventions
-----------------
* lowercase words, single digits, and the literal token ``.`` as step separator
* enumeration item:      ``{color} {shape} at row {r} col {c}``
* size enumeration item: ``{size} {color} {shape} at row {r} col {c}``
* empty filtered set:    ``no {plural descriptor} found``
* counts:                ``count is {n}`` / ``count of {desc} is {n}``
                         / ``count on {side} is {n}``
* parity:                ``{n} is even`` / ``{n} is odd``
* count difference:      ``difference is {d}``
* chain resolution:      ``qualifying {desc} at row {r} col {c}``
                         then ``found`` (yes) or ``none found`` (no)
* ordinal:               ``{ordinal} from {side} is {shape}``
* correction:            ``wait`` (see inject_correction)

All spatial ground truth is computed from the quantized grid coordinates
(shapes.grid_row / shapes.grid_col), never from raw pixel centers.
"""

import random
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from shapes import (COLORS, MAX_OBJECTS, ObjSize, ObjType, ShapeGenerator,
                    grid_col, grid_row)

# ---------------------------------------------------------------------------
# Vocabulary-bearing constants
# ---------------------------------------------------------------------------

SHAPE_NAMES: List[str] = [t.value for t in ObjType]
COLOR_NAMES: List[str] = list(COLORS.keys())
SIZE_NAMES: List[str] = [s.value for s in ObjSize]

RELATIONS: List[str] = ['above', 'below', 'left of', 'right of']
SIDES: List[str] = ['left', 'right', 'top', 'bottom']
OPPOSITE_SIDE: Dict[str, str] = {
    'left': 'right', 'right': 'left', 'top': 'bottom', 'bottom': 'top',
}

# Ordinal words, in rank order: ORDINALS[k - 1] names the k-th object.
ORDINALS: List[str] = ['first', 'second', 'third', 'fourth']

# Head noun used when a descriptor names only a color ("red shape").
GENERIC_NOUN = 'shape'

# Compositional chain depths (number of relation hops) that get their own
# generator, so the evaluator reports one accuracy bucket per depth.
CHAIN_DEPTHS: List[int] = [2, 3, 4]

DIFFICULTY_MAP: Dict[str, List[str]] = {
    'easy': ['existence', 'positional_existence'],
    'medium': ['counting', 'size', 'relative_position', 'side_count_comparison',
               'parity'],
    'hard': (['comparison']
             + [f'compositional_h{h}' for h in CHAIN_DEPTHS]
             + ['count_difference', 'ordinal']),
}

# Literal template words that are not derivable from the lists above.
TEMPLATE_WORDS = frozenset({
    # question templates
    'is', 'a', 'an', 'there', 'any', 'are', 'on', 'the', 'how', 'many', 'than',
    'more', 'that', 'what', 'number', 'difference', 'between', 'and', 'from',
    # trace steps
    'at', 'row', 'col', 'count', 'of', 'no', 'none', 'found', 'qualifying',
    'greater', 'less', 'equal', 'to', 'even', 'odd',
    # answers
    'yes',
})

# ---------------------------------------------------------------------------
# Sequence budget
# ---------------------------------------------------------------------------
# text.prepare_input_sequence wraps question/rationale/answer in 74 fixed
# tokens (BOS + <IMG_START> + 64 image tokens + <IMG_END> + <|user|> +
# <|assistant|> + THINK/FINAL open+close + EOS) inside MAX_SEQ_LEN = 256.
# Drafts whose word count exceeds this are re-drawn so a training sample can
# never overflow the model's context.
WORD_BUDGET = 256 - 74  # 182

# Answer-prior balancing: how many times question parameters are re-drawn while
# trying to hit the coin-flipped target answer.
BALANCE_ATTEMPTS = 20
# Deep chains need more tries: a 'yes' needs an actual chain in the scene.
CHAIN_BALANCE_ATTEMPTS = 40

# Probability that a descriptor is derived from an object actually present in
# the scene (rather than sampled from the full attribute space). Grounded draws
# make "yes" targets reachable; ungrounded draws let absent shapes/colors --
# and therefore counts of zero -- occur naturally.
GROUNDED_DESCRIPTOR_PROB = 0.7
COUNTING_GROUNDED_PROB = 0.5
# Probability that a pair of descriptors is anchored on two different objects
# of the scene (relational questions need both sets non-empty to ever be true).
JOINT_DESCRIPTOR_PROB = 0.85
# Probability that a compositional chain is built backwards from an actual
# chain of objects in the scene. Unanchored chains are almost always 'no' at
# depth 3+, so without scene-anchored draws the answer prior collapses.
CHAIN_ANCHORED_PROB = 0.8
# Largest |a - b| the count-difference generator aims for. Targets are drawn
# uniformly from 0..this, which spreads the answers instead of letting them
# pile up on 0 and 1.
DIFFERENCE_TARGET_MAX = 6
DIFFERENCE_ATTEMPTS = 40
# Share of count-difference draws that deliberately pair a broad descriptor
# with a narrow one (the only way to reach a large difference).
DIFFERENCE_SPREAD_PROB = 0.5


def pluralize(word: str) -> str:
    """Pluralize a shape/head noun: cross -> crosses, else append 's'."""
    return 'crosses' if word == 'cross' else word + 's'


class Descriptor:
    """An object description: a shape, a color, or a color+shape pair.

    An object matches iff every stated attribute matches.
    """

    __slots__ = ('color', 'shape')

    def __init__(self, color: Optional[str] = None, shape: Optional[str] = None):
        if color is None and shape is None:
            raise ValueError("Descriptor needs at least one attribute")
        self.color = color
        self.shape = shape

    @property
    def head(self) -> str:
        return self.shape if self.shape is not None else GENERIC_NOUN

    def text(self, plural: bool = False) -> str:
        head = pluralize(self.head) if plural else self.head
        return f"{self.color} {head}" if self.color is not None else head

    def matches(self, m: Dict[str, Any]) -> bool:
        if self.color is not None and m['color'] != self.color:
            return False
        if self.shape is not None and m['shape'] != self.shape:
            return False
        return True

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Descriptor)
                and self.color == other.color and self.shape == other.shape)

    def __hash__(self) -> int:
        return hash((self.color, self.shape))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Descriptor({self.color!r}, {self.shape!r})"


# ---------------------------------------------------------------------------
# Ground-truth helpers (all quantized)
# ---------------------------------------------------------------------------

def cell_of(m: Dict[str, Any]) -> Tuple[int, int]:
    """Return the quantized (row, col) of an object."""
    return grid_row(m['cy']), grid_col(m['cx'])


def raster_sorted(objs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Raster order: row ascending, then column ascending."""
    return sorted(objs, key=cell_of)


def filter_objects(metadata_list: Sequence[Dict[str, Any]], desc: Descriptor) -> List[Dict[str, Any]]:
    """All objects matching a descriptor, in raster order."""
    return raster_sorted([m for m in metadata_list if desc.matches(m)])


# Sort keys for "k-th from the {side}". The secondary key is the cross-axis,
# ascending, so the order is total whenever all (row, col) pairs are distinct.
AXIS_KEY: Dict[str, Callable[[Tuple[int, int]], Tuple[int, int]]] = {
    'left': lambda rc: (rc[1], rc[0]),
    'right': lambda rc: (-rc[1], rc[0]),
    'top': lambda rc: (rc[0], rc[1]),
    'bottom': lambda rc: (-rc[0], rc[1]),
}


def axis_sorted(objs: Sequence[Dict[str, Any]], side: str) -> List[Dict[str, Any]]:
    """Objects ordered by distance from the named side of the canvas."""
    key = AXIS_KEY[side]
    return sorted(objs, key=lambda m: key(cell_of(m)))


def all_cells_distinct(objs: Sequence[Dict[str, Any]]) -> bool:
    """True iff no two objects share a quantized cell (ordering is then total)."""
    cells = [cell_of(m) for m in objs]
    return len(set(cells)) == len(cells)


def relation_holds(a: Dict[str, Any], b: Dict[str, Any], relation: str) -> bool:
    """True iff `a {relation} b` holds on the quantized grid. Ties are False."""
    ra, ca = cell_of(a)
    rb, cb = cell_of(b)
    if relation == 'above':
        return ra < rb
    if relation == 'below':
        return ra > rb
    if relation == 'left of':
        return ca < cb
    if relation == 'right of':
        return ca > cb
    raise ValueError(f"unknown relation: {relation}")


def on_side(m: Dict[str, Any], side: str) -> bool:
    """True iff the object lies on the named half of the canvas."""
    row, col = cell_of(m)
    if side == 'left':
        return col <= 3
    if side == 'right':
        return col >= 4
    if side == 'top':
        return row <= 3
    if side == 'bottom':
        return row >= 4
    raise ValueError(f"unknown side: {side}")


def comparison_phrase(a: int, b: int) -> str:
    if a > b:
        return 'greater than'
    if a < b:
        return 'less than'
    return 'equal to'


def enumeration_steps(objs: Sequence[Dict[str, Any]], empty_text: str,
                      with_size: bool = False) -> List[str]:
    """Enumerate objects (assumed raster-ordered) as trace steps."""
    if not objs:
        return [empty_text]
    steps = []
    for m in objs:
        row, col = cell_of(m)
        prefix = f"{m['size_category']} " if with_size else ''
        steps.append(f"{prefix}{m['color']} {m['shape']} at row {row} col {col}")
    return steps


def witness_step(a: Dict[str, Any], b: Dict[str, Any], relation: str) -> str:
    """Cite the satisfying pair on the axis the relation is about."""
    ra, ca = cell_of(a)
    rb, cb = cell_of(b)
    if relation in ('above', 'below'):
        return f"row {ra} is {relation} row {rb}"
    return f"col {ca} is {relation} col {cb}"


def _fits(candidate: Tuple[str, str, str]) -> bool:
    """True iff (question, answer, rationale) fits the model's sequence budget."""
    question, answer, rationale = candidate
    total = len(question.split()) + len(answer.split()) + len(rationale.split())
    return total <= WORD_BUDGET


# ---------------------------------------------------------------------------
# Error-injected correction traces
# ---------------------------------------------------------------------------
# inject_correction rewrites a gold trace so that one enumeration fact is stated
# wrongly, caught, and restated correctly:
#
#     ... . red circle at row 2 col 5 . wait . red circle at row 3 col 5 . ...
#          `-------- corrupted --------'         `-------- original --------'
#
# The corrupted step is returned as an *unsupervised* segment: its tokens are in
# the context the model reads, but never in the rationale loss (see
# text.prepare_input_sequence), so the model learns to recover from an error it
# is never taught to produce.

STEP_SEP = ' . '
CORRECTION_MARKER = 'wait'

# Grid coordinates a corrupted fact may name. shapes.grid_row/grid_col span
# 0..7, but the placement margins keep every object center inside rows/cols
# 1..6 (validate_traces reports the occupied band), so a +/-1 shift that would
# leave the band is simply not offered: the wrong coordinate always names a cell
# an object could really occupy, never an impossible one the model could learn
# to spot without looking at the image.
CORRUPT_COORD_MIN = 1
CORRUPT_COORD_MAX = 6

# Corruptions are re-drawn until the wrong fact matches *no* object in the
# scene: a corrupted attribute may coincidentally describe a different object,
# and such a step would be true, not an error to correct.
CORRECTION_ATTEMPTS = 10

# Words the insertion may add: the corrupted fact (7 words, 8 with a size
# prefix), the marker `wait`, and the two ' . ' separators that frame it -- 11
# at most. WORD_BUDGET guarantees a fitting sample *without* the insertion, so
# inject_correction declines whenever the base draw is within this reserve of
# the budget. The margin (16 - 11) is slack, not a second budget.
CORRECTION_RESERVE = 16

# WORD_BUDGET bounds question + answer + rationale together, and the longest
# gold draws already sit exactly on it (a dense compositional_h4 sample tokenizes
# to all 256 positions), so a guard that looked at the rationale alone would let
# ~0.3% of injections overflow the context. inject_correction therefore also
# charges the words spent outside the rationale: a caller that knows them passes
# `reserved_words`, and a caller that does not is charged this worst case.
# 31 = the longest question the generators can build (a depth-4 chain:
# 'is a {D1} {r1} the {D2}' = 9 words, then 7 per extra hop) plus a one-word
# answer. validate_traces asserts no sample ever exceeds it.
QUESTION_RESERVE = 31

_SIZE_ALT = '|'.join(SIZE_NAMES)
_COLOR_ALT = '|'.join(COLOR_NAMES)
_SHAPE_ALT = '|'.join(SHAPE_NAMES)
# Enumeration fact, with the optional size prefix used by the `size` type.
ENUM_FACT = re.compile(
    rf"(?:(?P<size>{_SIZE_ALT}) )?(?P<color>{_COLOR_ALT}) (?P<shape>{_SHAPE_ALT})"
    rf" at row (?P<row>\d) col (?P<col>\d)")

# A corrupted fact as (size, color, shape, row, col); size is None or a word.
Fact = Tuple[Optional[str], str, str, int, int]


def split_steps(rationale: str) -> List[str]:
    """Split a rationale into its ' . '-separated steps (round-trips exactly)."""
    return rationale.split(STEP_SEP)


def join_steps(steps: Sequence[str]) -> str:
    """Inverse of split_steps: no doubled, no missing separators."""
    return STEP_SEP.join(steps)


def fact_text(fact: Fact) -> str:
    """Render an enumeration fact back into its trace step."""
    size, color, shape, row, col = fact
    prefix = f"{size} " if size is not None else ''
    return f"{prefix}{color} {shape} at row {row} col {col}"


def parse_fact(step: str) -> Optional[Fact]:
    """Parse an enumeration step, or None if the step is not one."""
    m = ENUM_FACT.fullmatch(step)
    if m is None:
        return None
    return (m.group('size'), m.group('color'), m.group('shape'),
            int(m.group('row')), int(m.group('col')))


def corruption_options(fact: Fact) -> List[Fact]:
    """Every single-attribute corruption of `fact` that stays well-formed.

    Exactly one attribute changes: the row or column by +/-1 (clamped to the
    occupied grid band, so the shifted coordinate is always a cell an object
    could really sit in), the color, or the shape.
    """
    size, color, shape, row, col = fact
    options: List[Fact] = []
    for r in (row - 1, row + 1):
        if CORRUPT_COORD_MIN <= r <= CORRUPT_COORD_MAX:
            options.append((size, color, shape, r, col))
    for c in (col - 1, col + 1):
        if CORRUPT_COORD_MIN <= c <= CORRUPT_COORD_MAX:
            options.append((size, color, shape, row, c))
    options += [(size, other, shape, row, col)
                for other in COLOR_NAMES if other != color]
    options += [(size, color, other, row, col)
                for other in SHAPE_NAMES if other != shape]
    return options


def fact_names_an_object(fact: Fact, metadata_list: Sequence[Dict[str, Any]]) -> bool:
    """True iff some object in the scene could make `fact` true.

    The size prefix is deliberately ignored: a line that names a real object of
    a different size still half-describes the scene, and an injected error must
    be unambiguously false.
    """
    _size, color, shape, row, col = fact
    return any(m['color'] == color and m['shape'] == shape
               and cell_of(m) == (row, col) for m in metadata_list)


def inject_correction(rationale: str, metadata_list: Sequence[Dict[str, Any]],
                      rng=random, reserved_words: Optional[int] = None
                      ) -> Optional[List[Tuple[str, bool]]]:
    """Corrupt one enumeration fact of `rationale` and correct it right after.

    Returns segments ``[(text, supervised), ...]`` whose concatenation is the
    corrected trace, with the corrupted step -- and only the corrupted step --
    marked unsupervised. Returns None (caller falls back to the plain rationale)
    when the trace has no enumeration fact, when no corruption of the chosen
    fact is unambiguously false, or when the insertion would not fit the
    sequence budget.

    Args:
        rationale: a gold trace, ' . '-separated steps.
        metadata_list: the scene, used to keep the injected error genuinely wrong.
        rng: source of randomness (anything with .choice).
        reserved_words: words the sample already spends outside the rationale
            (question + answer). WORD_BUDGET bounds their *sum*, so passing the
            exact figure buys back the few injections the default declines;
            leaving it None charges the worst case (QUESTION_RESERVE), which
            keeps the no-overflow guarantee for callers that only hold the
            rationale.
    """
    if not rationale:
        return None

    steps = split_steps(rationale)
    assert join_steps(steps) == rationale, (
        f"[questions] step split/join round-trip failed for {rationale!r}")

    candidates = [i for i, step in enumerate(steps) if ENUM_FACT.fullmatch(step)]
    if not candidates:
        return None

    spent = QUESTION_RESERVE if reserved_words is None else reserved_words
    if len(rationale.split()) + spent > WORD_BUDGET - CORRECTION_RESERVE:
        return None

    index = rng.choice(candidates)
    original = parse_fact(steps[index])
    options = corruption_options(original)
    if not options:
        return None

    corrupted = None
    for _ in range(CORRECTION_ATTEMPTS):
        draw = rng.choice(options)
        if not fact_names_an_object(draw, metadata_list):
            corrupted = draw
            break
    if corrupted is None:
        return None

    # Rebuild the trace around the insertion. Each piece carries the separators
    # it owns, so the concatenation is exactly the original token stream with
    # `{corrupted} . wait . ` spliced in front of the corrupted step's original.
    prefix = ''.join(step + STEP_SEP for step in steps[:index])
    corrupted_text = fact_text(corrupted)
    tail = join_steps([CORRECTION_MARKER] + steps[index:])
    segments = [(prefix, True), (corrupted_text, False), (STEP_SEP + tail, True)]
    segments = [(text, supervised) for text, supervised in segments if text]

    expected = join_steps(steps[:index] + [corrupted_text, CORRECTION_MARKER]
                          + steps[index:])
    assert ''.join(text for text, _ in segments) == expected, (
        f"[questions] segment reassembly does not reproduce {expected!r}")
    return segments


class RationaleGenerator:
    """Generates structured rationales (program traces) for questions."""

    def __init__(self):
        self.shape_gen = ShapeGenerator()
        self.generators: Dict[str, Callable[[List[Dict[str, Any]]], Tuple[str, str, str]]] = {
            'existence': self.generate_existence_qa,
            'positional_existence': self.generate_positional_existence_qa,
            'counting': self.generate_counting_qa,
            'size': self.generate_size_qa,
            'relative_position': self.generate_relative_position_qa,
            'side_count_comparison': self.generate_side_count_comparison_qa,
            'parity': self.generate_parity_qa,
            'comparison': self.generate_comparison_qa,
            'compositional_h2': self.generate_compositional_positional_qa,
            'compositional_h3': self.generate_compositional_h3_qa,
            'compositional_h4': self.generate_compositional_h4_qa,
            'count_difference': self.generate_count_difference_qa,
            'ordinal': self.generate_ordinal_qa,
        }

    # ------------------------------------------------------------------
    # Draw helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _descriptor_from(color: str, shape: str, kinds: Sequence[str]) -> Descriptor:
        """Build a descriptor of a randomly chosen kind from a color/shape pair."""
        kind = random.choice(list(kinds))
        if kind == 'shape':
            return Descriptor(None, shape)
        if kind == 'color':
            return Descriptor(color, None)
        return Descriptor(color, shape)

    def _draw_descriptor(self, metadata_list: Sequence[Dict[str, Any]],
                         kinds: Sequence[str] = ('shape', 'color_shape', 'color'),
                         grounded_prob: float = GROUNDED_DESCRIPTOR_PROB) -> Descriptor:
        """Draw a descriptor, usually grounded in an object present in the scene."""
        if metadata_list and random.random() < grounded_prob:
            m = random.choice(list(metadata_list))
            return self._descriptor_from(m['color'], m['shape'], kinds)
        return self._descriptor_from(random.choice(COLOR_NAMES),
                                     random.choice(SHAPE_NAMES), kinds)

    def _draw_descriptor_pair(self, metadata_list: Sequence[Dict[str, Any]],
                              kinds_a: Sequence[str] = ('shape', 'color_shape', 'color'),
                              kinds_b: Sequence[str] = ('shape', 'color_shape', 'color'),
                              joint_prob: float = JOINT_DESCRIPTOR_PROB
                              ) -> Tuple[Descriptor, Descriptor]:
        """Draw two descriptors, usually anchored on two *different* objects.

        Relational questions can only be answered 'yes' when both descriptors
        actually select something, so anchoring on a real pair keeps the
        answer-prior balancing loop able to reach its target.
        """
        if len(metadata_list) >= 2 and random.random() < joint_prob:
            m_a, m_b = random.sample(list(metadata_list), 2)
            return (self._descriptor_from(m_a['color'], m_a['shape'], kinds_a),
                    self._descriptor_from(m_b['color'], m_b['shape'], kinds_b))
        return (self._draw_descriptor(metadata_list, kinds_a),
                self._draw_descriptor(metadata_list, kinds_b))

    def _draw_toward(self, draw: Callable[[], Optional[Tuple[str, str, str]]],
                     target: str, attempts: int = BALANCE_ATTEMPTS
                     ) -> Tuple[str, str, str]:
        """Re-draw question parameters until the computed answer hits `target`.

        Only drafts that fit the sequence budget are ever returned: a draft that
        overflows would make text.prepare_input_sequence assert, so declining
        (None, None, None) -- which every caller treats as "resample" -- is the
        only safe fallback.
        """
        fallback: Optional[Tuple[str, str, str]] = None

        for _ in range(attempts):
            candidate = draw()
            if candidate is None or not _fits(candidate):
                continue
            fallback = candidate
            if candidate[1] == target:
                return candidate

        if fallback is not None:
            return fallback
        return None, None, None

    def _balanced(self, draw: Callable[[], Optional[Tuple[str, str, str]]],
                  attempts: int = BALANCE_ATTEMPTS) -> Tuple[str, str, str]:
        """Answer-prior balancing for yes/no questions: flip a coin, then draw
        toward that answer."""
        return self._draw_toward(draw, 'yes' if random.random() < 0.5 else 'no',
                                 attempts)

    # ------------------------------------------------------------------
    # Small counting utilities (kept for external convenience)
    # ------------------------------------------------------------------

    def count_shapes(self, metadata_list: List[Dict[str, Any]], target_shape: str) -> int:
        """Count how many shapes of a given type are in the image."""
        return sum(1 for m in metadata_list if m['shape'] == target_shape)

    def count_sizes(self, metadata_list: List[Dict[str, Any]], target_size: str) -> int:
        """Count how many shapes of a given size are in the image."""
        return sum(1 for m in metadata_list if m['size_category'] == target_size)

    def count_colors(self, metadata_list: List[Dict[str, Any]], target_color: str) -> int:
        """Count how many objects of a given color are in the image."""
        return sum(1 for m in metadata_list if m['color'] == target_color)

    def exists(self, metadata_list: List[Dict[str, Any]], shape: str) -> bool:
        """Check if a shape exists in the image."""
        return any(m['shape'] == shape for m in metadata_list)

    # ------------------------------------------------------------------
    # 1. existence (easy)
    # ------------------------------------------------------------------

    def generate_existence_qa(self, metadata_list: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        """'is there a red circle' -- enumerate matches, then count."""
        if not metadata_list:
            return None, None, None

        def draw():
            desc = self._draw_descriptor(metadata_list)
            objs = filter_objects(metadata_list, desc)
            steps = enumeration_steps(objs, f"no {desc.text(True)} found")
            steps.append(f"count is {len(objs)}")
            return (f"is there a {desc.text()}",
                    'yes' if objs else 'no',
                    ' . '.join(steps))

        return self._balanced(draw)

    # ------------------------------------------------------------------
    # 2. positional existence (easy)
    # ------------------------------------------------------------------

    def generate_positional_existence_qa(self, metadata_list: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        """'is there a circle on the left' -- enumerate matches, then count on that side."""
        if not metadata_list:
            return None, None, None

        def draw():
            desc = self._draw_descriptor(metadata_list)
            side = random.choice(SIDES)
            objs = filter_objects(metadata_list, desc)
            on_that_side = sum(1 for m in objs if on_side(m, side))

            steps = enumeration_steps(objs, f"no {desc.text(True)} found")
            steps.append(f"count on {side} is {on_that_side}")

            question = random.choice([
                f"is there a {desc.text()} on the {side}",
                f"are there any {desc.text(True)} on the {side}",
            ])
            return question, 'yes' if on_that_side else 'no', ' . '.join(steps)

        return self._balanced(draw)

    # ------------------------------------------------------------------
    # 3. counting (medium)
    # ------------------------------------------------------------------

    def generate_counting_qa(self, metadata_list: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        """'how many blue crosses are there' -- enumerate matches, then count."""
        if not metadata_list:
            return None, None, None

        candidate = None
        for _ in range(BALANCE_ATTEMPTS):
            desc = self._draw_descriptor(metadata_list, grounded_prob=COUNTING_GROUNDED_PROB)
            objs = filter_objects(metadata_list, desc)
            steps = enumeration_steps(objs, f"no {desc.text(True)} found")
            steps.append(f"count is {len(objs)}")
            candidate = (f"how many {desc.text(True)} are there",
                         str(len(objs)),
                         ' . '.join(steps))
            if _fits(candidate):
                return candidate
        return candidate

    # ------------------------------------------------------------------
    # 4. size (medium)
    # ------------------------------------------------------------------

    def generate_size_qa(self, metadata_list: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        """'are there any large shapes' -- enumerate size matches, then count."""
        if not metadata_list:
            return None, None, None

        def draw():
            size = random.choice(SIZE_NAMES)
            objs = raster_sorted([m for m in metadata_list if m['size_category'] == size])
            steps = enumeration_steps(objs, f"no {size} {pluralize(GENERIC_NOUN)} found",
                                      with_size=True)
            steps.append(f"count is {len(objs)}")
            return (f"are there any {size} {pluralize(GENERIC_NOUN)}",
                    'yes' if objs else 'no',
                    ' . '.join(steps))

        return self._balanced(draw)

    # ------------------------------------------------------------------
    # 5. relative position (medium)
    # ------------------------------------------------------------------

    def generate_relative_position_qa(self, metadata_list: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        """'is a circle above a red square' -- enumerate both sets, then cite a witness."""
        if not metadata_list:
            return None, None, None

        def draw():
            desc_a, desc_b = self._draw_descriptor_pair(metadata_list)
            if desc_a == desc_b:
                return None
            relation = random.choice(RELATIONS)

            objs_a = filter_objects(metadata_list, desc_a)
            objs_b = filter_objects(metadata_list, desc_b)

            steps = enumeration_steps(objs_a, f"no {desc_a.text(True)} found")
            steps += enumeration_steps(objs_b, f"no {desc_b.text(True)} found")

            witness = None
            for a in objs_a:
                for b in objs_b:
                    if relation_holds(a, b, relation):
                        witness = (a, b)
                        break
                if witness is not None:
                    break

            if witness is not None:
                steps.append(witness_step(witness[0], witness[1], relation))
                steps.append('found')
                answer = 'yes'
            else:
                steps.append('none found')
                answer = 'no'

            question = f"is a {desc_a.text()} {relation} a {desc_b.text()}"
            return question, answer, ' . '.join(steps)

        return self._balanced(draw)

    # ------------------------------------------------------------------
    # 6. side count comparison (medium)
    # ------------------------------------------------------------------

    def generate_side_count_comparison_qa(self, metadata_list: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        """'are there more circles on the left' -- counts are computed per named side."""
        if not metadata_list:
            return None, None, None

        def draw():
            desc = self._draw_descriptor(metadata_list)
            side = random.choice(SIDES)
            other = OPPOSITE_SIDE[side]

            objs = filter_objects(metadata_list, desc)
            here = sum(1 for m in objs if on_side(m, side))
            elsewhere = sum(1 for m in objs if on_side(m, other))

            steps = enumeration_steps(objs, f"no {desc.text(True)} found")
            steps.append(f"count on {side} is {here}")
            steps.append(f"count on {other} is {elsewhere}")
            steps.append(f"{here} {comparison_phrase(here, elsewhere)} {elsewhere}")

            return (f"are there more {desc.text(True)} on the {side}",
                    'yes' if here > elsewhere else 'no',
                    ' . '.join(steps))

        return self._balanced(draw)

    # ------------------------------------------------------------------
    # 7. comparison (hard)
    # ------------------------------------------------------------------

    def generate_comparison_qa(self, metadata_list: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        """'are there more circles than red shapes' -- scoped counts, then compare."""
        if not metadata_list:
            return None, None, None

        def draw():
            desc_a = self._draw_descriptor(metadata_list)
            desc_b = self._draw_descriptor(metadata_list)
            if desc_a == desc_b:
                return None

            objs_a = filter_objects(metadata_list, desc_a)
            objs_b = filter_objects(metadata_list, desc_b)
            count_a, count_b = len(objs_a), len(objs_b)

            steps = enumeration_steps(objs_a, f"no {desc_a.text(True)} found")
            steps.append(f"count of {desc_a.text(True)} is {count_a}")
            steps += enumeration_steps(objs_b, f"no {desc_b.text(True)} found")
            steps.append(f"count of {desc_b.text(True)} is {count_b}")
            steps.append(f"{count_a} {comparison_phrase(count_a, count_b)} {count_b}")

            return (f"are there more {desc_a.text(True)} than {desc_b.text(True)}",
                    'yes' if count_a > count_b else 'no',
                    ' . '.join(steps))

        return self._balanced(draw)

    # ------------------------------------------------------------------
    # 8. compositional chains, depth h = 2, 3, 4 (hard)
    # ------------------------------------------------------------------

    @staticmethod
    def _walk_chain(metadata_list: Sequence[Dict[str, Any]], hops: int
                    ) -> Optional[List[Dict[str, Any]]]:
        """A chain of objects x1..x(h+1) that really satisfies some relation at
        every hop -- the scene anchor that a 'yes' answer is built from.

        The chain-existence semantics never asks the objects to be distinct, so
        the walk may revisit an object; only *adjacent* objects must differ in
        color+shape, because adjacent descriptors have to differ.
        """
        objs = [random.choice(list(metadata_list))]
        for step in range(hops):
            prev = objs[-1]
            pool = [m for m in metadata_list
                    if cell_of(m) != cell_of(prev)  # else no relation holds
                    and (step == 0
                         or (m['color'], m['shape']) != (prev['color'], prev['shape']))]
            if not pool:
                return None
            objs.append(random.choice(pool))
        return objs

    def _draw_chain(self, metadata_list: Sequence[Dict[str, Any]], hops: int
                    ) -> Optional[Tuple[List[Descriptor], List[str]]]:
        """Draw descriptors D1..D(h+1) and relations r1..rh for a chain question.

        With probability CHAIN_ANCHORED_PROB the chain is read off an actual
        chain of objects in the scene (relations are picked among those that
        really hold), which is what makes a 'yes' reachable at depth 3 and 4.
        """
        # D2..D(h+1) are always color+shape: it keeps every enumeration short
        # and each clause unambiguous about the object it refers to.
        anchor = (self._walk_chain(metadata_list, hops)
                  if random.random() < CHAIN_ANCHORED_PROB else None)
        if anchor is not None:
            relations = []
            for x, y in zip(anchor, anchor[1:]):
                holding = [r for r in RELATIONS if relation_holds(x, y, r)]
                if not holding:  # x and y share a cell: no relation holds
                    return None
                relations.append(random.choice(holding))
            descs = [self._descriptor_from(anchor[0]['color'], anchor[0]['shape'],
                                           ('shape', 'color_shape', 'color'))]
            descs += [Descriptor(o['color'], o['shape']) for o in anchor[1:]]
        else:
            descs = [self._draw_descriptor(metadata_list)]
            descs += [self._draw_descriptor(metadata_list, kinds=('color_shape',))
                      for _ in range(hops)]
            relations = [random.choice(RELATIONS) for _ in range(hops)]

        # Adjacent referents must differ, as in the original 2-hop question
        # ("the red square that is above the red square" refers to nothing).
        if any(descs[j] == descs[j + 1] for j in range(1, len(descs) - 1)):
            return None
        return descs, relations

    @staticmethod
    def _chain_question(descs: Sequence[Descriptor], relations: Sequence[str]) -> str:
        """'is a D1 r1 the D2 that is r2 the D3 that is r3 the D4'."""
        question = f"is a {descs[0].text()} {relations[0]} the {descs[1].text()}"
        for i in range(1, len(relations)):
            question += f" that is {relations[i]} the {descs[i + 1].text()}"
        return question

    def _chain_trace(self, metadata_list: Sequence[Dict[str, Any]],
                     descs: Sequence[Descriptor], relations: Sequence[str]
                     ) -> Tuple[str, str]:
        """Resolve the chain from the tail and return (rationale, answer).

        The answer is 'yes' iff objects x1..x(h+1) exist with xi matching Di and
        r_i(x_i, x_i+1) for every i. The trace enumerates D2..D(h+1), then walks
        back down the chain listing the objects that still qualify at each level,
        and finally cites a witness for the first hop.
        """
        hops = len(relations)
        sets = [filter_objects(metadata_list, d) for d in descs]

        steps: List[str] = []
        for j in range(1, hops + 1):
            steps += enumeration_steps(sets[j], f"no {descs[j].text(True)} found")

        # At loop index j, `qualifying` holds the objects matching descs[j] that
        # start a valid chain from level j all the way to the end.
        qualifying = [b for b in sets[hops - 1]
                      if any(relation_holds(b, c, relations[hops - 1])
                             for c in sets[hops])]
        for j in range(hops - 1, 0, -1):
            if not qualifying:
                steps.append('none found')
                return ' . '.join(steps), 'no'
            for b in qualifying:
                row, col = cell_of(b)
                steps.append(f"qualifying {descs[j].text()} at row {row} col {col}")
            if j == 1:
                break
            qualifying = [b for b in sets[j - 1]
                          if any(relation_holds(b, c, relations[j - 1])
                                 for c in qualifying)]

        steps += enumeration_steps(sets[0], f"no {descs[0].text(True)} found")

        witness = None
        for a in sets[0]:
            for b in qualifying:
                if relation_holds(a, b, relations[0]):
                    witness = (a, b)
                    break
            if witness is not None:
                break

        if witness is None:
            steps.append('none found')
            return ' . '.join(steps), 'no'

        steps.append(witness_step(witness[0], witness[1], relations[0]))
        steps.append('found')
        return ' . '.join(steps), 'yes'

    def generate_chain_qa(self, metadata_list: List[Dict[str, Any]], hops: int
                          ) -> Tuple[str, str, str]:
        """Compositional chain question with `hops` relation hops."""
        if not metadata_list:
            return None, None, None

        def draw():
            drawn = self._draw_chain(metadata_list, hops)
            if drawn is None:
                return None
            descs, relations = drawn
            rationale, answer = self._chain_trace(metadata_list, descs, relations)
            return self._chain_question(descs, relations), answer, rationale

        return self._balanced(draw, attempts=CHAIN_BALANCE_ATTEMPTS)

    def generate_compositional_positional_qa(self, metadata_list: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        """'is a circle left of the red square that is above the blue cross' (h=2)."""
        return self.generate_chain_qa(metadata_list, 2)

    def generate_compositional_h3_qa(self, metadata_list: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        return self.generate_chain_qa(metadata_list, 3)

    def generate_compositional_h4_qa(self, metadata_list: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        return self.generate_chain_qa(metadata_list, 4)

    # ------------------------------------------------------------------
    # 9. parity (medium)
    # ------------------------------------------------------------------

    def generate_parity_qa(self, metadata_list: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        """'are there an even number of red circles' -- count, then read parity."""
        if not metadata_list:
            return None, None, None

        def draw():
            desc = self._draw_descriptor(metadata_list,
                                         grounded_prob=COUNTING_GROUNDED_PROB)
            objs = filter_objects(metadata_list, desc)
            n = len(objs)
            steps = enumeration_steps(objs, f"no {desc.text(True)} found")
            steps.append(f"count is {n}")
            steps.append(f"{n} is {'even' if n % 2 == 0 else 'odd'}")
            return (f"are there an even number of {desc.text(True)}",
                    'yes' if n % 2 == 0 else 'no',
                    ' . '.join(steps))

        return self._balanced(draw)

    # ------------------------------------------------------------------
    # 10. count difference (hard)
    # ------------------------------------------------------------------

    def generate_count_difference_qa(self, metadata_list: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        """'what is the difference between the number of circles and the number
        of red shapes' -- two scoped counts, then their absolute difference."""
        if not metadata_list:
            return None, None, None

        def draw():
            # Half the draws pair a broad, scene-grounded descriptor with a
            # narrow (often absent) one: that is what makes a difference bigger
            # than one or two reachable at all. The rest are drawn like any
            # other counting question, to keep the question surface varied.
            if random.random() < DIFFERENCE_SPREAD_PROB:
                desc_a = self._draw_descriptor(metadata_list, kinds=('shape', 'color'),
                                               grounded_prob=0.9)
                desc_b = self._draw_descriptor(metadata_list, kinds=('color_shape',),
                                               grounded_prob=0.3)
            else:
                desc_a = self._draw_descriptor(metadata_list,
                                               grounded_prob=COUNTING_GROUNDED_PROB)
                desc_b = self._draw_descriptor(metadata_list,
                                               grounded_prob=COUNTING_GROUNDED_PROB)
            if desc_a == desc_b:
                return None

            objs_a = filter_objects(metadata_list, desc_a)
            objs_b = filter_objects(metadata_list, desc_b)
            count_a, count_b = len(objs_a), len(objs_b)

            steps = enumeration_steps(objs_a, f"no {desc_a.text(True)} found")
            steps.append(f"count of {desc_a.text(True)} is {count_a}")
            steps += enumeration_steps(objs_b, f"no {desc_b.text(True)} found")
            steps.append(f"count of {desc_b.text(True)} is {count_b}")
            steps.append(f"difference is {abs(count_a - count_b)}")

            return (f"what is the difference between the number of "
                    f"{desc_a.text(True)} and the number of {desc_b.text(True)}",
                    str(abs(count_a - count_b)),
                    ' . '.join(steps))

        return self._draw_toward(draw, str(random.randint(0, DIFFERENCE_TARGET_MAX)),
                                 attempts=DIFFERENCE_ATTEMPTS)

    # ------------------------------------------------------------------
    # 11. ordinal (hard)
    # ------------------------------------------------------------------

    def generate_ordinal_qa(self, metadata_list: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        """'what shape is third from the left' -- enumerate along the axis, then
        read off the k-th object.

        Declines whenever two objects share a cell: the ordering along the axis
        would not be total and the question would have no defined answer.
        """
        if not metadata_list or not all_cells_distinct(metadata_list):
            return None, None, None

        k = random.randint(1, min(len(ORDINALS), len(metadata_list)))
        side = random.choice(SIDES)
        ordered = axis_sorted(metadata_list, side)

        steps = enumeration_steps(ordered, f"no {pluralize(GENERIC_NOUN)} found")
        target = ordered[k - 1]
        steps.append(f"{ORDINALS[k - 1]} from {side} is {target['shape']}")

        candidate = (f"what shape is {ORDINALS[k - 1]} from the {side}",
                     target['shape'],
                     ' . '.join(steps))
        return candidate if _fits(candidate) else (None, None, None)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def generate_qa_with_rationale(self, metadata_list: List[Dict[str, Any]],
                                   difficulty: str = 'easy') -> Tuple[str, str, str]:
        """Generate question, answer, and rationale based on difficulty level.

        Args:
            metadata_list: List of shape metadata from image
            difficulty: 'easy', 'medium', or 'hard'

        Returns:
            (question, answer, rationale) tuple
        """
        names = DIFFICULTY_MAP.get(difficulty, DIFFICULTY_MAP['easy'])

        # Keep trying until we get a valid result
        for _ in range(10):
            generator = self.generators[random.choice(names)]
            result = generator(metadata_list)
            if result[0] is not None:
                return result

        # Fallback to existence question
        return self.generate_existence_qa(metadata_list)

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------

    def vocabulary(self) -> set:
        """Every word that any question, answer or rationale can ever contain.

        Deterministic and complete -- the tokenizer is built from this set, so
        training never sees an out-of-vocabulary token.
        """
        words = set()

        # shapes (singular + plural) and the generic head noun
        for shape in SHAPE_NAMES:
            words.add(shape)
            words.add(pluralize(shape))
        words.add(GENERIC_NOUN)
        words.add(pluralize(GENERIC_NOUN))

        # colors, sizes
        words.update(COLOR_NAMES)
        words.update(SIZE_NAMES)

        # relations ('left of' contributes two words) and sides
        for relation in RELATIONS:
            words.update(relation.split())
        words.update(SIDES)

        # ordinals ('what shape is third from the left')
        words.update(ORDINALS)

        # numbers: grid coordinates 0-7 and single digits, plus every count /
        # count difference reachable in a scene of up to MAX_OBJECTS objects
        # (10, 11, 12 are single word-level tokens)
        words.update(str(d) for d in range(10))
        words.update(str(n) for n in range(MAX_OBJECTS + 1))
        words.add('.')

        # answers ('no' is also a template word)
        words.add('yes')
        words.add('no')

        # self-correction marker (only ever emitted by inject_correction, but
        # the tokenizer must know it whether or not corrections are enabled)
        words.add(CORRECTION_MARKER)

        # literal template words
        words.update(TEMPLATE_WORDS)

        return words
