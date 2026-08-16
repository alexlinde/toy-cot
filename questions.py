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

All spatial ground truth is computed from the quantized grid coordinates
(shapes.grid_row / shapes.grid_col), never from raw pixel centers.
"""

import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from shapes import COLORS, ObjSize, ObjType, ShapeGenerator, grid_col, grid_row

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

# Head noun used when a descriptor names only a color ("red shape").
GENERIC_NOUN = 'shape'

DIFFICULTY_MAP: Dict[str, List[str]] = {
    'easy': ['existence', 'positional_existence'],
    'medium': ['counting', 'size', 'relative_position', 'side_count_comparison'],
    'hard': ['comparison', 'compositional'],
}

# Literal template words that are not derivable from the lists above.
TEMPLATE_WORDS = frozenset({
    # question templates
    'is', 'a', 'there', 'any', 'are', 'on', 'the', 'how', 'many', 'than',
    'more', 'that',
    # trace steps
    'at', 'row', 'col', 'count', 'of', 'no', 'none', 'found', 'qualifying',
    'greater', 'less', 'equal', 'to',
    # answers
    'yes',
})

# ---------------------------------------------------------------------------
# Sequence budget
# ---------------------------------------------------------------------------
# text.prepare_input_sequence wraps question/rationale/answer in 74 fixed
# tokens (BOS + <IMG_START> + 64 image tokens + <IMG_END> + <|user|> +
# <|assistant|> + THINK/FINAL open+close + EOS) inside MAX_SEQ_LEN = 192.
# Drafts whose word count exceeds this are re-drawn so a training sample can
# never overflow the model's context.
WORD_BUDGET = 118

# Answer-prior balancing: how many times question parameters are re-drawn while
# trying to hit the coin-flipped target answer.
BALANCE_ATTEMPTS = 20

# Probability that a descriptor is derived from an object actually present in
# the scene (rather than sampled from the full attribute space). Grounded draws
# make "yes" targets reachable; ungrounded draws let absent shapes/colors --
# and therefore counts of zero -- occur naturally.
GROUNDED_DESCRIPTOR_PROB = 0.7
COUNTING_GROUNDED_PROB = 0.5
# Probability that a pair of descriptors is anchored on two different objects
# of the scene (relational questions need both sets non-empty to ever be true).
JOINT_DESCRIPTOR_PROB = 0.85


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
            'comparison': self.generate_comparison_qa,
            'compositional': self.generate_compositional_positional_qa,
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

    def _balanced(self, draw: Callable[[], Optional[Tuple[str, str, str]]]) -> Tuple[str, str, str]:
        """Answer-prior balancing for yes/no questions.

        Flip a coin for the target answer, then re-draw the question parameters
        until the computed answer matches it. Drafts that would overflow the
        sequence budget are never returned unless no fitting draft was found.
        """
        target = 'yes' if random.random() < 0.5 else 'no'
        fallback: Optional[Tuple[str, str, str]] = None
        last: Optional[Tuple[str, str, str]] = None

        for _ in range(BALANCE_ATTEMPTS):
            candidate = draw()
            if candidate is None:
                continue
            last = candidate
            if not _fits(candidate):
                continue
            fallback = candidate
            if candidate[1] == target:
                return candidate

        if fallback is not None:
            return fallback
        if last is not None:
            return last
        return None, None, None

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
    # 8. compositional (hard)
    # ------------------------------------------------------------------

    def generate_compositional_positional_qa(self, metadata_list: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        """'is a circle left of the red square that is above the blue cross'

        Yes iff some B-object has a C-object in relation rel2 *and* some
        A-object in relation rel1 to it.
        """
        if not metadata_list:
            return None, None, None

        def draw():
            # B and C are always color+shape: it keeps the trace short and the
            # question unambiguous about which object the clause refers to.
            desc_b, desc_c = self._draw_descriptor_pair(
                metadata_list, kinds_a=('color_shape',), kinds_b=('color_shape',))
            if desc_b == desc_c:
                return None
            desc_a = self._draw_descriptor(metadata_list)
            rel1 = random.choice(RELATIONS)
            rel2 = random.choice(RELATIONS)

            objs_b = filter_objects(metadata_list, desc_b)
            objs_c = filter_objects(metadata_list, desc_c)

            steps = enumeration_steps(objs_b, f"no {desc_b.text(True)} found")
            steps += enumeration_steps(objs_c, f"no {desc_c.text(True)} found")

            qualifying = [b for b in objs_b
                          if any(relation_holds(b, c, rel2) for c in objs_c)]

            question = (f"is a {desc_a.text()} {rel1} the {desc_b.text()} "
                        f"that is {rel2} the {desc_c.text()}")

            if not qualifying:
                steps.append('none found')
                return question, 'no', ' . '.join(steps)

            for b in qualifying:
                row, col = cell_of(b)
                steps.append(f"qualifying {desc_b.text()} at row {row} col {col}")

            objs_a = filter_objects(metadata_list, desc_a)
            steps += enumeration_steps(objs_a, f"no {desc_a.text(True)} found")

            witness = None
            for a in objs_a:
                for b in qualifying:
                    if relation_holds(a, b, rel1):
                        witness = (a, b)
                        break
                if witness is not None:
                    break

            if witness is not None:
                steps.append(witness_step(witness[0], witness[1], rel1))
                steps.append('found')
                answer = 'yes'
            else:
                steps.append('none found')
                answer = 'no'

            return question, answer, ' . '.join(steps)

        return self._balanced(draw)

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

        # digits (grid coordinates 0-7 and counts) and the step separator
        words.update(str(d) for d in range(10))
        words.add('.')

        # answers ('no' is also a template word)
        words.add('yes')
        words.add('no')

        # literal template words
        words.update(TEMPLATE_WORDS)

        return words
