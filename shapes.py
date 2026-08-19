"""
Shape generation module for the Toy VLM.
Handles creation of geometric shapes and related functionality.

Scenes are RGB (64, 64, 3) uint8 images on a black background. Every object
carries a color drawn uniformly from COLORS. All spatial ground truth used by
the question/rationale layer is derived from the quantized 8x8 grid via
grid_row() / grid_col() -- never from raw pixel coordinates.

Both the RNG draws and the rasterization here are a cross-language contract
with the web UI (web/src/lib/shapes.ts): given the same rng, both produce
byte-identical images and metadata. That is why shapes are filled by the
explicit integer scanline rules below instead of PIL -- an algorithm we wrote
down once beats replicating a C library's boundary behavior in a second
language. All fills use INCLUSIVE bounds (x2/y2 are filled), matching what
PIL.ImageDraw did for the square and cross; circle and triangle boundaries
moved by single pixels in the swap (accuracy re-verified against the trained
checkpoint at the time).
"""

import math
import numpy as np
import random
from enum import Enum
from typing import List, Tuple, Dict, Any

from rng import SceneRandom  # noqa: F401  (re-exported: canonical seeded-scene rng)

# Image constants
IMAGE_SIZE = 64

# Spatial quantization: the canvas is split into an 8x8 grid of 8x8 pixel cells.
GRID_CELLS = 8
CELL_SIZE = IMAGE_SIZE // GRID_CELLS  # 8

# Scene density -- single source of truth for every script that builds scenes.
# MAX_OBJECTS also sizes the auxiliary count heads (see model.py), so it must
# stay the only place the ceiling is written down.
MIN_OBJECTS = 1
MAX_OBJECTS = 12

# Dense scenes (crossover experiment): above DENSE_THRESHOLD objects the 64x64
# canvas cannot host large shapes without starving the rejection sampler, so
# dense draws are restricted to the two smaller size categories (see
# DENSE_SIZES below) and small shapes get a tighter exclusion margin.
DENSE_THRESHOLD = 6

# Object colors (name -> RGB triple).
COLORS = {
    'red': (220, 50, 50),
    'green': (60, 180, 80),
    'blue': (60, 90, 220),
    'yellow': (235, 200, 50),
}


def grid_row(cy: int) -> int:
    """Quantize a pixel y-coordinate to its grid row (0..7)."""
    return int(cy) // CELL_SIZE


def grid_col(cx: int) -> int:
    """Quantize a pixel x-coordinate to its grid column (0..7)."""
    return int(cx) // CELL_SIZE


class ObjType(Enum):
    """Object type enumeration for shape classification."""
    SQUARE = "square"
    CIRCLE = "circle"
    CROSS = "cross"
    TRIANGLE = "triangle"

class ObjSize(Enum):
    """Object size enumeration for shape classification."""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

# Bands are gap-separated (not contiguous): a contiguous 8-15/16-25/26-35
# split makes boundary sizes (e.g. 15px vs 16px) pixel-identical yet
# differently labeled, which is irreducible label noise for the 'size'
# question type. Gaps between bands give each category a perceptual margin.
SIZE_RANGES = {
    ObjSize.SMALL: (8, 12),
    ObjSize.MEDIUM: (16, 22),
    ObjSize.LARGE: (28, 35)
}

# --- dense-scene placement tuning (see DENSE_THRESHOLD above) ---------------
# Size categories a dense scene may draw from, and their relative weights.
# Weights are INTEGERS (17:3 == the old 0.85:0.15) because the seeded-scene
# RNG contract bans floats; stdlib random.choices picks identically for
# proportional weights, so legacy training streams are unchanged.
DENSE_SIZES = [ObjSize.SMALL, ObjSize.MEDIUM]
DENSE_SIZE_WEIGHTS = [17, 3]
# Exclusion margin added to size//2 when reserving a shape's box. The drawn
# shape never extends further than size//2 from its center, so a margin of 1
# still guarantees strict no-pixel-overlap (numpy's exclusive upper slice bound
# costs one pixel on the far side); the default 3 keeps sparse scenes airy.
EXCLUSION_MARGIN = 3
DENSE_EXCLUSION_MARGIN = 1
# Rejection-sampling budget per requested object. Generous budgets keep the
# achieved object count close to the requested one at every density (the old
# budget of 10 delivered a full 6-object scene only ~3% of the time, which put
# a hole in the density curve the crossover experiment sweeps over).
ATTEMPTS_PER_OBJECT = 80
DENSE_ATTEMPTS_PER_OBJECT = 120

# Draw orders for the rng contract: the index an rng.choice() call maps to a
# value through is part of what makes a seed reproduce a scene, so the lists
# are pinned here rather than left to enum/dict iteration in two languages.
SHAPE_DRAW_ORDER = list(ObjType)            # square, circle, cross, triangle
SIZE_DRAW_ORDER = list(ObjSize)             # small, medium, large
COLOR_DRAW_ORDER = list(COLORS.keys())      # red, green, blue, yellow


class _StdlibRng:
    """Adapter giving module-level `random` the SceneRandom draw interface.

    Default rng for callers outside the seeded-scene contract (training,
    evaluation): their draws go through the same stdlib calls as before this
    interface existed, so a given random.seed() still yields the exact
    pre-existing stream.
    """

    def randint(self, a, b):
        return random.randint(a, b)

    def choice(self, seq):
        return random.choice(seq)

    def weighted_choice(self, seq, weights):
        return random.choices(seq, weights=weights, k=1)[0]


_STDLIB_RNG = _StdlibRng()


# --- integer rasterizer (cross-language contract, see module docstring) -----

def _fill_rect(img: np.ndarray, color, x1: int, y1: int, x2: int, y2: int):
    """Fill an axis-aligned rectangle, bounds INCLUSIVE, clipped to canvas."""
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(IMAGE_SIZE - 1, x2), min(IMAGE_SIZE - 1, y2)
    if x1 <= x2 and y1 <= y2:
        img[y1:y2 + 1, x1:x2 + 1] = color


def _fill_circle(img: np.ndarray, color, cx: int, cy: int, r: int):
    """Fill the disk (x-cx)^2 + (y-cy)^2 <= r^2 + r, one inclusive row-span
    per scanline: |dx| <= isqrt(r^2 + r - dy^2).

    The +r slack is the classic 'rounder disk' rule: the bare <= r^2 disk has
    width-1 rows at the cardinal extremes, which at this scale read as tiny
    cross-like nubs (a hazard for the circle/cross distinction the cross's
    slim-arm tuning exists to protect). With the slack the silhouette tracks
    PIL's ellipse -- what the checkpoints were trained on -- more closely too.
    """
    t = r * r + r
    for y in range(cy - r, cy + r + 1):
        dy = y - cy
        w = _isqrt(t - dy * dy)
        _fill_rect(img, color, cx - w, y, cx + w, y)


def _fill_triangle(img: np.ndarray, color, cx: int, cy: int, half: int):
    """Fill the isoceles triangle with apex (cx, cy-half) and base corners
    (cx +/- half, cy + half): row dy below the apex spans |dx| <= dy // 2
    (the exact integer interpolation (half * dy) // (2 * half))."""
    for dy in range(0, 2 * half + 1):
        w = dy // 2
        _fill_rect(img, color, cx - w, cy - half + dy, cx + w, cy - half + dy)


def _isqrt(n: int) -> int:
    """floor(sqrt(n)); named so the TS port has an explicit counterpart."""
    return math.isqrt(n)


class ShapeGenerator:
    """Generates simple geometric shapes as RGB images."""

    def _draw_single_shape(self, img: np.ndarray, shape_type: ObjType, size: int,
                           cx: int, cy: int, color_name: str) -> Dict[str, Any]:
        """Draw a single axis-aligned shape onto the RGB image and return its metadata."""
        color = COLORS[color_name]

        if shape_type == ObjType.SQUARE:
            half = size // 2
            _fill_rect(img, color, cx - half, cy - half, cx + half, cy + half)

        elif shape_type == ObjType.CIRCLE:
            _fill_circle(img, color, cx, cy, size // 2)

        elif shape_type == ObjType.CROSS:
            # Arms must stay proportionally slim at every size so the plus
            # silhouette reads distinctly from a circle even at SMALL sizes
            # (size // 8 made bars nearly as thick as the shape is wide,
            # which read as a blob indistinguishable from a circle). Divisor
            # 7 was chosen over 6/5 by sweeping IoU-vs-best-matching-circle
            # across sizes 8-15: it matches or beats divisor 6 at every size
            # (strictly better at 12-15, where divisor 6 ties the old //8
            # formula because both floor to thickness 2) while leaving
            # MEDIUM/LARGE sizes (16+) unchanged from divisor 6.
            thickness = max(1, size // 7)
            length = size // 2
            _fill_rect(img, color, cx - length, cy - thickness,
                       cx + length, cy + thickness)     # horizontal bar
            _fill_rect(img, color, cx - thickness, cy - length,
                       cx + thickness, cy + length)     # vertical bar

        elif shape_type == ObjType.TRIANGLE:
            _fill_triangle(img, color, cx, cy, size // 2)

        metadata = {
            'shape': shape_type.value,
            'color': color_name,
            'size': size,
            'cx': cx,
            'cy': cy,
        }

        return metadata

    def generate_multi_shape_image(self, num_shapes: int, add_noise: bool,
                                   rng=None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Generate a 64x64 RGB image with multiple shapes and return metadata.

        Args:
            num_shapes: requested object count (rejection sampling may deliver
                fewer; see ATTEMPTS_PER_OBJECT).
            add_noise: add per-channel Gaussian noise (training only; draws
                from np.random and is NOT part of the seeded-scene contract).
            rng: draw source. None (the default) keeps the historical
                behavior -- module-level `random`, so training/eval streams
                under random.seed() are unchanged. The seeded-scene contract
                (GUI seeds, web parity) passes a rng.SceneRandom instead.

        Returns:
            image: RGB numpy array of shape (64, 64, 3) with values 0-255
            metadata_list: List of dicts with keys
                shape, color, size, cx, cy, size_category
        """
        if rng is None:
            rng = _STDLIB_RNG

        # Initialize RGB image (black background)
        img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        metadata_list = []

        # Track occupied regions to avoid too much overlap
        occupied = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)

        # Dense scenes need smaller shapes, a tighter exclusion margin and a
        # bigger rejection budget, or the sampler starves well short of the
        # requested count.
        dense = num_shapes > DENSE_THRESHOLD

        attempts = 0
        max_attempts = num_shapes * (DENSE_ATTEMPTS_PER_OBJECT if dense
                                     else ATTEMPTS_PER_OBJECT)

        while len(metadata_list) < num_shapes and attempts < max_attempts:
            attempts += 1

            # Random shape, size and color. Draw kinds, order of draws, and
            # the *_DRAW_ORDER lists are all part of the seeded-scene
            # contract with web/src/lib/shapes.ts.
            shape_type = rng.choice(SHAPE_DRAW_ORDER)
            if dense:
                size_category = rng.weighted_choice(DENSE_SIZES, DENSE_SIZE_WEIGHTS)
            else:
                size_category = rng.choice(SIZE_DRAW_ORDER)
            size_min, size_max = SIZE_RANGES[size_category]
            size = rng.randint(size_min, size_max)
            color_name = rng.choice(COLOR_DRAW_ORDER)

            # Random position with margin (unchanged: it is what keeps objects
            # off the border cells, i.e. what defines the effective grid).
            margin = size // 2 + 5
            if margin >= IMAGE_SIZE // 2:
                continue

            cx = rng.randint(margin, IMAGE_SIZE - margin)
            cy = rng.randint(margin, IMAGE_SIZE - margin)

            # Check overlap (no overlap allowed for clearer images)
            if dense and size_category is ObjSize.SMALL:
                check_radius = size // 2 + DENSE_EXCLUSION_MARGIN
            else:
                check_radius = size // 2 + EXCLUSION_MARGIN
            y1, y2 = max(0, cy - check_radius), min(IMAGE_SIZE, cy + check_radius)
            x1, x2 = max(0, cx - check_radius), min(IMAGE_SIZE, cx + check_radius)
            overlap_ratio = occupied[y1:y2, x1:x2].sum() / max(1, (y2-y1) * (x2-x1))

            if overlap_ratio > 0.0:  # No overlap allowed
                continue

            # Draw the shape
            metadata = self._draw_single_shape(img, shape_type, size, cx, cy, color_name)
            metadata['size_category'] = size_category.value
            metadata_list.append(metadata)

            # Mark region as occupied
            occupied[y1:y2, x1:x2] = True

        # Add slight per-channel noise
        if add_noise:
            noise = np.random.normal(0, 5, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return img, metadata_list


    def get_available_shapes(self) -> List[str]:
        """Return list of available shape types."""
        return [shape.value for shape in ObjType]


    def get_available_sizes(self) -> List[str]:
        """Return list of available size categories."""
        return [size.value for size in ObjSize]


    def get_available_colors(self) -> List[str]:
        """Return list of available color names."""
        return list(COLORS.keys())
