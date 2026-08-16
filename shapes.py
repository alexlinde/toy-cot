"""
Shape generation module for the Toy VLM.
Handles creation of geometric shapes and related functionality.

Scenes are RGB (64, 64, 3) uint8 images on a black background. Every object
carries a color drawn uniformly from COLORS. All spatial ground truth used by
the question/rationale layer is derived from the quantized 8x8 grid via
grid_row() / grid_col() -- never from raw pixel coordinates.
"""

import numpy as np
import random
from enum import Enum
from typing import List, Tuple, Dict, Any
from PIL import Image, ImageDraw

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

SIZE_RANGES = {
    ObjSize.SMALL: (8, 15),
    ObjSize.MEDIUM: (16, 25),
    ObjSize.LARGE: (26, 35)
}

# --- dense-scene placement tuning (see DENSE_THRESHOLD above) ---------------
# Size categories a dense scene may draw from, and their relative weights.
DENSE_SIZES = [ObjSize.SMALL, ObjSize.MEDIUM]
DENSE_SIZE_WEIGHTS = [0.85, 0.15]
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

class ShapeGenerator:
    """Generates simple geometric shapes as RGB images."""

    def _draw_single_shape(self, img: np.ndarray, shape_type: ObjType, size: int,
                           cx: int, cy: int, color_name: str) -> Dict[str, Any]:
        """Draw a single axis-aligned shape onto the RGB image and return its metadata."""
        # Create a temporary single-channel mask image for the shape
        shape_mask = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE), 0)
        draw = ImageDraw.Draw(shape_mask)

        if shape_type == ObjType.SQUARE:
            half = size // 2
            x1, y1 = cx - half, cy - half
            x2, y2 = cx + half, cy + half
            draw.rectangle([x1, y1, x2, y2], fill=255)

        elif shape_type == ObjType.CIRCLE:
            radius = size // 2
            x1, y1 = cx - radius, cy - radius
            x2, y2 = cx + radius, cy + radius
            draw.ellipse([x1, y1, x2, y2], fill=255)

        elif shape_type == ObjType.CROSS:
            thickness = max(2, size // 8)
            length = size // 2
            # Horizontal bar
            hx1, hy1 = cx - length, cy - thickness
            hx2, hy2 = cx + length, cy + thickness
            draw.rectangle([hx1, hy1, hx2, hy2], fill=255)
            # Vertical bar
            vx1, vy1 = cx - thickness, cy - length
            vx2, vy2 = cx + thickness, cy + length
            draw.rectangle([vx1, vy1, vx2, vy2], fill=255)

        elif shape_type == ObjType.TRIANGLE:
            half = size // 2
            x1, y1 = cx, cy - half  # top
            x2, y2 = cx - half, cy + half  # bottom-left
            x3, y3 = cx + half, cy + half  # bottom-right
            draw.polygon([(x1, y1), (x2, y2), (x3, y3)], fill=255)

        # Composite the mask onto the RGB canvas using the object's color
        mask_np = np.array(shape_mask, dtype=np.uint8)
        img[mask_np > 0] = COLORS[color_name]

        metadata = {
            'shape': shape_type.value,
            'color': color_name,
            'size': size,
            'cx': cx,
            'cy': cy,
        }

        return metadata

    def generate_multi_shape_image(self, num_shapes: int, add_noise: bool) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Generate a 64x64 RGB image with multiple shapes and return metadata.

        Returns:
            image: RGB numpy array of shape (64, 64, 3) with values 0-255
            metadata_list: List of dicts with keys
                shape, color, size, cx, cy, size_category
        """

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

            # Random shape, size and color
            shape_type = random.choice(list(ObjType))
            if dense:
                size_category = random.choices(DENSE_SIZES,
                                               weights=DENSE_SIZE_WEIGHTS, k=1)[0]
            else:
                size_category = random.choice(list(ObjSize))
            size_min, size_max = SIZE_RANGES[size_category]
            size = random.randint(size_min, size_max)
            color_name = random.choice(list(COLORS.keys()))

            # Random position with margin (unchanged: it is what keeps objects
            # off the border cells, i.e. what defines the effective grid).
            margin = size // 2 + 5
            if margin >= IMAGE_SIZE // 2:
                continue

            cx = random.randint(margin, IMAGE_SIZE - margin)
            cy = random.randint(margin, IMAGE_SIZE - margin)

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
