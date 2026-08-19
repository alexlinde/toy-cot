/**
 * Scene generation: TS half of the cross-language contract with shapes.py.
 * Given the same seed, makeSeededScene here and make_seeded_scene in the
 * Python GUI produce byte-identical images and identical metadata. Golden
 * fixtures: web/fixtures/scenes.json (tests/shapes.test.ts).
 */

import { SceneRandom } from "./rng";

export const IMAGE_SIZE = 64;
export const GRID_CELLS = 8;
export const CELL_SIZE = IMAGE_SIZE / GRID_CELLS;
export const MIN_OBJECTS = 1;
export const MAX_OBJECTS = 12;

export type ShapeName = "square" | "circle" | "cross" | "triangle";
export type ColorName = "red" | "green" | "blue" | "yellow";
export type SizeName = "small" | "medium" | "large";

export const COLORS: Record<ColorName, [number, number, number]> = {
  red: [220, 50, 50],
  green: [60, 180, 80],
  blue: [60, 90, 220],
  yellow: [235, 200, 50],
};

export interface SceneObject {
  shape: ShapeName;
  color: ColorName;
  size: number;
  cx: number;
  cy: number;
  size_category: SizeName;
}

export interface Scene {
  seed: number;
  /** 64*64*3 row-major RGB bytes. */
  image: Uint8Array;
  metadata: SceneObject[];
}

// Draw orders for the rng contract: the index an rng.choice() call maps to a
// value through is part of what makes a seed reproduce a scene, so the lists
// are pinned here exactly as SHAPE_DRAW_ORDER / SIZE_DRAW_ORDER /
// COLOR_DRAW_ORDER pin them in shapes.py.
const SHAPE_DRAW_ORDER: readonly ShapeName[] = [
  "square",
  "circle",
  "cross",
  "triangle",
];
const SIZE_DRAW_ORDER: readonly SizeName[] = ["small", "medium", "large"];
const COLOR_DRAW_ORDER: readonly ColorName[] = [
  "red",
  "green",
  "blue",
  "yellow",
];

const SIZE_RANGES: Record<SizeName, [number, number]> = {
  small: [8, 12],
  medium: [16, 22],
  large: [28, 35],
};

// Dense-scene placement tuning (shapes.py DENSE_* constants).
const DENSE_THRESHOLD = 6;
const DENSE_SIZES: readonly SizeName[] = ["small", "medium"];
const DENSE_SIZE_WEIGHTS: readonly number[] = [17, 3];
const EXCLUSION_MARGIN = 3;
const DENSE_EXCLUSION_MARGIN = 1;
const ATTEMPTS_PER_OBJECT = 80;
const DENSE_ATTEMPTS_PER_OBJECT = 120;

export function gridRow(cy: number): number {
  return Math.floor(cy / CELL_SIZE);
}

export function gridCol(cx: number): number {
  return Math.floor(cx / CELL_SIZE);
}

// --- integer rasterizer (cross-language contract, see shapes.py) -----------

/** floor(sqrt(n)) for a non-negative integer; counterpart of math.isqrt. */
function isqrt(n: number): number {
  let w = Math.floor(Math.sqrt(n));
  while ((w + 1) * (w + 1) <= n) w++;
  while (w * w > n) w--;
  return w;
}

/** Fill an axis-aligned rectangle, bounds INCLUSIVE, clipped to canvas. */
function fillRect(
  img: Uint8Array,
  color: readonly [number, number, number],
  x1: number,
  y1: number,
  x2: number,
  y2: number,
): void {
  const xa = Math.max(0, x1);
  const ya = Math.max(0, y1);
  const xb = Math.min(IMAGE_SIZE - 1, x2);
  const yb = Math.min(IMAGE_SIZE - 1, y2);
  if (xa > xb || ya > yb) return;
  for (let y = ya; y <= yb; y++) {
    let i = (y * IMAGE_SIZE + xa) * 3;
    for (let x = xa; x <= xb; x++) {
      img[i] = color[0];
      img[i + 1] = color[1];
      img[i + 2] = color[2];
      i += 3;
    }
  }
}

/** Disk (x-cx)^2 + (y-cy)^2 <= r^2 + r, one inclusive row-span per scanline. */
function fillCircle(
  img: Uint8Array,
  color: readonly [number, number, number],
  cx: number,
  cy: number,
  r: number,
): void {
  const t = r * r + r;
  for (let y = cy - r; y <= cy + r; y++) {
    const dy = y - cy;
    const w = isqrt(t - dy * dy);
    fillRect(img, color, cx - w, y, cx + w, y);
  }
}

/** Isoceles triangle, apex (cx, cy-half), base corners (cx +/- half, cy+half). */
function fillTriangle(
  img: Uint8Array,
  color: readonly [number, number, number],
  cx: number,
  cy: number,
  half: number,
): void {
  for (let dy = 0; dy <= 2 * half; dy++) {
    const w = Math.floor(dy / 2);
    fillRect(img, color, cx - w, cy - half + dy, cx + w, cy - half + dy);
  }
}

/** Port of ShapeGenerator._draw_single_shape. */
function drawSingleShape(
  img: Uint8Array,
  shape: ShapeName,
  size: number,
  cx: number,
  cy: number,
  colorName: ColorName,
): void {
  const color = COLORS[colorName];

  if (shape === "square") {
    const half = Math.floor(size / 2);
    fillRect(img, color, cx - half, cy - half, cx + half, cy + half);
  } else if (shape === "circle") {
    fillCircle(img, color, cx, cy, Math.floor(size / 2));
  } else if (shape === "cross") {
    // Arms stay proportionally slim at every size (divisor 7, see shapes.py).
    const thickness = Math.max(1, Math.floor(size / 7));
    const length = Math.floor(size / 2);
    fillRect(img, color, cx - length, cy - thickness, cx + length, cy + thickness);
    fillRect(img, color, cx - thickness, cy - length, cx + thickness, cy + length);
  } else {
    fillTriangle(img, color, cx, cy, Math.floor(size / 2));
  }
}

/** Port of ShapeGenerator.generate_multi_shape_image (add_noise is never
 * used on the web: noise is a training-only feature outside the contract). */
export function generateMultiShapeImage(
  numShapes: number,
  rng: SceneRandom,
): { image: Uint8Array; metadata: SceneObject[] } {
  // RGB image, black background.
  const image = new Uint8Array(IMAGE_SIZE * IMAGE_SIZE * 3);
  const metadata: SceneObject[] = [];

  // Occupied regions, so shapes never overlap.
  const occupied = new Uint8Array(IMAGE_SIZE * IMAGE_SIZE);

  const dense = numShapes > DENSE_THRESHOLD;

  let attempts = 0;
  const maxAttempts =
    numShapes * (dense ? DENSE_ATTEMPTS_PER_OBJECT : ATTEMPTS_PER_OBJECT);

  while (metadata.length < numShapes && attempts < maxAttempts) {
    attempts += 1;

    // Draw kinds, order of draws and the *_DRAW_ORDER lists are all part of
    // the seeded-scene contract with shapes.py.
    const shape = rng.choice(SHAPE_DRAW_ORDER);
    const sizeCategory = dense
      ? rng.weightedChoice(DENSE_SIZES, DENSE_SIZE_WEIGHTS)
      : rng.choice(SIZE_DRAW_ORDER);
    const [sizeMin, sizeMax] = SIZE_RANGES[sizeCategory];
    const size = rng.randint(sizeMin, sizeMax);
    const colorName = rng.choice(COLOR_DRAW_ORDER);

    // Position margin: what keeps objects off the border cells.
    const margin = Math.floor(size / 2) + 5;
    if (margin >= IMAGE_SIZE / 2) continue;

    const cx = rng.randint(margin, IMAGE_SIZE - margin);
    const cy = rng.randint(margin, IMAGE_SIZE - margin);

    const checkRadius =
      Math.floor(size / 2) +
      (dense && sizeCategory === "small"
        ? DENSE_EXCLUSION_MARGIN
        : EXCLUSION_MARGIN);
    // Upper bounds are EXCLUSIVE here (numpy slice semantics), unlike the
    // inclusive bounds the rasterizer fills with.
    const y1 = Math.max(0, cy - checkRadius);
    const y2 = Math.min(IMAGE_SIZE, cy + checkRadius);
    const x1 = Math.max(0, cx - checkRadius);
    const x2 = Math.min(IMAGE_SIZE, cx + checkRadius);

    let overlap = false;
    for (let y = y1; y < y2 && !overlap; y++) {
      const row = y * IMAGE_SIZE;
      for (let x = x1; x < x2; x++) {
        if (occupied[row + x]) {
          overlap = true;
          break;
        }
      }
    }
    if (overlap) continue; // no overlap allowed

    drawSingleShape(image, shape, size, cx, cy, colorName);
    metadata.push({
      shape,
      color: colorName,
      size,
      cx,
      cy,
      size_category: sizeCategory,
    });

    for (let y = y1; y < y2; y++) {
      occupied.fill(1, y * IMAGE_SIZE + x1, y * IMAGE_SIZE + x2);
    }
  }

  return { image, metadata };
}

/** Port of test_model.make_seeded_scene: the whole derivation, object count
 * included, runs off one SceneRandom(seed) stream. */
export function makeSeededScene(seed: number): Scene {
  const rng = new SceneRandom(seed);
  const numShapes = rng.randint(MIN_OBJECTS, MAX_OBJECTS);
  const { image, metadata } = generateMultiShapeImage(numShapes, rng);
  return { seed, image, metadata };
}

/** Random seed for a fresh scene (mirrors random.randrange(1_000_000)). */
export function drawSceneSeed(): number {
  return Math.floor(Math.random() * 1_000_000);
}
