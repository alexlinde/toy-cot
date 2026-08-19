/**
 * Attention-overlay math and drawing, ported from test_model.py's
 * render_attention_overlay. Pure functions -- no DOM/canvas state kept here
 * beyond the 2D context passed in, so they're trivially unit-testable even
 * though no test is required.
 */

import { GRID_CELLS, NUM_HEADS, NUM_IMG_TOKENS, NUM_LAYERS } from "./constants";
import type { LayerChoice } from "./store";

/** Overlay canvas backing-store size (see components/SceneCanvas.tsx for why
 * it's larger than the 64x64 base canvas: a 2px argmax outline needs real
 * pixels to draw with). */
export const OVERLAY_CANVAS_SIZE = 512;

const HEAT_RGB = "255, 0, 255";
const HEAT_MAX_ALPHA = 0.7;
const HEAT_GAMMA = 0.55;

/**
 * Reduce a token's raw attention -- Float32Array(NUM_LAYERS * NUM_HEADS *
 * NUM_IMG_TOKENS), layer-major then head then cell -- to the NUM_IMG_TOKENS
 * per-cell weights the overlay paints. 'mean' averages all
 * layer*head rows; a layer index averages just that layer's NUM_HEADS rows.
 */
export function computeAttentionWeights(
  attn: Float32Array,
  layerChoice: LayerChoice,
): Float32Array {
  const weights = new Float32Array(NUM_IMG_TOKENS);
  const rows: number[] = [];
  if (layerChoice === "mean") {
    for (let r = 0; r < NUM_LAYERS * NUM_HEADS; r++) rows.push(r);
  } else {
    for (let h = 0; h < NUM_HEADS; h++) rows.push(layerChoice * NUM_HEADS + h);
  }
  for (const row of rows) {
    const base = row * NUM_IMG_TOKENS;
    for (let c = 0; c < NUM_IMG_TOKENS; c++) weights[c] += attn[base + c];
  }
  const n = rows.length || 1;
  for (let c = 0; c < NUM_IMG_TOKENS; c++) weights[c] /= n;
  return weights;
}

/**
 * Draw (or, given null/empty weights, clear) the heat map + white argmax
 * outline onto a GRID_CELLS x GRID_CELLS overlay. `ctx` is expected to back
 * an OVERLAY_CANVAS_SIZE x OVERLAY_CANVAS_SIZE canvas.
 *
 * Alpha is normalized per call against the strongest cell (which gets
 * HEAT_MAX_ALPHA), gamma-lifted so the mid-range of a peaky distribution
 * stays visible rather than washing out to nothing.
 */
export function drawAttentionOverlay(
  ctx: CanvasRenderingContext2D,
  weights: Float32Array | null,
): void {
  const size = OVERLAY_CANVAS_SIZE;
  ctx.clearRect(0, 0, size, size);
  if (!weights || weights.length === 0) return;

  let max = 0;
  let argmax = 0;
  for (let i = 0; i < weights.length; i++) {
    if (weights[i] > max) {
      max = weights[i];
      argmax = i;
    }
  }
  if (max <= 0) return;

  const cell = size / GRID_CELLS;
  for (let row = 0; row < GRID_CELLS; row++) {
    for (let col = 0; col < GRID_CELLS; col++) {
      const norm = weights[row * GRID_CELLS + col] / max;
      const alpha = Math.pow(norm, HEAT_GAMMA) * HEAT_MAX_ALPHA;
      if (alpha <= 0) continue;
      ctx.fillStyle = `rgba(${HEAT_RGB}, ${alpha})`;
      ctx.fillRect(col * cell, row * cell, cell, cell);
    }
  }

  const argRow = Math.floor(argmax / GRID_CELLS);
  const argCol = argmax % GRID_CELLS;
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2;
  ctx.strokeRect(argCol * cell + 1, argRow * cell + 1, cell - 2, cell - 2);
}
