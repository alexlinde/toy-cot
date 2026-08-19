"use client";

import { useEffect, useRef } from "react";
import { computeAttentionWeights, drawAttentionOverlay, OVERLAY_CANVAS_SIZE } from "@/lib/overlay";
import { IMAGE_SIZE } from "@/lib/shapes";
import { resolveToken, useStore } from "@/lib/store";

/**
 * Two stacked canvases: a 64x64 backing-store base painted straight from the
 * scene's RGB bytes (CSS-scaled, pixelated), and a 512x512 overlay -- big
 * enough that its 2px argmax outline is a real 2 device pixels -- that paints
 * the active token's attention heat map, or nothing.
 */
export function SceneCanvas() {
  const baseRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);

  const scene = useStore((s) => s.scene);
  const responses = useStore((s) => s.responses);
  const hovered = useStore((s) => s.hovered);
  const selection = useStore((s) => s.selection);
  const layerChoice = useStore((s) => s.layerChoice);

  useEffect(() => {
    const canvas = baseRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx || !scene) return;
    const imageData = ctx.createImageData(IMAGE_SIZE, IMAGE_SIZE);
    const { image } = scene;
    for (let i = 0, p = 0; i < image.length; i += 3, p += 4) {
      imageData.data[p] = image[i];
      imageData.data[p + 1] = image[i + 1];
      imageData.data[p + 2] = image[i + 2];
      imageData.data[p + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
  }, [scene]);

  // Hovered preview wins over the pinned selection; nothing shows the bare scene.
  const activeToken = resolveToken(responses, hovered ?? selection);

  useEffect(() => {
    const canvas = overlayRef.current;
    const ctx = canvas?.getContext("2d");
    if (!ctx) return;
    const weights = activeToken?.attn
      ? computeAttentionWeights(activeToken.attn, layerChoice)
      : null;
    drawAttentionOverlay(ctx, weights);
  }, [activeToken, layerChoice]);

  return (
    <div className="scene-canvas-wrap">
      <canvas
        ref={baseRef}
        width={IMAGE_SIZE}
        height={IMAGE_SIZE}
        className="scene-canvas-base"
      />
      <canvas
        ref={overlayRef}
        width={OVERLAY_CANVAS_SIZE}
        height={OVERLAY_CANVAS_SIZE}
        className="scene-canvas-overlay"
      />
    </div>
  );
}
