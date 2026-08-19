"use client";

import { NUM_LAYERS } from "@/lib/constants";
import type { LayerChoice } from "@/lib/store";
import { useStore } from "@/lib/store";

export function AttentionControls() {
  const layerChoice = useStore((s) => s.layerChoice);
  const setLayerChoice = useStore((s) => s.setLayerChoice);
  const clearSelection = useStore((s) => s.clearSelection);

  return (
    <div className="controls-row">
      <select
        className="select"
        aria-label="Attention layer"
        value={layerChoice === "mean" ? "mean" : String(layerChoice)}
        onChange={(e) => {
          const v = e.target.value;
          setLayerChoice(v === "mean" ? "mean" : (Number(v) as LayerChoice));
        }}
      >
        <option value="mean">all layers (mean)</option>
        {Array.from({ length: NUM_LAYERS }, (_, i) => (
          <option key={i} value={i}>{`layer ${i}`}</option>
        ))}
      </select>
      <button type="button" className="btn" onClick={() => clearSelection()}>
        Clear
      </button>
    </div>
  );
}
