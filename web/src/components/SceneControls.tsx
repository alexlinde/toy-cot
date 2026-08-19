"use client";

import { useState } from "react";
import { drawSceneSeed, makeSeededScene } from "@/lib/shapes";
import { useStore } from "@/lib/store";

function writeSeedToUrl(seed: number): void {
  const url = new URL(window.location.href);
  url.search = "";
  url.searchParams.set("seed", String(seed));
  window.history.replaceState(null, "", url.toString());
}

/** Seed input + Load Seed + New Scene -- everything that swaps the scene on
 * the canvas (worker notification happens reactively in App.tsx, keyed off
 * store.scene). Sharing a scene is the URL itself: the seed is always
 * mirrored into ?seed=. */
export function SceneControls() {
  const scene = useStore((s) => s.scene);
  const setScene = useStore((s) => s.setScene);

  const [seedInput, setSeedInput] = useState("");
  const [error, setError] = useState<string | null>(null);

  // Mirror the current scene's seed into the field whenever it changes
  // (mount, New Scene, Load Seed) without clobbering it on every render --
  // the "adjust state during rendering" pattern, so no effect is needed.
  const [mirroredSeed, setMirroredSeed] = useState<number | null>(null);
  if (scene && scene.seed !== mirroredSeed) {
    setMirroredSeed(scene.seed);
    setSeedInput(String(scene.seed));
  }

  function applyScene(seed: number): void {
    const next = makeSeededScene(seed);
    setScene(next);
    writeSeedToUrl(next.seed);
    setError(null);
  }

  function handleLoadSeed(): void {
    const trimmed = seedInput.trim();
    if (!/^-?\d+$/.test(trimmed)) {
      setError(`'${trimmed}' isn't a valid seed - please enter an integer.`);
      return;
    }
    applyScene(parseInt(trimmed, 10));
  }

  return (
    <div className="scene-controls">
      <div className="seed-row">
        <input
          className="seed-input"
          aria-label="Scene seed"
          inputMode="numeric"
          value={seedInput}
          onChange={(e) => {
            setSeedInput(e.target.value);
            setError(null);
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter") handleLoadSeed();
          }}
        />
        <button type="button" className="btn" onClick={handleLoadSeed}>
          Load Seed
        </button>
        <button type="button" className="btn" onClick={() => applyScene(drawSceneSeed())}>
          New Scene
        </button>
        {error && <div className="field-error">{error}</div>}
      </div>
    </div>
  );
}
