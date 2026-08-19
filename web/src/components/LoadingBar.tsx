"use client";

import { useStore } from "@/lib/store";

/** Replaces the question input entirely while the ONNX sessions download. */
export function LoadingBar() {
  const modelStatus = useStore((s) => s.modelStatus);
  if (modelStatus.status !== "loading") return null;

  const { loadedBytes, totalBytes } = modelStatus;
  const indeterminate = totalBytes <= 0;
  const pct = indeterminate ? 0 : Math.min(100, (loadedBytes / totalBytes) * 100);
  const mb = (n: number) => (n / 1e6).toFixed(1);

  return (
    <div className="loading-bar">
      <div className="loading-bar__track">
        <div
          className={
            "loading-bar__fill" + (indeterminate ? " loading-bar__fill--indeterminate" : "")
          }
          style={indeterminate ? undefined : { width: `${pct}%` }}
        />
      </div>
      <div className="loading-bar__label">
        Loading model... {mb(loadedBytes)} MB{totalBytes > 0 ? ` / ${mb(totalBytes)} MB` : ""}
      </div>
    </div>
  );
}
