"use client";

import type { ModelManifest } from "@/lib/protocol";

function formatNumber(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

export function ModelStatsCard({ manifest }: { manifest: ModelManifest }) {
  const { stats, checkpoint } = manifest;
  return (
    <div className="stats-card">
      <div className="stats-card__title">Model statistics</div>
      <div>Total parameters: {formatNumber(stats.total_params)}</div>
      <div>Vision encoder: {formatNumber(stats.vision_params)}</div>
      <div>Auxiliary heads: {formatNumber(stats.aux_params)}</div>
      <div>Transformer blocks: {formatNumber(stats.transformer_params)}</div>
      <div>Vocabulary size: {formatNumber(stats.vocab_size)}</div>
      <div>
        Architecture: {stats.hidden_dim}d &middot; {stats.num_layers} layers &middot;{" "}
        {stats.num_heads} heads
      </div>
      <div>Checkpoint: {checkpoint}</div>
    </div>
  );
}
