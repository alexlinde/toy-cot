/**
 * Progress-denominator logic for the model download, kept ort-free so it is
 * unit-testable (tests/progress.test.ts) -- this is exactly where the "bar
 * never moves" bug lived: Vercel serves every bundle file brotli-compressed
 * and therefore WITHOUT Content-Length, so header-based totals come out
 * unknowable in production while working fine against a local next start.
 */

import type { ModelManifest } from "./protocol";

/**
 * Total expected bytes for `files`, preferring the manifest's on-disk sizes
 * (fetch streams yield decoded bytes, which match them exactly) and falling
 * back to Content-Length header values (NaN where absent). Returns 0 when
 * neither source covers every file -- the UI then shows an indeterminate bar.
 */
export function resolveTotalBytes(
  manifest: ModelManifest,
  files: readonly string[],
  headerSizes: readonly number[],
): number {
  const manifestSizes = files.map((f) => {
    const entry = manifest.files?.[f];
    // Tolerate the pre-`bytes` manifest shape (a bare sha256 string).
    return typeof entry === "object" && Number.isFinite(entry?.bytes) ? entry.bytes : NaN;
  });
  const sizes = manifestSizes.every(Number.isFinite) ? manifestSizes : headerSizes;
  return sizes.every(Number.isFinite) ? sizes.reduce((a, b) => a + b, 0) : 0;
}
