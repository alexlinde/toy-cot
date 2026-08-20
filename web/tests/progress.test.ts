import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { resolveTotalBytes } from "../src/lib/progress";
import type { ModelManifest } from "../src/lib/protocol";

const FILES = ["vocab.json", "step.onnx"] as const;
const NO_HEADERS = [NaN, NaN]; // Vercel: brotli-compressed, no Content-Length

function manifestWith(files: ModelManifest["files"]): ModelManifest {
  return { files } as ModelManifest;
}

describe("resolveTotalBytes", () => {
  it("uses manifest byte sizes when headers are absent (the prod regression)", () => {
    const m = manifestWith({
      "vocab.json": { sha256: "a", bytes: 1375 },
      "step.onnx": { sha256: "b", bytes: 20714340 },
    });
    expect(resolveTotalBytes(m, FILES, NO_HEADERS)).toBe(1375 + 20714340);
  });

  it("counts only the requested files, not everything the manifest lists", () => {
    const m = manifestWith({
      "vocab.json": { sha256: "a", bytes: 1375 },
      "step.onnx": { sha256: "b", bytes: 20714340 },
      "extra.bin": { sha256: "c", bytes: 1104924 },
    });
    expect(resolveTotalBytes(m, FILES, NO_HEADERS)).toBe(1375 + 20714340);
  });

  it("falls back to Content-Length when the manifest predates files[].bytes", () => {
    // old shape: bare sha256 strings
    const m = manifestWith({
      "vocab.json": "a",
      "step.onnx": "b",
    } as unknown as ModelManifest["files"]);
    expect(resolveTotalBytes(m, FILES, [1375, 20714340])).toBe(1375 + 20714340);
  });

  it("returns 0 (indeterminate) when neither source covers every file", () => {
    const m = manifestWith({
      "vocab.json": { sha256: "a", bytes: 1375 },
    } as unknown as ModelManifest["files"]);
    expect(resolveTotalBytes(m, FILES, [1375, NaN])).toBe(0);
  });

  it("agrees with the really exported manifest for the files the loader fetches", () => {
    const manifest = JSON.parse(
      readFileSync(join(__dirname, "..", "public", "model", "manifest.json"), "utf8"),
    ) as ModelManifest;
    const total = resolveTotalBytes(manifest, FILES, NO_HEADERS);
    // Must be positive without any header help, and match the actual files.
    expect(total).toBeGreaterThan(1_000_000);
    for (const f of FILES) {
      expect(manifest.files[f].bytes).toBeGreaterThan(0);
    }
  });
});
