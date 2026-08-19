// Copy the onnxruntime-web WASM runtime into public/ort/ so the site serves
// it same-origin (no CDN, no COOP/COEP requirements at numThreads=1).
// Runs via predev/prebuild.
import { copyFileSync, mkdirSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const dist = join(here, "..", "node_modules", "onnxruntime-web", "dist");
const out = join(here, "..", "public", "ort");

mkdirSync(out, { recursive: true });
const wanted = readdirSync(dist).filter(
  (f) => f.startsWith("ort-wasm-simd-threaded") && (f.endsWith(".wasm") || f.endsWith(".mjs")),
);
if (wanted.length === 0) throw new Error(`no ort wasm artifacts found in ${dist}`);
for (const f of wanted) copyFileSync(join(dist, f), join(out, f));
console.log(`copied ${wanted.length} ORT runtime files -> public/ort/`);
