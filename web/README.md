# Toy CoT VLM — web UI

Browser port of the Tkinter GUI (`../test_model.py`), live at
**https://toy-cot.vercel.app**. The whole model runs client-side:
onnxruntime-web (WASM, single-threaded) decodes the chain token by token in
a Web Worker, so the site is fully static — no inference server.

- `?seed=N` reproduces a scene; `&q=...` auto-asks a question. A seed
  renders the **byte-identical** scene here and in the Python GUI.
- Click (or hover, on desktop) any token of a chain to paint that token's
  attention over the 64 image cells; the layer selector averages heads per
  layer or across all layers.

## Cross-language contracts

Scene generation is deterministic across Python and TS by construction, not
by porting stdlib internals:

- `src/lib/rng.ts` ↔ `../rng.py` — PCG32 plus integer-only draw helpers.
- `src/lib/shapes.ts` ↔ `../shapes.py` — draw order and an explicit integer
  rasterizer (inclusive rects, `dx²+dy² ≤ r²+r` disks, scanline triangles).
- `src/lib/tokenizer.ts` ↔ `../text.py` — word-level tokenizer.
- `src/lib/engine.ts` ↔ `../model.py generate_response` — the greedy
  two-stage decode loop, confidence probs, answer top-k, attention capture.

`../export_web.py` regenerates everything the site consumes after a
retrain: `public/model/{step,aux}.onnx` + vocab + manifest (with a
torch↔onnxruntime parity gate), and the golden fixtures in `fixtures/`
that `npm test` verifies the TS side against — RNG vectors, 20 byte-exact
scenes, and 10 full decode transcripts with attention maps.

## Commands

```bash
npm run dev    # local dev (copies the ORT wasm runtime into public/ort first)
npm test       # golden-fixture suites: rng, shapes, tokenizer, engine
npm run build  # production build
npx vercel deploy --prod   # deploy (project: somefamilies/toy-cot)
```

Re-export after retraining (from the repo root):

```bash
uv run python export_web.py --checkpoint toy_vlm_cot.pth
cd web && npm test && npx vercel deploy --prod
```
