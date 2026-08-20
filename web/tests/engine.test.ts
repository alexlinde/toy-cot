/**
 * Golden-transcript parity: the TS decode loop against transcripts produced by
 * the real Python `generate_response` on CPU.
 *
 * The graphs run under onnxruntime-node here and onnxruntime-web in the
 * browser; the engine only ever sees the structural Session interface, so the
 * adapter below is the whole difference between the two.
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import * as ort from "onnxruntime-node";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

import { NUM_HEADS, NUM_IMG_TOKENS, NUM_LAYERS } from "../src/lib/constants";
import { Engine, type EngineTensor, type Session, type TokenRecord } from "../src/lib/engine";
import type { ModelManifest } from "../src/lib/protocol";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const MODEL_DIR = join(ROOT, "public", "model");
const FIXTURE_DIR = join(ROOT, "fixtures");

const PROB_TOL = 2e-3;
const ATTN_TOL = 1e-3;

type TokenFixture = [string, number];
interface AttnSummary {
  mean: number[];
  argmax: number;
}
interface TranscriptCase {
  seed: number;
  question: string;
  rationale: string;
  answer: string;
  rationale_tokens: TokenFixture[];
  answer_tokens: TokenFixture[];
  answer_topk: TokenFixture[];
  rationale_attn_mean: AttnSummary[];
  answer_attn_mean: AttnSummary[];
  /** [token][layer][head][cell] for the first three rationale tokens. */
  rationale_attn_full_first3: number[][][][];
}

function readJson<T>(path: string): T {
  return JSON.parse(readFileSync(path, "utf8")) as T;
}

/** onnxruntime-node -> the engine's Session interface. */
function adapt(session: ort.InferenceSession): Session {
  return {
    async run(feeds: Record<string, EngineTensor>) {
      const ortFeeds: Record<string, ort.Tensor> = {};
      for (const [name, t] of Object.entries(feeds)) {
        ortFeeds[name] = new ort.Tensor(t.type, t.data as never, t.dims as number[]);
      }
      const out = await session.run(ortFeeds);
      const result: Record<string, { data: Float32Array }> = {};
      for (const [name, value] of Object.entries(out)) {
        result[name] = { data: value.data as Float32Array };
      }
      return result;
    },
  };
}

/** Mean over (layers, heads) of a flattened (L, H, 64) attention map. */
function attnMean(attn: Float32Array): Float64Array {
  const rows = NUM_LAYERS * NUM_HEADS;
  const mean = new Float64Array(NUM_IMG_TOKENS);
  for (let r = 0; r < rows; r++) {
    const base = r * NUM_IMG_TOKENS;
    for (let c = 0; c < NUM_IMG_TOKENS; c++) mean[c] += attn[base + c];
  }
  for (let c = 0; c < NUM_IMG_TOKENS; c++) mean[c] /= rows;
  return mean;
}

function argmaxOf(values: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let worst = 0;
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
  return worst;
}

const manifest = readJson<ModelManifest>(join(MODEL_DIR, "manifest.json"));
const vocab = readJson<Record<string, number>>(join(MODEL_DIR, "vocab.json"));
const transcripts = readJson<{ cases: TranscriptCase[] }>(
  join(FIXTURE_DIR, "transcripts.json"),
).cases;
const scenes = readJson<{ cases: { seed: number; image_rgb_base64: string }[] }>(
  join(FIXTURE_DIR, "scenes.json"),
).cases;

const sceneBySeed = new Map<number, Uint8Array>(
  scenes.map((s) => [s.seed, new Uint8Array(Buffer.from(s.image_rgb_base64, "base64"))]),
);

let engine: Engine;
const timings: { label: string; ms: number; steps: number }[] = [];

beforeAll(async () => {
  const step = await ort.InferenceSession.create(join(MODEL_DIR, "step.onnx"));
  engine = new Engine({ step: adapt(step), vocab, manifest });
}, 120_000);

afterAll(() => {
  if (timings.length === 0) return;
  const lines = timings.map(
    ({ label, ms, steps }) =>
      `  ${label}: ${ms.toFixed(0)} ms (${steps} steps, ${(ms / steps).toFixed(1)} ms/step)`,
  );
  const total = timings.reduce((a, t) => a + t.ms, 0);
  console.log(`decode timings (onnxruntime-node, CPU):\n${lines.join("\n")}
  total ${total.toFixed(0)} ms over ${timings.length} transcripts`);
});

describe("Engine construction", () => {
  it("rejects a manifest whose MAX_SEQ_LEN disagrees with the build", () => {
    const doctored: ModelManifest = {
      ...manifest,
      constants: { ...manifest.constants, MAX_SEQ_LEN: 512 },
    };
    const noop: Session = { run: async () => ({}) };
    expect(
      () => new Engine({ step: noop, vocab, manifest: doctored }),
    ).toThrow(/MAX_SEQ_LEN=512.*expects 256/);
  });

  it("rejects a manifest whose vocab_size disagrees with vocab.json", () => {
    const doctored: ModelManifest = {
      ...manifest,
      stats: { ...manifest.stats, vocab_size: manifest.stats.vocab_size + 1 },
    };
    const noop: Session = { run: async () => ({}) };
    expect(() => new Engine({ step: noop, vocab, manifest: doctored })).toThrow(
      /vocab_size/,
    );
  });
});

describe("golden transcripts", () => {
  for (const [index, tc] of transcripts.entries()) {
    const label = `#${index} seed ${tc.seed} "${tc.question}"`;

    it(
      label,
      async () => {
        const imageRGB = sceneBySeed.get(tc.seed);
        expect(imageRGB, `scenes.json has no seed ${tc.seed}`).toBeDefined();

        const streamed: TokenRecord[] = [];
        const streamedTopk: [string, number][][] = [];
        const started = performance.now();
        const result = await engine.generate(imageRGB!, tc.question, {
          onToken: (token) => streamed.push(token),
          onAnswerTopk: (topk) => streamedTopk.push(topk),
        });
        const elapsed = performance.now() - started;

        const steps = result.rationaleTokens.length + result.answerTokens.length;
        timings.push({ label, ms: elapsed, steps });

        // --- text -------------------------------------------------------
        expect(result.rationale).toBe(tc.rationale);
        expect(result.answer).toBe(tc.answer);

        // --- per-token words, probabilities, attention -------------------
        const stages: [string, TokenRecord[], TokenFixture[], AttnSummary[]][] = [
          ["rationale", result.rationaleTokens, tc.rationale_tokens, tc.rationale_attn_mean],
          ["answer", result.answerTokens, tc.answer_tokens, tc.answer_attn_mean],
        ];
        for (const [stage, got, want, attnWant] of stages) {
          expect(got.map((t) => t.word), `${stage} words`).toEqual(want.map(([w]) => w));
          got.forEach((token, i) => {
            expect(
              Math.abs(token.prob - want[i][1]),
              `${stage} token ${i} (${token.word}) prob ${token.prob} vs ${want[i][1]}`,
            ).toBeLessThanOrEqual(PROB_TOL);
          });

          expect(attnWant.length, `${stage} attention rows`).toBe(got.length);
          got.forEach((token, i) => {
            expect(token.attn.length).toBe(NUM_LAYERS * NUM_HEADS * NUM_IMG_TOKENS);
            const mean = attnMean(token.attn);
            expect(
              maxAbsDiff(mean, attnWant[i].mean),
              `${stage} token ${i} (${token.word}) attention mean`,
            ).toBeLessThanOrEqual(ATTN_TOL);
            expect(argmaxOf(mean), `${stage} token ${i} (${token.word}) attention argmax`).toBe(
              attnWant[i].argmax,
            );
          });
        }

        // Full (L, H, 64) maps for the first three rationale tokens.
        tc.rationale_attn_full_first3.forEach((want, i) => {
          const flat = new Float64Array(NUM_LAYERS * NUM_HEADS * NUM_IMG_TOKENS);
          let k = 0;
          for (const layer of want) for (const head of layer) for (const v of head) flat[k++] = v;
          expect(k, `full attention fixture ${i} length`).toBe(flat.length);
          expect(
            maxAbsDiff(result.rationaleTokens[i].attn, flat),
            `rationale token ${i} full attention map`,
          ).toBeLessThanOrEqual(ATTN_TOL);
        });

        // --- answer alternatives ----------------------------------------
        expect(result.answerTopk.map(([w]) => w), "topk words").toEqual(
          tc.answer_topk.map(([w]) => w),
        );
        result.answerTopk.forEach(([word, prob], i) => {
          expect(
            Math.abs(prob - tc.answer_topk[i][1]),
            `topk ${i} (${word}) prob ${prob} vs ${tc.answer_topk[i][1]}`,
          ).toBeLessThanOrEqual(PROB_TOL);
        });

        // --- streaming matches the returned transcript --------------------
        expect(streamed.map((t) => `${t.stage}:${t.word}`)).toEqual(
          [...result.rationaleTokens, ...result.answerTokens].map(
            (t) => `${t.stage}:${t.word}`,
          ),
        );
        expect(streamedTopk).toEqual([result.answerTopk]);
      },
      120_000,
    );
  }
});
