/**
 * Browser-side bundle loader: fetches public/model/* with byte progress,
 * creates the onnxruntime-web sessions, and hands the engine everything it
 * needs. This is the only file that knows onnxruntime-web exists (the worker
 * imports it, the engine does not).
 *
 * Single-threaded WASM on purpose: numThreads = 1 means no SharedArrayBuffer,
 * so the site needs no COOP/COEP cross-origin isolation headers. The runtime
 * is served same-origin from /ort/ (scripts/copy-ort.mjs puts it there).
 */

import * as ort from "onnxruntime-web";

import { Engine, type EngineTensor, type Session } from "./engine";
import type { ModelManifest } from "./protocol";

export interface LoadProgress {
  loadedBytes: number;
  /** 0 when any response omitted Content-Length (progress is then unbounded). */
  totalBytes: number;
}

// aux.onnx is deliberately NOT fetched up front: the UI no longer shows the
// encoder-counts panel, so its graph loads lazily on the first 'aux' request
// (if one ever comes) instead of costing every visitor ~1MB.
const MODEL_FILES = ["manifest.json", "vocab.json", "step.onnx"] as const;

/** Wrap an ort session so the engine never sees an ort type. */
export function adaptSession(session: ort.InferenceSession): Session {
  return {
    async run(feeds: Record<string, EngineTensor>) {
      const ortFeeds: Record<string, ort.Tensor> = {};
      for (const [name, t] of Object.entries(feeds)) {
        ortFeeds[name] = new ort.Tensor(
          t.type,
          t.data as never,
          t.dims as number[],
        );
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

/**
 * Fetch + instantiate the whole bundle from `${baseUrl}/model/`.
 * `onProgress` fires as stream chunks land, with the running total across all
 * four files.
 */
export async function load(
  baseUrl: string,
  onProgress?: (progress: LoadProgress) => void,
): Promise<Engine> {
  const base = baseUrl.replace(/\/+$/, "");

  const responses = await Promise.all(
    MODEL_FILES.map(async (file) => {
      const url = `${base}/model/${file}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(`GET ${url} failed: ${res.status} ${res.statusText}`);
      return res;
    }),
  );

  // Content-Length is advisory: if any response withholds it we cannot report a
  // denominator, so the UI gets totalBytes 0 and shows an indeterminate bar.
  const sizes = responses.map((res) => {
    const header = res.headers.get("content-length");
    return header === null ? NaN : Number(header);
  });
  const totalBytes = sizes.some((n) => !Number.isFinite(n))
    ? 0
    : sizes.reduce((a, b) => a + b, 0);

  let loadedBytes = 0;
  const buffers = await Promise.all(
    responses.map((res) =>
      readWithProgress(res, (chunk) => {
        loadedBytes += chunk;
        onProgress?.({ loadedBytes, totalBytes });
      }),
    ),
  );

  const [manifestBuf, vocabBuf, stepBuf] = buffers;
  const manifest = decodeJson<ModelManifest>(manifestBuf, "manifest.json");
  const vocab = decodeJson<Record<string, number>>(vocabBuf, "vocab.json");

  ort.env.wasm.wasmPaths = `${base}/ort/`;
  ort.env.wasm.numThreads = 1;
  const options: ort.InferenceSession.SessionOptions = { executionProviders: ["wasm"] };
  const step = await ort.InferenceSession.create(stepBuf, options);

  return new Engine({
    step: adaptSession(step),
    aux: lazySession(`${base}/model/aux.onnx`, options),
    vocab,
    manifest,
  });
}

/** A Session that fetches and instantiates its graph on first run(). */
function lazySession(url: string, options: ort.InferenceSession.SessionOptions): Session {
  let pending: Promise<Session> | null = null;
  return {
    async run(feeds) {
      pending ??= (async () => {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`GET ${url} failed: ${res.status} ${res.statusText}`);
        return adaptSession(await ort.InferenceSession.create(await res.arrayBuffer(), options));
      })();
      return (await pending).run(feeds);
    },
  };
}

/** Drain a response body, reporting each chunk's size. */
async function readWithProgress(
  res: Response,
  onChunk: (bytes: number) => void,
): Promise<ArrayBuffer> {
  if (!res.body) {
    const buf = await res.arrayBuffer();
    onChunk(buf.byteLength);
    return buf;
  }
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    size += value.byteLength;
    onChunk(value.byteLength);
  }
  const out = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return out.buffer;
}

function decodeJson<T>(buf: ArrayBuffer, what: string): T {
  try {
    return JSON.parse(new TextDecoder().decode(buf)) as T;
  } catch (err) {
    throw new Error(`${what} is not valid JSON: ${(err as Error).message}`);
  }
}
