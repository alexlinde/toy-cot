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
import { resolveTotalBytes } from "./progress";
import type { ModelManifest } from "./protocol";

export interface LoadProgress {
  loadedBytes: number;
  /** 0 when any response omitted Content-Length (progress is then unbounded). */
  totalBytes: number;
}

const PROGRESS_FILES = ["vocab.json", "step.onnx"] as const;

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
 * Fetch + instantiate the bundle from `${baseUrl}/model/`.
 *
 * The manifest is fetched FIRST and alone: it carries the on-disk byte size
 * of every other file, which is the only reliable progress denominator --
 * CDNs that compress a response (Vercel serves even step.onnx as brotli)
 * drop Content-Length, while fetch streams report DECODED bytes, which match
 * the manifest sizes exactly. Header-based totals remain as a fallback for a
 * manifest that predates the `files[].bytes` field.
 */
export async function load(
  baseUrl: string,
  onProgress?: (progress: LoadProgress) => void,
): Promise<Engine> {
  const base = baseUrl.replace(/\/+$/, "");

  const fetchOk = async (file: string): Promise<Response> => {
    const url = `${base}/model/${file}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`GET ${url} failed: ${res.status} ${res.statusText}`);
    return res;
  };

  const manifest = decodeJson<ModelManifest>(
    await (await fetchOk("manifest.json")).arrayBuffer(),
    "manifest.json",
  );

  const responses = await Promise.all(PROGRESS_FILES.map(fetchOk));

  const headerSizes = responses.map((res) => {
    const header = res.headers.get("content-length");
    return header === null ? NaN : Number(header);
  });
  const totalBytes = resolveTotalBytes(manifest, PROGRESS_FILES, headerSizes);
  onProgress?.({ loadedBytes: 0, totalBytes });

  let loadedBytes = 0;
  const buffers = await Promise.all(
    responses.map((res) =>
      readWithProgress(res, (chunk) => {
        loadedBytes += chunk;
        onProgress?.({ loadedBytes, totalBytes });
      }),
    ),
  );

  const [vocabBuf, stepBuf] = buffers;
  const vocab = decodeJson<Record<string, number>>(vocabBuf, "vocab.json");

  ort.env.wasm.wasmPaths = `${base}/ort/`;
  ort.env.wasm.numThreads = 1;
  const options: ort.InferenceSession.SessionOptions = { executionProviders: ["wasm"] };
  const step = await ort.InferenceSession.create(stepBuf, options);

  return new Engine({
    step: adaptSession(step),
    vocab,
    manifest,
  });
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
