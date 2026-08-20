/**
 * Inference Web Worker: owns the ONNX sessions and the decode loop so the UI
 * thread never blocks on WASM.
 *
 * Every request runs on one promise chain, which is the whole concurrency
 * story: an 'ask' that arrives mid-decode waits its turn (the Python GUI's
 * inference lock, without the thread). Tokens stream out as they are emitted,
 * each carrying its attention map -- transferred, not copied, so a 1536-float
 * buffer per token costs nothing.
 */

import { Engine } from "@/lib/engine";
import { load } from "@/lib/loader";
import type { WorkerRequest, WorkerResponse } from "@/lib/protocol";

/** The slice of DedicatedWorkerGlobalScope this file uses (lib.dom is loaded). */
interface WorkerScope {
  postMessage(message: WorkerResponse, transfer?: Transferable[]): void;
  onmessage: ((event: MessageEvent<WorkerRequest>) => void) | null;
}

const ctx = self as unknown as WorkerScope;

let engine: Engine | null = null;
/** FIFO: each request appends to the chain and waits for everything before it. */
let queue: Promise<void> = Promise.resolve();

function post(message: WorkerResponse, transfer?: Transferable[]): void {
  ctx.postMessage(message, transfer);
}

function message(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function enqueue(task: () => Promise<void>): void {
  queue = queue.then(task, task);
}

async function handleInit(req: Extract<WorkerRequest, { type: "init" }>): Promise<void> {
  try {
    engine = await load(req.baseUrl, ({ loadedBytes, totalBytes }) => {
      post({ type: "init-progress", loadedBytes, totalBytes });
    });
    post({ type: "ready", manifest: engine.manifest });
  } catch (err) {
    engine = null;
    post({ type: "init-error", message: message(err) });
  }
}

async function handleAsk(req: Extract<WorkerRequest, { type: "ask" }>): Promise<void> {
  if (!engine) {
    post({ type: "ask-error", id: req.id, message: "model is not loaded yet" });
    return;
  }
  try {
    const { rationale, answer } = await engine.generate(req.imageRGB, req.question, {
      onToken: ({ stage, word, prob, attn }) => {
        post({ type: "token", id: req.id, stage, word, prob, attn }, [attn.buffer]);
      },
      onAnswerTopk: (topk) => post({ type: "answer-topk", id: req.id, topk }),
    });
    post({ type: "done", id: req.id, sceneSeed: req.sceneSeed, rationale, answer });
  } catch (err) {
    post({ type: "ask-error", id: req.id, message: message(err) });
  }
}

ctx.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const req = event.data;
  switch (req.type) {
    case "init":
      enqueue(() => handleInit(req));
      break;
    case "ask":
      enqueue(() => handleAsk(req));
      break;
  }
};
