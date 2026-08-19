/**
 * Singleton wrapper around the inference Web Worker. The worker itself lives
 * at src/worker/inference.worker.ts (owned by another agent); this module
 * only needs the message contract in protocol.ts to talk to it, so it
 * compiles fine whether or not that file exists yet.
 *
 * One worker per page load, created lazily on first use so import-time has
 * no side effects (safe to import during SSR).
 */

import type { WorkerRequest, WorkerResponse } from "./protocol";

type Listener = (msg: WorkerResponse) => void;

class WorkerClient {
  private worker: Worker | null = null;
  private listeners = new Set<Listener>();
  private nextAskId = 1;

  private ensureWorker(): Worker {
    if (!this.worker) {
      const worker = new Worker(
        new URL("../worker/inference.worker.ts", import.meta.url),
        { type: "module" },
      );
      worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
        for (const listener of this.listeners) listener(event.data);
      };
      this.worker = worker;
    }
    return this.worker;
  }

  /** Subscribe to every message the worker sends; returns an unsubscribe fn. */
  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  private post(msg: WorkerRequest): void {
    this.ensureWorker().postMessage(msg);
  }

  /** Load ONNX sessions + vocab + manifest from `${baseUrl}/model/...`. */
  init(baseUrl: string): void {
    this.post({ type: "init", baseUrl });
  }

  /** Queue a question against a scene; returns the id its responses echo.
   * The worker serializes asks itself, so it's safe to call this repeatedly
   * without waiting for the previous ask to finish. */
  ask(sceneSeed: number, imageRGB: Uint8Array, question: string): number {
    const id = this.nextAskId++;
    this.post({ type: "ask", id, sceneSeed, imageRGB, question });
    return id;
  }
}

export const workerClient = new WorkerClient();
