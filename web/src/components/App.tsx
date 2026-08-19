"use client";

import { useEffect, useRef } from "react";
import { drawSceneSeed, makeSeededScene } from "@/lib/shapes";
import { useStore } from "@/lib/store";
import { workerClient } from "@/lib/workerClient";
import type { WorkerResponse } from "@/lib/protocol";
import { DesktopLayout } from "./DesktopLayout";
import { MobileLayout } from "./MobileLayout";
import { useAskQuestion } from "./useAskQuestion";
import { useIsDesktop } from "./useIsDesktop";

/** Integer seeds only (matches the Python GUI's `int(text)` seed parsing). */
function parseSeedParam(raw: string | null): number | null {
  if (raw === null) return null;
  const trimmed = raw.trim();
  if (!/^-?\d+$/.test(trimmed)) return null;
  return parseInt(trimmed, 10);
}

/**
 * Top-level orchestrator: owns the one-time mount sequence (parse the URL,
 * draw the initial scene, wire up the worker) and the once-only ?q=
 * auto-ask. Everything else reads/writes the zustand store directly, so no
 * state is threaded through props here.
 */
export default function App() {
  const isDesktop = useIsDesktop();
  const modelStatus = useStore((s) => s.modelStatus);
  const setScene = useStore((s) => s.setScene);
  const setModelLoading = useStore((s) => s.setModelLoading);
  const setModelReady = useStore((s) => s.setModelReady);
  const setModelError = useStore((s) => s.setModelError);
  const appendToken = useStore((s) => s.appendToken);
  const setTopk = useStore((s) => s.setTopk);
  const completeResponse = useStore((s) => s.completeResponse);
  const errorResponse = useStore((s) => s.errorResponse);

  const ask = useAskQuestion();
  const pendingQRef = useRef<string | null>(null);
  const askedQRef = useRef(false);

  // Mount, once: read ?seed=/?q= from the URL, draw the first scene, and
  // subscribe to the worker (which is created lazily on first postMessage).
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const seedParam = parseSeedParam(params.get("seed"));
    const qParam = params.get("q");
    const initialScene = makeSeededScene(seedParam ?? drawSceneSeed());
    setScene(initialScene);
    pendingQRef.current = qParam;

    const url = new URL(window.location.href);
    url.search = "";
    url.searchParams.set("seed", String(initialScene.seed));
    if (qParam) url.searchParams.set("q", qParam);
    window.history.replaceState(null, "", url.toString());

    const unsubscribe = workerClient.subscribe((msg: WorkerResponse) => {
      switch (msg.type) {
        case "init-progress":
          setModelLoading(msg.loadedBytes, msg.totalBytes);
          break;
        case "ready":
          setModelReady(msg.manifest);
          break;
        case "init-error":
          setModelError(msg.message);
          break;
        case "token":
          appendToken(msg.id, msg.stage, msg.word, msg.prob, msg.attn);
          break;
        case "answer-topk":
          setTopk(msg.id, msg.topk);
          break;
        case "done":
          completeResponse(msg.id, msg.sceneSeed, msg.rationale, msg.answer);
          break;
        case "ask-error":
          errorResponse(msg.id, msg.message);
          break;
      }
    });

    workerClient.init(window.location.origin);

    return unsubscribe;
    // Runs once: URL parsing and the worker subscription are mount-only.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Auto-ask a ?q= question exactly once, as soon as the model is ready.
  useEffect(() => {
    if (askedQRef.current) return;
    if (modelStatus.status !== "ready") return;
    if (!pendingQRef.current) return;
    askedQRef.current = true;
    ask(pendingQRef.current);
  }, [modelStatus.status, ask]);

  return isDesktop ? <DesktopLayout /> : <MobileLayout />;
}
