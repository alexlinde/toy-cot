/**
 * Zustand store: the single source of UI state for the app. Everything here
 * is plain, serializable-ish data plus pure reducer-style actions -- side
 * effects (talking to the worker, touching the URL, clipboard, etc.) live in
 * components/hooks, not here.
 *
 * Response identity: a ChatResponse represents one full turn (the question
 * plus the model's streamed rationale/answer). Token selection/hover use a
 * flat index over the token's own response's tokens, concatenated as
 * [...rationaleTokens, ...answerTokens] -- the same order the Python GUI's
 * `tags` list ends up in (its rationale loop appends before its answer loop).
 */

import { create } from "zustand";
import type { ModelManifest, Stage } from "./protocol";
import type { Scene } from "./shapes";

/** Responses whose tokens stay clickable at once (GUI parity, see test_model.py). */
export const MAX_LIVE_RESPONSES = 8;

export type ModelStatus =
  | { status: "loading"; loadedBytes: number; totalBytes: number }
  | { status: "ready"; manifest: ModelManifest }
  | { status: "error"; message: string };

export interface ChatToken {
  word: string;
  prob: number;
  /** Null for a token with no captured map, or once its response is retired. */
  attn: Float32Array | null;
}

export type ResponseStatus = "decoding" | "done" | "error";

export interface ChatResponse {
  id: number;
  sceneSeed: number;
  question: string;
  rationaleTokens: ChatToken[];
  answerTokens: ChatToken[];
  topk: [string, number][];
  rationale: string;
  answer: string;
  status: ResponseStatus;
  /** True once retired: attention dropped, tokens grayed and inert. */
  stale: boolean;
  error?: string;
}

/** 'mean' averages all layers/heads; a number selects one layer (0..NUM_LAYERS-1). */
export type LayerChoice = "mean" | number;

/** A flat pointer into one response's concatenated token stream. */
export interface TokenRef {
  responseId: number;
  tokenIndex: number;
}

interface StoreState {
  modelStatus: ModelStatus;
  scene: Scene | null;
  responses: ChatResponse[];
  selection: TokenRef | null;
  hovered: TokenRef | null;
  layerChoice: LayerChoice;
  questionHistory: string[];

  setModelLoading: (loadedBytes: number, totalBytes: number) => void;
  setModelReady: (manifest: ModelManifest) => void;
  setModelError: (message: string) => void;

  /** Sets a new scene, retiring every live response (their maps belong to
   * the image now leaving the canvas). */
  setScene: (scene: Scene) => void;

  /** Registers a pending response for a just-sent 'ask', retiring the
   * oldest live response if this pushes the live count past the cap. */
  askQuestion: (id: number, sceneSeed: number, question: string) => void;
  appendToken: (
    id: number,
    stage: Stage,
    word: string,
    prob: number,
    attn: Float32Array,
  ) => void;
  setTopk: (id: number, topk: [string, number][]) => void;
  completeResponse: (
    id: number,
    sceneSeed: number,
    rationale: string,
    answer: string,
  ) => void;
  errorResponse: (id: number, message: string) => void;

  setLayerChoice: (choice: LayerChoice) => void;
  setHovered: (ref: TokenRef | null) => void;
  toggleSelection: (ref: TokenRef) => void;
  clearSelection: () => void;

  pushQuestionHistory: (question: string) => void;
}

function retireToken(t: ChatToken): ChatToken {
  return t.attn === null ? t : { ...t, attn: null };
}

function retireResponse(r: ChatResponse): ChatResponse {
  if (r.stale) return r;
  return {
    ...r,
    stale: true,
    rationaleTokens: r.rationaleTokens.map(retireToken),
    answerTokens: r.answerTokens.map(retireToken),
  };
}

export const useStore = create<StoreState>((set) => ({
  modelStatus: { status: "loading", loadedBytes: 0, totalBytes: 0 },
  scene: null,
  responses: [],
  selection: null,
  hovered: null,
  layerChoice: "mean",
  questionHistory: [],

  setModelLoading: (loadedBytes, totalBytes) =>
    set({ modelStatus: { status: "loading", loadedBytes, totalBytes } }),
  setModelReady: (manifest) => set({ modelStatus: { status: "ready", manifest } }),
  setModelError: (message) => set({ modelStatus: { status: "error", message } }),

  setScene: (scene) =>
    set((s) => ({
      scene,
      responses: s.responses.map(retireResponse),
      selection: null,
      hovered: null,
    })),

  askQuestion: (id, sceneSeed, question) =>
    set((s) => {
      const pending: ChatResponse = {
        id,
        sceneSeed,
        question,
        rationaleTokens: [],
        answerTokens: [],
        topk: [],
        rationale: "",
        answer: "",
        status: "decoding",
        stale: false,
      };
      let responses = [...s.responses, pending];
      const liveIndices: number[] = [];
      responses.forEach((r, i) => {
        if (!r.stale) liveIndices.push(i);
      });
      if (liveIndices.length > MAX_LIVE_RESPONSES) {
        const toRetire = new Set(
          liveIndices.slice(0, liveIndices.length - MAX_LIVE_RESPONSES),
        );
        responses = responses.map((r, i) => (toRetire.has(i) ? retireResponse(r) : r));
      }
      return { responses };
    }),

  appendToken: (id, stage, word, prob, attn) =>
    set((s) => ({
      responses: s.responses.map((r) => {
        if (r.id !== id) return r;
        const token: ChatToken = { word, prob, attn: r.stale ? null : attn };
        return stage === "rationale"
          ? { ...r, rationaleTokens: [...r.rationaleTokens, token] }
          : { ...r, answerTokens: [...r.answerTokens, token] };
      }),
    })),

  setTopk: (id, topk) =>
    set((s) => ({
      responses: s.responses.map((r) => (r.id === id ? { ...r, topk } : r)),
    })),

  completeResponse: (id, sceneSeed, rationale, answer) =>
    set((s) => ({
      responses: s.responses.map((r) => {
        if (r.id !== id) return r;
        const staleNow = r.stale || sceneSeed !== s.scene?.seed;
        const base = staleNow ? retireResponse(r) : r;
        return { ...base, status: "done", stale: staleNow, rationale, answer };
      }),
    })),

  errorResponse: (id, message) =>
    set((s) => ({
      responses: s.responses.map((r) =>
        r.id === id ? { ...r, status: "error", error: message } : r,
      ),
    })),

  setLayerChoice: (choice) => set({ layerChoice: choice }),
  setHovered: (ref) => set({ hovered: ref }),
  toggleSelection: (ref) =>
    set((s) => ({
      selection:
        s.selection &&
        s.selection.responseId === ref.responseId &&
        s.selection.tokenIndex === ref.tokenIndex
          ? null
          : ref,
    })),
  clearSelection: () => set({ selection: null, hovered: null }),

  pushQuestionHistory: (question) =>
    set((s) => ({ questionHistory: [...s.questionHistory, question] })),
}));

/** The token a TokenRef points to, or null if the response/index doesn't
 * (or no longer) exist -- e.g. a selection surviving past a retirement. */
export function resolveToken(
  responses: ChatResponse[],
  ref: TokenRef | null,
): ChatToken | null {
  if (!ref) return null;
  const r = responses.find((x) => x.id === ref.responseId);
  if (!r) return null;
  const rLen = r.rationaleTokens.length;
  const total = rLen + r.answerTokens.length;
  if (ref.tokenIndex < 0 || ref.tokenIndex >= total) return null;
  return ref.tokenIndex < rLen
    ? r.rationaleTokens[ref.tokenIndex]
    : r.answerTokens[ref.tokenIndex - rLen];
}
