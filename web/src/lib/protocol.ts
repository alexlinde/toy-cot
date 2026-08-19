/**
 * Message protocol between the UI thread and the inference Web Worker, plus
 * the shared result types. This file is the contract both sides build
 * against -- change it only with both src/worker/ and src/components/ in
 * hand.
 *
 * Conventions:
 * - Scenes are generated on the MAIN thread (pure TS, fast); the worker only
 *   ever sees the raw 64x64 RGB bytes (length 12288, row-major, R,G,B).
 * - The worker serializes work: an 'ask' received while another is decoding
 *   queues behind it (the Python GUI's inference lock, without the thread).
 * - Every ask is tagged with the sceneSeed it was asked against, and every
 *   response echoes it; the UI uses this to gray out (retire) responses whose
 *   scene has left the canvas by the time they arrive.
 * - Attention maps travel as Float32Array of length NUM_LAYERS * NUM_HEADS *
 *   NUM_IMG_TOKENS, layer-major then head then cell (cell = gridRow * 8 +
 *   gridCol, the vision encoder's raster order). Each (layer, head) row is a
 *   slice of a softmax over ALL keys, so it sums to the fraction of that
 *   head's attention spent on the image -- not to 1.
 */

/** [name, argmaxCount, probOfThatCount] per aux head. */
export type AuxRow = [string, number, number];
export interface AuxReadout {
  shape: AuxRow[];
  size: AuxRow[];
  color: AuxRow[];
}

export interface ModelManifest {
  checkpoint: string;
  checkpoint_sha256: string;
  stats: {
    total_params: number;
    vision_params: number;
    aux_params: number;
    transformer_params: number;
    hidden_dim: number;
    num_layers: number;
    num_heads: number;
    vocab_size: number;
  };
  constants: Record<string, number>;
  special_tokens: Record<string, number>;
  aux_heads: { shape: string[]; size: string[]; color: string[]; num_classes: number };
  files: Record<string, string>;
}

export type Stage = "rationale" | "answer";

export type WorkerRequest =
  | {
      /** Load ONNX sessions + vocab + manifest from `${baseUrl}/model/...`. */
      type: "init";
      baseUrl: string;
    }
  | {
      /** Decode one question against a scene. Responses echo `id`. */
      type: "ask";
      id: number;
      sceneSeed: number;
      imageRGB: Uint8Array;
      question: string;
    }
  | {
      /** Aux count-head readout for a scene (runs on scene change). */
      type: "aux";
      sceneSeed: number;
      imageRGB: Uint8Array;
    };

export type WorkerResponse =
  | { type: "init-progress"; loadedBytes: number; totalBytes: number }
  | { type: "ready"; manifest: ModelManifest }
  | { type: "init-error"; message: string }
  | { type: "aux-result"; sceneSeed: number; aux: AuxReadout }
  | {
      /** One emitted token, streamed as decoding proceeds. `prob` is the
       * model's untempered confidence in the token it emitted. `attn` is the
       * token's image-attention map (see header), never null in practice. */
      type: "token";
      id: number;
      stage: Stage;
      word: string;
      prob: number;
      attn: Float32Array;
    }
  | {
      /** Top answer alternatives at the first answer position (<= 3). */
      type: "answer-topk";
      id: number;
      topk: [string, number][];
    }
  | { type: "done"; id: number; sceneSeed: number; rationale: string; answer: string }
  | { type: "ask-error"; id: number; message: string };
