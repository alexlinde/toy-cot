/**
 * Browser inference engine: a faithful port of model.py `generate_response`
 * (temperature 0) and `read_aux_counts` over the exported ONNX graphs.
 *
 * The engine never imports an onnxruntime package. It talks to whatever ran
 * the graph through the tiny structural `Session` interface below, so the
 * browser can hand it onnxruntime-web sessions while tests hand it
 * onnxruntime-node ones (see loader.ts / tests/engine.test.ts for the adapter,
 * which is four lines either way).
 *
 * Decode semantics that MUST match Python exactly:
 * - the prompt is [BOS] <IMG_START> <IMG>x64 <IMG_END> <|user|> q <|assistant|> <THINK>;
 * - each step pads the ids to MAX_SEQ_LEN and reads the logits row at
 *   pos = min(len(ids) - 1, MAX_SEQ_LEN - 1);
 * - every special token except the one the stage is allowed to emit is banned
 *   (logit -inf) before the softmax, so the reported probability is the
 *   model's confidence over the masked distribution;
 * - greedy argmax with the FIRST maximum winning ties;
 * - stage 1 stops at </THINK> (or when the sequence gets within 4 of the
 *   window), stage 2 at </FINAL> (or within 1).
 */

import {
  ALL_SPECIAL_IDS,
  GRID_CELLS,
  IMAGE_SIZE,
  IMG_POS_START,
  MAX_GEN_LEN,
  MAX_SEQ_LEN,
  NUM_HEADS,
  NUM_IMG_TOKENS,
  NUM_LAYERS,
  PREFIX_LEN,
  SPECIAL,
} from "./constants";
import type { AuxReadout, AuxRow, ModelManifest, Stage } from "./protocol";
import { Tokenizer } from "./tokenizer";

/** A tensor handed to a session. Both ort packages accept these fields. */
export interface EngineTensor {
  readonly type: "float32" | "int64";
  readonly data: Float32Array | BigInt64Array;
  readonly dims: readonly number[];
}

/** All graph outputs here are float32, so this is all the engine reads back. */
export interface EngineOutput {
  readonly data: Float32Array;
}

/** Minimal view of an ONNX session (onnxruntime-web / -node, via an adapter). */
export interface Session {
  run(feeds: Record<string, EngineTensor>): Promise<Record<string, EngineOutput>>;
}

export interface EngineParts {
  step: Session;
  aux: Session;
  vocab: Record<string, number>;
  manifest: ModelManifest;
}

/** One emitted token: the word, the model's confidence, and where it looked. */
export interface TokenRecord {
  stage: Stage;
  word: string;
  prob: number;
  /** (NUM_LAYERS, NUM_HEADS, NUM_IMG_TOKENS) flattened, layer-major. */
  attn: Float32Array;
}

export interface GenerateHooks {
  onToken?: (token: TokenRecord) => void;
  onAnswerTopk?: (topk: [string, number][]) => void;
}

export interface GenerateResult {
  rationale: string;
  answer: string;
  /** The same tokens the hooks streamed, in order (details parity with Python). */
  rationaleTokens: TokenRecord[];
  answerTokens: TokenRecord[];
  answerTopk: [string, number][];
}

const ATTN_LEN = NUM_LAYERS * NUM_HEADS * NUM_IMG_TOKENS;
const PIXELS = IMAGE_SIZE * IMAGE_SIZE;

/**
 * (64,64,3) row-major RGB bytes -> planar CHW float32 in [0,1], the layout
 * export_web.image_to_tensor produces: data[c*4096 + y*64 + x].
 */
export function imageToCHW(rgb: Uint8Array): Float32Array {
  if (rgb.length !== PIXELS * 3) {
    throw new Error(`expected ${PIXELS * 3} RGB bytes (64x64x3), got ${rgb.length}`);
  }
  const out = new Float32Array(3 * PIXELS);
  for (let p = 0; p < PIXELS; p++) {
    const src = p * 3;
    out[p] = rgb[src] / 255;
    out[PIXELS + p] = rgb[src + 1] / 255;
    out[2 * PIXELS + p] = rgb[src + 2] / 255;
  }
  return out;
}

/** Softmax over a slice, in doubles; -Infinity entries fall out as exactly 0. */
function softmax(logits: ArrayLike<number>, offset = 0, length = logits.length): Float64Array {
  let max = -Infinity;
  for (let i = 0; i < length; i++) {
    const v = logits[offset + i];
    if (v > max) max = v;
  }
  const probs = new Float64Array(length);
  let sum = 0;
  for (let i = 0; i < length; i++) {
    const e = Math.exp(logits[offset + i] - max);
    probs[i] = e;
    sum += e;
  }
  for (let i = 0; i < length; i++) probs[i] /= sum;
  return probs;
}

/** argmax with the first maximum winning ties (torch.argmax on CPU). */
function argmax(values: ArrayLike<number>): number {
  let best = 0;
  let bestVal = values[0];
  for (let i = 1; i < values.length; i++) {
    if (values[i] > bestVal) {
      bestVal = values[i];
      best = i;
    }
  }
  return best;
}

export class Engine {
  readonly manifest: ModelManifest;
  readonly tokenizer: Tokenizer;
  private readonly stepSession: Session;
  private readonly auxSession: Session;

  constructor({ step, aux, vocab, manifest }: EngineParts) {
    assertManifest(manifest, vocab);
    this.stepSession = step;
    this.auxSession = aux;
    this.manifest = manifest;
    this.tokenizer = new Tokenizer(vocab);
  }

  /** Prompt ids for a question, before any generation. */
  promptIds(question: string): number[] {
    const ids: number[] = [SPECIAL.BOS, SPECIAL.IMG_START];
    for (let i = 0; i < NUM_IMG_TOKENS; i++) ids.push(SPECIAL.IMG);
    ids.push(SPECIAL.IMG_END, SPECIAL.USER);
    for (const id of this.tokenizer.tokenize(question)) ids.push(id);
    ids.push(SPECIAL.ASSISTANT, SPECIAL.THINK);
    return ids;
  }

  /**
   * One decode step: pad the ids, run the graph, ban every special token but
   * `allowedSpecial`, and return the greedy pick plus the masked distribution
   * and this query row's image attention (a private copy -- ort reuses its
   * output buffers between runs).
   */
  private async stepOnce(
    image: EngineTensor,
    ids: number[],
    allowedSpecial: number,
  ): Promise<{ next: number; probs: Float64Array; attn: Float32Array }> {
    const padded = new BigInt64Array(MAX_SEQ_LEN); // PAD is 0, so zeros pad
    const n = Math.min(ids.length, MAX_SEQ_LEN);
    for (let i = 0; i < n; i++) padded[i] = BigInt(ids[i]);
    const pos = Math.min(ids.length - 1, MAX_SEQ_LEN - 1);

    const out = await this.stepSession.run({
      image,
      ids: { type: "int64", data: padded, dims: [1, MAX_SEQ_LEN] },
      pos: { type: "int64", data: BigInt64Array.from([BigInt(pos)]), dims: [1] },
    });

    const rawLogits = out.logits_row?.data;
    const rawAttn = out.attn_row?.data;
    if (!rawLogits || !rawAttn) {
      throw new Error("step.onnx did not return logits_row and attn_row");
    }
    if (rawAttn.length !== ATTN_LEN) {
      throw new Error(`attn_row has ${rawAttn.length} floats, expected ${ATTN_LEN}`);
    }

    const logits = Float64Array.from(rawLogits);
    for (const id of ALL_SPECIAL_IDS) {
      if (id !== allowedSpecial) logits[id] = -Infinity;
    }
    return { next: argmax(logits), probs: softmax(logits), attn: rawAttn.slice() };
  }

  /**
   * Two-stage greedy decode. Tokens stream through `hooks.onToken` as they are
   * emitted; the answer stage's top alternatives arrive once, before the first
   * answer token, exactly where Python computes them.
   */
  async generate(
    imageRGB: Uint8Array,
    question: string,
    hooks: GenerateHooks = {},
  ): Promise<GenerateResult> {
    const image: EngineTensor = {
      type: "float32",
      data: imageToCHW(imageRGB),
      dims: [1, 3, IMAGE_SIZE, IMAGE_SIZE],
    };
    const ids = this.promptIds(question);
    const rationaleTokens: TokenRecord[] = [];
    const answerTokens: TokenRecord[] = [];
    let answerTopk: [string, number][] = [];

    const emit = (stage: Stage, next: number, probs: Float64Array, attn: Float32Array) => {
      const token: TokenRecord = {
        stage,
        word: this.tokenizer.decodeOne(next),
        prob: probs[next],
        attn,
      };
      (stage === "rationale" ? rationaleTokens : answerTokens).push(token);
      hooks.onToken?.(token);
    };

    // Stage 1: rationale until </THINK>, leaving room for </THINK> <FINAL> a </FINAL>.
    for (let i = 0; i < MAX_GEN_LEN; i++) {
      if (ids.length >= MAX_SEQ_LEN - 4) break;
      const { next, probs, attn } = await this.stepOnce(image, ids, SPECIAL.THINK_END);
      ids.push(next);
      if (next === SPECIAL.THINK_END) break;
      emit("rationale", next, probs, attn);
    }
    if (ids[ids.length - 1] !== SPECIAL.THINK_END) ids.push(SPECIAL.THINK_END);
    ids.push(SPECIAL.FINAL);

    // Stage 2: answer until </FINAL>.
    for (let i = 0; i < MAX_GEN_LEN; i++) {
      if (ids.length >= MAX_SEQ_LEN - 1) break;
      const { next, probs, attn } = await this.stepOnce(image, ids, SPECIAL.FINAL_END);
      if (i === 0) {
        answerTopk = this.topAlternatives(probs);
        hooks.onAnswerTopk?.(answerTopk);
      }
      ids.push(next);
      if (next === SPECIAL.FINAL_END) break;
      emit("answer", next, probs, attn);
    }
    if (ids[ids.length - 1] !== SPECIAL.FINAL_END) ids.push(SPECIAL.FINAL_END);

    return {
      rationale: joinWords(rationaleTokens),
      answer: joinWords(answerTokens),
      rationaleTokens,
      answerTokens,
      answerTopk,
    };
  }

  /**
   * The auxiliary count heads' readout of the image alone -- what the encoder
   * believes before any reasoning happens (model.py read_aux_counts).
   */
  async auxRead(imageRGB: Uint8Array): Promise<AuxReadout> {
    const out = await this.auxSession.run({
      image: {
        type: "float32",
        data: imageToCHW(imageRGB),
        dims: [1, 3, IMAGE_SIZE, IMAGE_SIZE],
      },
    });
    const heads = this.manifest.aux_heads;
    return {
      shape: readCountHead(out.shape_logits?.data, heads.shape, heads.num_classes, "shape"),
      size: readCountHead(out.size_logits?.data, heads.size, heads.num_classes, "size"),
      color: readCountHead(out.color_logits?.data, heads.color, heads.num_classes, "color"),
    };
  }

  /** torch.topk(probs, 4), specials dropped, first 3 kept. */
  private topAlternatives(probs: Float64Array): [string, number][] {
    const order = Array.from(probs.keys());
    // Descending by probability; ties resolved by the lower index, as torch does.
    order.sort((a, b) => probs[b] - probs[a] || a - b);
    const top: [string, number][] = [];
    for (const id of order.slice(0, Math.min(4, probs.length))) {
      if (ALL_SPECIAL_IDS.has(id)) continue;
      top.push([this.tokenizer.decodeOne(id), probs[id]]);
      if (top.length === 3) break;
    }
    return top;
  }
}

/** Python's tok.decode(...): the empty words special/unknown ids map to drop out. */
function joinWords(tokens: TokenRecord[]): string {
  return tokens
    .map((t) => t.word)
    .filter((w) => w !== "")
    .join(" ");
}

function readCountHead(
  data: Float32Array | undefined,
  names: string[],
  numClasses: number,
  family: string,
): AuxRow[] {
  if (!data) throw new Error(`aux.onnx did not return ${family}_logits`);
  if (data.length !== names.length * numClasses) {
    throw new Error(
      `${family}_logits has ${data.length} floats, expected ${names.length * numClasses}`,
    );
  }
  return names.map((name, row) => {
    const probs = softmax(data, row * numClasses, numClasses);
    const count = argmax(probs);
    return [name, count, probs[count]] as AuxRow;
  });
}

/**
 * Fail loudly when the bundle in public/model/ was exported from a model whose
 * shape no longer matches what this build hardcodes.
 */
function assertManifest(manifest: ModelManifest, vocab: Record<string, number>): void {
  if (!manifest?.constants || !manifest.stats || !manifest.aux_heads) {
    throw new Error("manifest.json is missing constants/stats/aux_heads");
  }
  const expected: Record<string, number> = {
    MAX_SEQ_LEN,
    NUM_IMG_TOKENS,
    IMG_POS_START,
    PREFIX_LEN,
    GRID_CELLS,
    MAX_GEN_LEN,
  };
  for (const [name, want] of Object.entries(expected)) {
    const got = manifest.constants[name];
    if (got !== want) {
      throw new Error(
        `model bundle mismatch: manifest constant ${name}=${got}, this build expects ${want}`,
      );
    }
  }
  const stats: [string, number, number][] = [
    ["num_layers", manifest.stats.num_layers, NUM_LAYERS],
    ["num_heads", manifest.stats.num_heads, NUM_HEADS],
  ];
  for (const [name, got, want] of stats) {
    if (got !== want) {
      throw new Error(
        `model bundle mismatch: manifest stats.${name}=${got}, this build expects ${want}`,
      );
    }
  }
  const vocabSize = Object.keys(vocab).length;
  if (manifest.stats.vocab_size !== vocabSize) {
    throw new Error(
      `model bundle mismatch: manifest stats.vocab_size=${manifest.stats.vocab_size}, ` +
        `vocab.json has ${vocabSize} entries`,
    );
  }
  for (const [name, id] of Object.entries(SPECIAL_TOKEN_NAMES)) {
    const got = manifest.special_tokens?.[name];
    if (got !== id) {
      throw new Error(
        `model bundle mismatch: manifest special token ${name}=${got}, this build expects ${id}`,
      );
    }
  }
}

/** manifest.special_tokens keys -> the ids constants.ts hardcodes. */
const SPECIAL_TOKEN_NAMES: Record<string, number> = {
  "<PAD>": SPECIAL.PAD,
  "<BOS>": SPECIAL.BOS,
  "<EOS>": SPECIAL.EOS,
  "<UNK>": SPECIAL.UNK,
  "<|user|>": SPECIAL.USER,
  "<|assistant|>": SPECIAL.ASSISTANT,
  "<THINK>": SPECIAL.THINK,
  "</THINK>": SPECIAL.THINK_END,
  "<FINAL>": SPECIAL.FINAL,
  "</FINAL>": SPECIAL.FINAL_END,
  "<IMG_START>": SPECIAL.IMG_START,
  "<IMG_END>": SPECIAL.IMG_END,
  "<IMG>": SPECIAL.IMG,
};
