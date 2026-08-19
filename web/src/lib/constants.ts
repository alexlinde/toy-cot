/**
 * Model/sequence constants, mirrored from the Python side (text.py, model.py).
 * The engine sanity-checks these against public/model/manifest.json at load,
 * so a retrained/re-exported model that changed shape fails loudly.
 */

export const MAX_SEQ_LEN = 256;
export const NUM_IMG_TOKENS = 64;
export const IMG_POS_START = 2;
export const PREFIX_LEN = 2 + NUM_IMG_TOKENS + 1; // 67
export const NUM_LAYERS = 6;
export const NUM_HEADS = 4;
export const GRID_CELLS = 8;
export const IMAGE_SIZE = 64;
export const MAX_GEN_LEN = 160; // max steps per decode stage (rationale/answer)

/** Fixed special-token ids (SimpleTokenizer.SPECIAL_TOKENS; always 0..12). */
export const SPECIAL = {
  PAD: 0,
  BOS: 1,
  EOS: 2,
  UNK: 3,
  USER: 4,
  ASSISTANT: 5,
  THINK: 6,
  THINK_END: 7,
  FINAL: 8,
  FINAL_END: 9,
  IMG_START: 10,
  IMG_END: 11,
  IMG: 12,
} as const;

export const ALL_SPECIAL_IDS: ReadonlySet<number> = new Set(Object.values(SPECIAL));

/** A mix of canonical templates and trained alternative phrasings (GUI parity). */
export const EXAMPLE_QUESTIONS = [
  "is there a red circle",
  "how many circles are there",
  "is there a square above a circle",
  "is the number of squares greater than the number of circles",
  "are there any triangles on the left",
  "which shape is second from the left",
];
