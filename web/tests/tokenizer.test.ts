import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import { Tokenizer } from "../src/lib/tokenizer";

const vocab: Record<string, number> = JSON.parse(
  readFileSync(new URL("../public/model/vocab.json", import.meta.url), "utf8"),
);

const tok = new Tokenizer(vocab);

describe("Tokenizer.preprocess", () => {
  it("lowercases and drops punctuation outside [a-z0-9. ]", () => {
    expect(Tokenizer.preprocess("Is there a RED circle?")).toBe(
      "is there a red circle",
    );
  });

  it("isolates '.' as its own token", () => {
    expect(Tokenizer.preprocess("row 2 col 3.")).toBe("row 2 col 3 .");
  });

  it("collapses runs of spaces and trims", () => {
    expect(Tokenizer.preprocess("  how   many  circles are there  ")).toBe(
      "how many circles are there",
    );
  });

  it("drops non-space whitespace outright, as text.py does", () => {
    // _preprocess_text strips [^a-z0-9. ] before splitting, so a tab or
    // newline glues its neighbours together rather than separating them.
    expect(Tokenizer.preprocess("many\tcircles")).toBe("manycircles");
  });

  it("returns '' for text with nothing tokenizable", () => {
    expect(Tokenizer.preprocess("!!!")).toBe("");
  });
});

describe("Tokenizer.tokenize", () => {
  it("maps each word through the vocab", () => {
    const words = ["how", "many", "circles", "are", "there"];
    for (const w of words) expect(vocab[w]).toBeDefined();
    expect(tok.tokenize("how many circles are there")).toEqual(
      words.map((w) => vocab[w]),
    );
  });

  it("emits '.' as the final token", () => {
    const ids = tok.tokenize("row 2 col 3.");
    expect(ids).toEqual(["row", "2", "col", "3", "."].map((w) => vocab[w]));
    expect(ids[ids.length - 1]).toBe(vocab["."]);
  });

  it("maps unknown words to <UNK>", () => {
    expect(vocab["zebra"]).toBeUndefined();
    expect(tok.tokenize("zebra")).toEqual([vocab["<UNK>"]]);
    expect(tok.unkId).toBe(vocab["<UNK>"]);
  });

  it("returns [] for empty preprocessed text", () => {
    expect(tok.tokenize("???")).toEqual([]);
  });

  it("exposes the vocab size", () => {
    expect(tok.size).toBe(Object.keys(vocab).length);
  });
});

describe("Tokenizer.decode", () => {
  it("skips special tokens (ids 0..12) and joins words with spaces", () => {
    const words = ["is", "there", "a", "red", "circle"];
    const specials = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    for (const id of specials) {
      expect(tok.decodeOne(id)).toBe("");
    }
    const ids = [1, 6, ...words.map((w) => vocab[w]), 7, 2];
    expect(tok.decode(ids)).toBe(words.join(" "));
  });

  it("keeps special tokens when asked", () => {
    expect(tok.decodeOne(1, false)).toBe("<BOS>");
    expect(tok.decode([1, vocab["red"], 2], false)).toBe("<BOS> red <EOS>");
  });

  it("round-trips a plain question", () => {
    expect(tok.decode(tok.tokenize("Is there a RED circle?"))).toBe(
      "is there a red circle",
    );
  });

  it("returns '' for ids outside the vocab", () => {
    expect(tok.decodeOne(10_000)).toBe("");
  });
});
