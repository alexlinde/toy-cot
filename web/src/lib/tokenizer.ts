/**
 * Word-level tokenizer, a port of text.py SimpleTokenizer over the exported
 * vocab.json. Preprocessing must match _preprocess_text exactly: lowercase,
 * '.' isolated as its own token, every character outside [a-z0-9. ] dropped,
 * whitespace collapsed.
 */

export class Tokenizer {
  readonly vocab: Map<string, number>;
  readonly idToWord: Map<number, string>;
  readonly unkId: number;

  constructor(vocab: Record<string, number>) {
    this.vocab = new Map(Object.entries(vocab));
    this.idToWord = new Map(
      Object.entries(vocab).map(([w, id]): [number, string] => [id, w]),
    );
    const unk = this.vocab.get("<UNK>");
    if (unk === undefined) throw new Error("vocab.json is missing <UNK>");
    this.unkId = unk;
  }

  get size(): number {
    return this.vocab.size;
  }

  /** text.py _preprocess_text */
  static preprocess(text: string): string {
    let t = text.toLowerCase();
    t = t.replaceAll(".", " . ");
    t = t.replace(/[^a-z0-9. ]/g, "");
    return t.split(/\s+/).filter(Boolean).join(" ");
  }

  /** Token ids; unknown words become <UNK> (inference behavior, allow_unk=True). */
  tokenize(text: string): number[] {
    const pre = Tokenizer.preprocess(text);
    if (!pre) return [];
    return pre.split(" ").map((w) => this.vocab.get(w) ?? this.unkId);
  }

  /** Single-token word, '' for unknown ids or (by default) special tokens. */
  decodeOne(id: number, skipSpecial = true): string {
    const w = this.idToWord.get(id);
    if (w === undefined) return "";
    if (skipSpecial && w.startsWith("<") && w.endsWith(">")) return "";
    return w;
  }

  decode(ids: number[], skipSpecial = true): string {
    return ids
      .map((id) => this.decodeOne(id, skipSpecial))
      .filter((w) => w !== "")
      .join(" ");
  }
}
