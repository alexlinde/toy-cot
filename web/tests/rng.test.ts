import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import { PCG_INITSEQ, SceneRandom } from "../src/lib/rng";

interface RngCase {
  seed: number;
  next_u32: number[];
  randbelow: number[];
  randint: number[];
  weighted_17_3: number[];
}

interface RngFixture {
  initseq: number;
  cases: RngCase[];
}

const fixture: RngFixture = JSON.parse(
  readFileSync(new URL("../fixtures/rng.json", import.meta.url), "utf8"),
);

// Mirrors the (n) / (a, b) sequences export_web.py:rng_fixtures draws with.
const BELOW_NS = [2, 6, 13, 64, 1000];
const RANDINT_BOUNDS: [number, number][] = [
  [1, 12],
  [8, 12],
  [16, 22],
  [28, 35],
  [0, 63],
];

describe("SceneRandom (PCG32) golden fixtures", () => {
  it("uses the contract's stream constant", () => {
    expect(PCG_INITSEQ).toBe(fixture.initseq);
  });

  it("has cases to check", () => {
    expect(fixture.cases.length).toBeGreaterThan(0);
  });

  for (const c of fixture.cases) {
    // Every fixture array was produced from a FRESH SceneRandom(seed).
    describe(`seed ${c.seed}`, () => {
      it("next_u32", () => {
        const r = new SceneRandom(c.seed);
        const got = c.next_u32.map(() => r.nextU32());
        expect(got).toEqual(c.next_u32);
      });

      it("randbelow", () => {
        const r = new SceneRandom(c.seed);
        expect(BELOW_NS.map((n) => r.randbelow(n))).toEqual(c.randbelow);
      });

      it("randint", () => {
        const r = new SceneRandom(c.seed);
        expect(RANDINT_BOUNDS.map(([a, b]) => r.randint(a, b))).toEqual(
          c.randint,
        );
      });

      it("weightedChoice([0, 1], [17, 3])", () => {
        const r = new SceneRandom(c.seed);
        const got = c.weighted_17_3.map(() => r.weightedChoice([0, 1], [17, 3]));
        expect(got).toEqual(c.weighted_17_3);
      });
    });
  }
});

describe("derived draws", () => {
  it("nextU32 stays in uint32 range", () => {
    const r = new SceneRandom(12345);
    for (let i = 0; i < 512; i++) {
      const u = r.nextU32();
      expect(Number.isInteger(u)).toBe(true);
      expect(u).toBeGreaterThanOrEqual(0);
      expect(u).toBeLessThanOrEqual(0xffffffff);
    }
  });

  it("randint respects inclusive bounds", () => {
    const r = new SceneRandom(999);
    for (let i = 0; i < 512; i++) {
      const v = r.randint(1, 12);
      expect(v).toBeGreaterThanOrEqual(1);
      expect(v).toBeLessThanOrEqual(12);
    }
  });

  it("choice indexes through randbelow(len)", () => {
    const seq = ["a", "b", "c", "d"] as const;
    const a = new SceneRandom(31337);
    const b = new SceneRandom(31337);
    for (let i = 0; i < 64; i++) {
      expect(a.choice(seq)).toBe(seq[b.randbelow(seq.length)]);
    }
  });
});
