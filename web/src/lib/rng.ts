/**
 * PCG32 + integer draw helpers: TS half of the cross-language RNG contract
 * with rng.py (see that file's docstring for the full spec). Golden
 * fixtures: web/fixtures/rng.json (tests/rng.test.ts).
 *
 * The 64-bit state lives in a BigInt. Constants are built with BigInt(...)
 * rather than the `123n` literal syntax because tsconfig targets ES2017,
 * where TypeScript rejects BigInt literals (the operators are fine).
 */

const MASK64 = BigInt("0xffffffffffffffff");
const MASK32 = BigInt("0xffffffff");
const PCG_MULT = BigInt("6364136223846793005");

const B0 = BigInt(0);
const B1 = BigInt(1);
const SHIFT_18 = BigInt(18);
const SHIFT_27 = BigInt(27);
const SHIFT_32 = BigInt(32);
const SHIFT_59 = BigInt(59);

/** Fixed stream constant (pcg32_srandom initseq); pinned by the contract. */
export const PCG_INITSEQ = 54;

export class SceneRandom {
  private state: bigint;
  private readonly inc: bigint;

  constructor(seed: number) {
    // pcg32_srandom_r(initstate=seed, initseq=PCG_INITSEQ)
    this.state = B0;
    this.inc = ((BigInt(PCG_INITSEQ) << B1) | B1) & MASK64;
    this.nextU32();
    this.state = (this.state + (BigInt(seed) & MASK64)) & MASK64;
    this.nextU32();
  }

  /** pcg32_random_r: 64-bit LCG step, output via xorshift + rotate. */
  nextU32(): number {
    const old = this.state;
    this.state = (old * PCG_MULT + this.inc) & MASK64;
    const xorshifted = Number((((old >> SHIFT_18) ^ old) >> SHIFT_27) & MASK32);
    const rot = Number(old >> SHIFT_59);
    return ((xorshifted >>> rot) | (xorshifted << ((-rot) & 31))) >>> 0;
  }

  /** Uniform int in [0, n) via multiply-shift. */
  randbelow(n: number): number {
    return Number((BigInt(this.nextU32()) * BigInt(n)) >> SHIFT_32);
  }

  /** Uniform int in [a, b], both bounds inclusive. */
  randint(a: number, b: number): number {
    return a + this.randbelow(b - a + 1);
  }

  choice<T>(seq: readonly T[]): T {
    return seq[this.randbelow(seq.length)];
  }

  /** Pick from seq with relative INTEGER weights. */
  weightedChoice<T>(seq: readonly T[], weights: readonly number[]): T {
    let total = 0;
    for (const w of weights) total += w;
    const r = this.randbelow(total);
    let acc = 0;
    for (let i = 0; i < seq.length; i++) {
      acc += weights[i];
      if (r < acc) return seq[i];
    }
    return seq[seq.length - 1]; // unreachable when weights are positive
  }
}
