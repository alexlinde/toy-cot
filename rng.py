"""
Shared deterministic RNG for seeded scene generation.

This is the cross-language contract behind "a seed reproduces a scene": the
web UI (web/src/lib/rng.ts) implements the exact same algorithm, so a seed
renders the identical scene in the Python GUI and in the browser. Change
nothing here without changing it there and regenerating the golden fixtures
(export_web.py --fixtures).

The generator is PCG32 (pcg32_random_r / pcg32_srandom_r from the reference
implementation, O'Neill 2014, Apache-2.0/MIT), chosen over replicating
CPython's Mersenne Twister because the whole contract then fits in ~30
published lines per language instead of chasing stdlib internals.

The derived draws are part of the contract too, and deliberately avoid
floats: integer arithmetic is bit-exact across languages with zero care
required.

    next_u32()                   raw PCG32 output
    randbelow(n)   = (next_u32() * n) >> 32        (multiply-shift; bias
                     ~n/2^32, irrelevant at the n <= 64 used here)
    randint(a, b)  = a + randbelow(b - a + 1)      (inclusive bounds)
    choice(seq)    = seq[randbelow(len(seq))]
    weighted_choice(seq, int_weights):
                     r = randbelow(sum(w)); first index whose cumulative
                     weight exceeds r

Streams here are unrelated to the stdlib `random` streams the training
scripts seed; only the seeded-scene path (test_model.make_seeded_scene and
the web export fixtures) uses SceneRandom.
"""

MASK64 = (1 << 64) - 1
MASK32 = (1 << 32) - 1

_PCG_MULT = 6364136223846793005
# Fixed stream constant (pcg32_srandom initseq). Arbitrary but part of the
# cross-language contract: the TS implementation uses the same value.
PCG_INITSEQ = 54


class SceneRandom:
    """PCG32 with the integer draw helpers scene generation needs."""

    def __init__(self, seed: int):
        # pcg32_srandom_r(initstate=seed, initseq=PCG_INITSEQ)
        self.state = 0
        self.inc = ((PCG_INITSEQ << 1) | 1) & MASK64
        self.next_u32()
        self.state = (self.state + (seed & MASK64)) & MASK64
        self.next_u32()

    def next_u32(self) -> int:
        """pcg32_random_r: 64-bit LCG step, output via xorshift + rotate."""
        old = self.state
        self.state = (old * _PCG_MULT + self.inc) & MASK64
        xorshifted = (((old >> 18) ^ old) >> 27) & MASK32
        rot = old >> 59
        return ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & MASK32

    def randbelow(self, n: int) -> int:
        """Uniform int in [0, n) via multiply-shift."""
        return (self.next_u32() * n) >> 32

    def randint(self, a: int, b: int) -> int:
        """Uniform int in [a, b] (both bounds inclusive)."""
        return a + self.randbelow(b - a + 1)

    def choice(self, seq):
        return seq[self.randbelow(len(seq))]

    def weighted_choice(self, seq, weights):
        """Pick from seq with relative INTEGER weights (floats are banned
        from the contract)."""
        r = self.randbelow(sum(weights))
        acc = 0
        for item, w in zip(seq, weights):
            acc += w
            if r < acc:
                return item
        return seq[-1]  # unreachable when weights are positive
