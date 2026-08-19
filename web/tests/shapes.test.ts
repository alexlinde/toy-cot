import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import { SceneRandom } from "../src/lib/rng";
import {
  COLORS,
  IMAGE_SIZE,
  MAX_OBJECTS,
  MIN_OBJECTS,
  generateMultiShapeImage,
  gridCol,
  gridRow,
  makeSeededScene,
  type SceneObject,
} from "../src/lib/shapes";

interface SceneCase {
  seed: number;
  num_shapes_requested: number;
  metadata: SceneObject[];
  image_sha256: string;
  image_rgb_base64: string;
}

interface SceneFixture {
  image_size: number;
  cases: SceneCase[];
}

const fixture: SceneFixture = JSON.parse(
  readFileSync(new URL("../fixtures/scenes.json", import.meta.url), "utf8"),
);

describe("makeSeededScene golden fixtures", () => {
  it("agrees with the fixture canvas size", () => {
    expect(IMAGE_SIZE).toBe(fixture.image_size);
    expect(fixture.cases.length).toBeGreaterThan(0);
  });

  for (const c of fixture.cases) {
    describe(`seed ${c.seed}`, () => {
      const scene = makeSeededScene(c.seed);

      it("draws the same requested object count", () => {
        const rng = new SceneRandom(c.seed);
        expect(rng.randint(MIN_OBJECTS, MAX_OBJECTS)).toBe(
          c.num_shapes_requested,
        );
      });

      it("metadata matches", () => {
        expect(scene.metadata).toEqual(c.metadata);
        // Key set (and its exact names, size_category included) is contract.
        for (let i = 0; i < scene.metadata.length; i++) {
          expect(Object.keys(scene.metadata[i]).sort()).toEqual(
            Object.keys(c.metadata[i]).sort(),
          );
        }
      });

      it("image bytes match", () => {
        const expected = Buffer.from(c.image_rgb_base64, "base64");
        expect(expected.length).toBe(IMAGE_SIZE * IMAGE_SIZE * 3);
        const got = Buffer.from(
          scene.image.buffer,
          scene.image.byteOffset,
          scene.image.byteLength,
        );
        expect(got.length).toBe(expected.length);
        expect(Buffer.compare(got, expected)).toBe(0);
      });

      it("image sha256 matches", () => {
        const got = createHash("sha256").update(scene.image).digest("hex");
        expect(got).toBe(c.image_sha256);
      });
    });
  }
});

describe("scene generation invariants", () => {
  it("is deterministic for a given seed", () => {
    const a = makeSeededScene(4711);
    const b = makeSeededScene(4711);
    expect(a.metadata).toEqual(b.metadata);
    expect(Buffer.compare(Buffer.from(a.image), Buffer.from(b.image))).toBe(0);
  });

  it("shares one rng stream with the object-count draw", () => {
    const rng = new SceneRandom(12345);
    const num = rng.randint(MIN_OBJECTS, MAX_OBJECTS);
    const direct = generateMultiShapeImage(num, rng);
    const viaSeed = makeSeededScene(12345);
    expect(direct.metadata).toEqual(viaSeed.metadata);
    expect(
      Buffer.compare(Buffer.from(direct.image), Buffer.from(viaSeed.image)),
    ).toBe(0);
  });

  it("only paints contract colors on a black background", () => {
    const palette = new Set(
      Object.values(COLORS).map((rgb) => rgb.join(",")),
    );
    palette.add("0,0,0");
    const { image } = makeSeededScene(31337);
    for (let i = 0; i < image.length; i += 3) {
      expect(palette.has(`${image[i]},${image[i + 1]},${image[i + 2]}`)).toBe(
        true,
      );
    }
  });

  it("quantizes centers to the 8x8 grid", () => {
    expect(gridRow(0)).toBe(0);
    expect(gridRow(7)).toBe(0);
    expect(gridRow(8)).toBe(1);
    expect(gridCol(63)).toBe(7);
    for (const c of fixture.cases) {
      for (const o of c.metadata) {
        expect(gridRow(o.cy)).toBeGreaterThanOrEqual(0);
        expect(gridRow(o.cy)).toBeLessThan(8);
        expect(gridCol(o.cx)).toBeGreaterThanOrEqual(0);
        expect(gridCol(o.cx)).toBeLessThan(8);
      }
    }
  });
});
