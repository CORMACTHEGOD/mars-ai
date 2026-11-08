/* Verifies the JavaScript BPE implementation matches our ranked-pair algorithm. */
import test from "node:test";
import assert from "node:assert/strict";
import { createMarsBPE } from "../mars-bpe-browser.mjs";

const sample = {
  version: "mars-bpe-v1",
  vocab_size: 260,
  merges: [[104, 101], [256, 108], [257, 108], [258, 111]]
};

test("merged token uses ranked original Mars BPE pairs", () => {
  const bpe = createMarsBPE(sample);
  assert.deepEqual(bpe.encode("hello"), [259]);
  assert.equal(bpe.decode([259]), "hello");
  assert.deepEqual(bpe.encode("hello hello"), [259, 32, 259]);
});

test("unseen text and Unicode still roundtrip", () => {
  const bpe = createMarsBPE(sample);
  for (const text of ["", "hello", "hello world", "How are you?", "🪐🚀", "café", "東京", "a\nb"]) {
    assert.equal(bpe.decode(bpe.encode(text)), text);
  }
});

test("reject invalid vocab sizes, invalid IDs and forward references", () => {
  assert.throws(() => createMarsBPE({ ...sample, vocab_size: 999 }));
  assert.throws(() => createMarsBPE({ ...sample, merges: [[300, 1]] }));
  const bpe = createMarsBPE(sample);
  assert.throws(() => bpe.decode([999]));
});
