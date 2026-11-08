/* Verify the published Python-trained BPE tokenizer produces identical
 * token IDs in JavaScript before exposing it to the live chat interface. */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createMarsBPE } from "../mars-bpe-browser.mjs";

const [vocabPath, fixturesPath] = process.argv.slice(2);
if (!vocabPath || !fixturesPath) {
  throw new Error("Usage: node tests/check-bpe-parity.mjs <vocab.json> <fixtures.json>");
}
const vocab = JSON.parse(readFileSync(vocabPath, "utf-8"));
const data = JSON.parse(readFileSync(fixturesPath, "utf-8"));
const tokenizer = createMarsBPE(vocab);
for (const item of data) {
  assert.deepEqual(tokenizer.encode(item.text), item.tokens, "Encoding mismatch: " + item.text);
  assert.equal(tokenizer.decode(item.tokens), item.text, "Decoding mismatch: " + item.text);
}
console.log("Verified original Mars Python/JS BPE parity for", data.length, "strings");
