/* Mars' ORIGINAL browser BPE tokenizer, matching mars_mini/tokenizer_bpe.py.
 * No pretrained vocabulary or remote tokenizer is required.
 * The learned merges are fetched from this Mars repository as JSON.
 */
export function createMarsBPE(payload) {
  if (!payload || payload.version !== "mars-bpe-v1" ||
      !Array.isArray(payload.merges) ||
      payload.vocab_size !== 256 + payload.merges.length ||
      payload.vocab_size > 4096) {
    throw new Error("Invalid Mars BPE vocabulary metadata.");
  }
  const vocab = Array.from({ length: 256 }, (_, id) => Uint8Array.of(id));
  const merges = [];
  for (const pair of payload.merges) {
    if (!Array.isArray(pair) || pair.length !== 2 ||
        pair.some(id => !Number.isInteger(id) || id < 0 || id >= vocab.length)) {
      throw new Error("Invalid Mars BPE merge pair.");
    }
    const [left, right] = pair;
    const a = vocab[left];
    const b = vocab[right];
    const joined = new Uint8Array(a.length + b.length);
    joined.set(a, 0);
    joined.set(b, a.length);
    vocab.push(joined);
    merges.push([left, right]);
  }

  const encoder = new TextEncoder();
  const decoder = new TextDecoder("utf-8", { fatal: false });

  function encode(value) {
    let tokens = Array.from(encoder.encode(value));
    for (let newId = 256; newId < vocab.length; newId++) {
      const [a, b] = merges[newId - 256];
      if (tokens.length < 2) break;
      const next = [];
      for (let i = 0; i < tokens.length;) {
        if (i + 1 < tokens.length && tokens[i] === a && tokens[i + 1] === b) {
          next.push(newId);
          i += 2;
        } else {
          next.push(tokens[i]);
          i++;
        }
      }
      tokens = next;
    }
    return tokens;
  }

  function decode(tokens) {
    const chunks = [];
    let length = 0;
    for (const id of tokens) {
      if (!Number.isInteger(id) || id < 0 || id >= vocab.length) {
        throw new Error("Invalid Mars token ID.");
      }
      const bytes = vocab[id];
      chunks.push(bytes);
      length += bytes.length;
    }
    const bytes = new Uint8Array(length);
    let index = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, index);
      index += chunk.length;
    }
    return decoder.decode(bytes);
  }

  return Object.freeze({
    vocabSize: vocab.length,
    encode,
    decode,
    tokenBytes: id => {
      if (!Number.isInteger(id) || id < 0 || id >= vocab.length) {
        throw new Error("Invalid Mars token ID.");
      }
      return vocab[id];
    }
  });
}
