/* Mars-1 Mini: original, locally executed browser inference.
 *
 * The model's own trained weights are downloaded from our repository as ONNX.
 * ONNX Runtime Web is only a numerical inference engine, not an AI provider.
 * Prompts and generated text stay in this browser. No API key or server needed.
 *
 * The model is a tiny research demo and is NOT a capable or reliable chatbot.
 */
// Model weights are our own. Opt-in modes never replace ordinary Mars chat.
import { createMarsBPE } from "./mars-bpe-browser.mjs";
const MODEL_MODE = new URLSearchParams(window.location.search).get("model");
const EXPERIMENTAL_MARS2 = MODEL_MODE === "mars2";
const EXPERIMENTAL_BPE = MODEL_MODE === "bpe";
const MODEL_URL = new URL(
  EXPERIMENTAL_BPE
    ? "./models/mars-bpe-mini.onnx?v=original-bpe-chat-1"
    : EXPERIMENTAL_MARS2
      ? "./models/mars2-mini.onnx?v=balanced-research-1"
      : "./models/mars1-mini.onnx?v=dialogue-v2-weights-1",
  import.meta.url
).href;
const BPE_VOCAB_URL = new URL(
  "./models/mars-bpe-mini.tokenizer.json?v=original-bpe-chat-1",
  import.meta.url
).href;
const ORT_VERSION = "1.30.0";
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const ORT_MODULE = ORT_BASE + "ort.min.mjs";
const CONTEXT_SIZE = (EXPERIMENTAL_MARS2 || EXPERIMENTAL_BPE) ? 160 : 128;
let loadPromise = null;

export function loadMarsMini() {
  if (loadPromise) return loadPromise;
  loadPromise = (async () => {
    const ort = await import(ORT_MODULE);
    ort.env.wasm.wasmPaths = ORT_BASE;
    ort.env.wasm.numThreads = 1;

    // We do not contact an AI service. This is our own model file on GitHub Pages.
    const session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all"
    });
    if (!session.inputNames.includes("tokens") || !session.outputNames.includes("logits")) {
      throw new Error("The Mars model file has unexpected inputs or outputs.");
    }
    let bpe = null;
    if (EXPERIMENTAL_BPE) {
      const response = await fetch(BPE_VOCAB_URL, { cache: "no-store" });
      if (!response.ok) throw new Error("Mars BPE vocabulary could not load (HTTP " + response.status + ").");
      bpe = createMarsBPE(await response.json());
      if (bpe.vocabSize !== 448) throw new Error("Mars BPE model and tokenizer vocabulary don't match.");
    }
    const vocabSize = bpe ? bpe.vocabSize : 256;
    const readable = new TextDecoder("utf-8", { fatal: true });
    const allowedTokens = [];
    for (let id = 0; id < vocabSize; id++) {
      if (!bpe) {
        if (id === 10 || (id >= 32 && id < 127)) allowedTokens.push(id);
      } else {
        try {
          const text = readable.decode(bpe.tokenBytes(id));
          if (text && !/[\\u0000-\\u001f\\u007f]/.test(text)) allowedTokens.push(id);
        } catch {
          // Incomplete UTF-8 byte fragments are not useful alone in a reply.
        }
      }
    }
    return { ort, session, bpe, vocabSize, allowedTokens };
  })().catch(error => {
    loadPromise = null; // Permit retry after a temporary CDN/network error.
    throw error;
  });
  return loadPromise;
}

function chooseNextToken(logits, offset, allowedIds, temperature, topK) {
  // Sample only readable tokens. Never substitute hardcoded answer text.
  const indices = allowedIds.slice();
  indices.sort((a, b) => logits[offset + b] - logits[offset + a]);
  const candidates = indices.slice(0, topK);
  const peak = logits[offset + candidates[0]] / temperature;
  const weights = candidates.map(id => Math.exp(logits[offset + id] / temperature - peak));
  const sum = weights.reduce((acc, weight) => acc + weight, 0);
  let random = Math.random() * sum;
  for (let i = 0; i < candidates.length; i += 1) {
    random -= weights[i];
    if (random <= 0) return candidates[i];
  }
  return candidates[candidates.length - 1];
}

export async function generateMarsMiniReply(prompt, options = {}) {
  const { ort, session, bpe, vocabSize, allowedTokens } = await loadMarsMini();
  const maxNewBytes = Math.min(120, Math.max(1, options.maxNewBytes ?? 90));
  const onProgress = typeof options.onProgress === "function" ? options.onProgress : () => {};
  const originalPrefix = `User: ${String(prompt).trim().slice(0, 220)}\nMars:`;
  const tokens = bpe ? bpe.encode(originalPrefix) : Array.from(new TextEncoder().encode(originalPrefix));
  const generated = [];
  const decoder = new TextDecoder("utf-8", { fatal: false });

  for (let step = 0; step < maxNewBytes; step += 1) {
    const recent = tokens.slice(-CONTEXT_SIZE);
    const tensor = new ort.Tensor(
      "int64",
      BigInt64Array.from(recent, id => BigInt(id)),
      [1, recent.length]
    );
    const results = await session.run({ tokens: tensor });
    const logits = results.logits.data;
    if (results.logits.dims[2] !== vocabSize) {
      throw new Error("Mars model has the wrong vocabulary size.");
    }
    const offset = (recent.length - 1) * vocabSize;
    if (!Number.isFinite(logits[offset])) {
      throw new Error("Mars returned invalid model values.");
    }
    const next = chooseNextToken(
      logits, offset, allowedTokens, 0.45,
      EXPERIMENTAL_MARS2 || EXPERIMENTAL_BPE ? 1 : 5
    );
    if (next === 10) break;
    tokens.push(next);
    generated.push(next);

    const partial = bpe ? bpe.decode(generated) : decoder.decode(Uint8Array.from(generated));
    if (/\n(?:User|Conversation|A short dialogue|Mars chat example):/i.test(partial)) {
      break;
    }
    if ((step + 1) % 8 === 0) {
      onProgress(step + 1, maxNewBytes);
      // Let the browser update the UI during CPU-heavy inference.
      await new Promise(resolve => requestAnimationFrame(resolve));
    }
  }

  const raw = bpe ? bpe.decode(generated) : decoder.decode(Uint8Array.from(generated));
  return raw
    .split(/\n(?:User|Conversation|A short dialogue|Mars chat example):/i)[0]
    .trim() || "(Mars generated no readable text this time.)";
}
