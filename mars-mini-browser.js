/* Mars-1 Mini: original, locally executed browser inference.
 *
 * The model's own trained weights are downloaded from our repository as ONNX.
 * ONNX Runtime Web is only a numerical inference engine, not an AI provider.
 * Prompts and generated text stay in this browser. No API key or server needed.
 *
 * The model is a tiny research demo and is NOT a capable or reliable chatbot.
 */
const MODEL_URL = new URL("./models/mars1-mini.onnx?v=dialogue-v2", import.meta.url).href;
const ORT_VERSION = "1.30.0";
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const ORT_MODULE = ORT_BASE + "ort.min.mjs";
const CONTEXT_SIZE = 128;
const VOCAB_SIZE = 256;
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
      throw new Error("The Mars-1 Mini model file has unexpected inputs or outputs.");
    }
    return { ort, session };
  })().catch(error => {
    loadPromise = null; // Permit retry after a temporary CDN/network error.
    throw error;
  });
  return loadPromise;
}

function chooseNextByte(logits, offset, temperature, topK) {
  // Exclude control bytes that cannot reasonably be shown in a chat.
  const indices = [];
  for (let id = 0; id < VOCAB_SIZE; id += 1) {
    if (id === 10 || (id >= 32 && id < 127)) {
      indices.push(id);
    }
  }
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
  const { ort, session } = await loadMarsMini();
  const maxNewBytes = Math.min(120, Math.max(1, options.maxNewBytes ?? 90));
  const onProgress = typeof options.onProgress === "function" ? options.onProgress : () => {};
  const originalPrefix = `User: ${String(prompt).trim().slice(0, 220)}\nMars:`;
  const tokens = Array.from(new TextEncoder().encode(originalPrefix));
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
    const offset = (recent.length - 1) * VOCAB_SIZE;
    if (!Number.isFinite(logits[offset])) {
      throw new Error("Mars returned invalid model values.");
    }
    const next = chooseNextByte(logits, offset, 0.45, 5);\n    if (next === 10) break;
    tokens.push(next);
    generated.push(next);

    const partial = decoder.decode(Uint8Array.from(generated));
    if (/\n(?:User|Conversation|A short dialogue|Mars chat example):/i.test(partial)) {
      break;
    }
    if ((step + 1) % 8 === 0) {
      onProgress(step + 1, maxNewBytes);
      // Let the browser update the UI during CPU-heavy inference.
      await new Promise(resolve => requestAnimationFrame(resolve));
    }
  }

  const raw = decoder.decode(Uint8Array.from(generated));
  return raw
    .split(/\n(?:User|Conversation|A short dialogue|Mars chat example):/i)[0]
    .trim() || "(Mars-1 Mini generated no readable text this time.)";
}
