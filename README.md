# Mars AI 🪐

**Imagine beyond.** Mars is an experimental AI platform for chat, research, coding, creative tools, plugins, and future 3D modeling.

## Current prototype

- Dark, responsive **single-chat** website with **YOUR IDEAS, UNLIMITED**
- Conversation history saved to the user's browser
- Original Mars-themed scroll indicators and an auto-growing message box
- **Mars-1 Mini**, an original, tiny Transformer neural network trained from random weights on original examples
- **In-browser inference**: the website downloads the model's public ONNX weights and runs them locally in your browser with ONNX Runtime Web. No paid AI API, secret key, separate backend account, or Python installation is required to try the experiment.

**Mars-1 Mini is still an experiment, not a general chatbot.** The November 8 dialogue-focused training run used 122 training dialogues, 12 held-out examples, 2,500 steps, and a 260,736-parameter model. It learned to reproduce some common answers such as `hello` and `what is Python?`, but dramatically overfit the small dataset (answer training loss 0.035 versus held-out loss 4.64). Expect unreliable answers to new questions. No paid inference provider is involved.

## Try our own model

When the ONNX model has been published by GitHub Actions, visit:

**https://cormacthegod.github.io/mars-ai/**

The page downloads the model and its numerical inference runtime, then generates text on your computer. It may need a few seconds to initialize. The original model itself is publicly downloadable as `models/mars1-mini.onnx`.

The original code and data are in `mars_mini/`. [See how training works](docs/MARS1_MINI.md).

## How publishing works

1. GitHub Actions trains the original PyTorch model.
2. `python -m mars_mini.export_onnx` converts the **same trained weights** to a browser-compatible ONNX file.
3. Python ONNX Runtime validates the exported model against the original PyTorch model at multiple context lengths.
4. The workflow publishes `models/mars1-mini.onnx` into the public repository.
5. GitHub Pages serves that file; ONNX Runtime Web handles inference **in the browser**. This is not a hosted Python model, and does not require API requests for each prompt.

Training is manual or triggered by changes to original model code. Ordinary frontend changes do not retrain Mars.

## Files

- `index.html`, `styles.css` — frontend and design
- `app.js` — chat history, statuses and UI
- `mars-mini-browser.js` — actual text generation using our public ONNX weights
- `models/mars1-mini.onnx` — published trained weights (generated automatically)
- `mars_mini/model.py`, `corpus.py`, `train_dialogue.py`, `train.py`, `export_onnx.py` — model implementation, focused answer training, and model export
- `docs/MARS1_MINI.md` — training experiment and limitations
- `docs/TOOL_ROUTING.md` — future intelligent tool-routing architecture

The earlier optional Hugging Face/Netlify beta code remains in `netlify/functions/` for reference, **but the GitHub Pages chat uses our own model and never calls that provider**. It also never asks visitors for an AI API token.

## Mars-2 Research — stronger testing, not just more memorization

We're now researching an **independent Mars-2** model from random weights. It has more capacity, more originally written and generated training examples, and a separate evaluation of unfamiliar question wording and held-out math problems. It uses **early stopping** to select the best checkpoint.

Read the [Mars-2 development and evaluation guide](docs/MARS2_RESEARCH.md). The trained research model is available as an **opt-in browser experiment** at [Mars-2 Lab](https://cormacthegod.github.io/mars-ai/?model=mars2). **The normal website stays on Mars-1.** Mars-2 reproduces some familiar example answers, but still fails most unfamiliar topics and arithmetic, so it is not a general-purpose upgrade.

## Mars Data V3 — fixing the memorization problem

We are preparing **new original training examples and more trustworthy evaluations** before spending more GitHub compute on another model. This is a separate data-quality project, **not** a new trained model or a change to the live website.

The [Mars Data V3 guide](docs/MARS_DATA_V3.md) explains our new practice sets, genuinely held-out word and addition inputs, data-leakage checks, and an exact-memorization baseline. GitHub Actions checks and exports the data **without training anything**.

## Mars original BPE tokenizer experiment

We've built a new [Mars BPE tokenizer](docs/MARS_BPE_TOKENIZER.md) from scratch. It learns common text fragments **only from original Mars Data V3 training examples** and can still encode unfamiliar text.

A separate CPU-only GitHub Actions experiment compares byte-token and BPE-token models using matched training settings and held-out **bits per raw byte**. Its new [BPE Lab chat preview](https://cormacthegod.github.io/mars-ai/?model=bpe) connects the matching trained model and vocabulary to website messages, without replacing the ordinary Mars-1 or Mars-2 site models. **It's still experimental and often gives wrong answers.** Read the [BPE chat integration guide](docs/MARS_BPE_CHAT.md) for details. No API key or external pretrained model is used.

## Future work

Improved training data and model quality, evaluations, general-use AI chat, search with citations, image understanding and generation, Mars Studio, plugins, and eventually 3D generation. Those future features are not implemented yet.

## Privacy and safety

Browser inference means prompts are not sent to a Mars inference server. ONNX Runtime Web is downloaded from a third-party code CDN, so standard CDN requests still occur. Avoid sharing sensitive data in public issues or project discussions. Before launching a public paid AI service, we would need proper moderation, abuse protection, accounts, reliability, and a much more capable model.
