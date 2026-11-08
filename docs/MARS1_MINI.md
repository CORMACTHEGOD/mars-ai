# Mars-1 Mini — our first ORIGINAL language model

**Current status (November 8, 2026):** Our [second training run](https://github.com/CORMACTHEGOD/mars-ai/actions/runs/42494963739) completed **2,500 steps** on **122 supervised question-answer pairs**, holding out another **12 examples**. The updated model has **260,736 parameters** and is available as ONNX weights on GitHub Pages. A [real Chromium browser test](https://github.com/CORMACTHEGOD/mars-ai/actions/runs/42495489186) confirmed that sending `hello` to the website generated **"Hello! I am Mars, an experimental AI."** using our model.

**Crucial limitation — overfitting:** The latest training loss dropped to approximately **0.035**, but the held-out loss **rose to 4.64**, so Mars-1 Mini mainly memorized familiar training answers. Unknown questions are still very unreliable. This is **not** evidence of general-purpose intelligence.

**Earlier experiment:** Our [first 150-step run](https://github.com/CORMACTHEGOD/mars-ai/actions/runs/42490292435) used a 257,664-parameter model and about 37 KB of raw text for next-byte prediction. It learned character patterns but output mostly gibberish. The response-focused training method is implemented in `mars_mini/train_dialogue.py`.

This is **Option A: from scratch**. We designed a small decoder-only Transformer with a byte tokenizer and started with random parameters. It does not call ChatGPT, Claude, Hugging Face, or any pretrained model.

## What is actually original?

- Mars tokenizer: UTF-8 bytes (256 token IDs), so no pretrained tokenizer download.
- Mars model: token and position embeddings, custom causal multi-head attention, feed-forward network, residual layers, language-model head.
- Mars data: small **hand-authored and synthetically expanded** dialogues in `mars_mini/corpus.py`.
- Mars training: next-token prediction and gradient descent from random initialization.
- Mars inference: sample tokens using weights from our checkpoint.
- Mars API: a small, local-only Python HTTP endpoint (`/api/chat`).

We use **PyTorch as a numerical computation library**. That does not make the Mars model pretrained; it supplies tensor operations and gradient calculations.

## What can it do yet?

An honest answer: **not much**. The starter dataset is tiny and the network has only a few hundred thousand parameters. After a short training run, its output may be repetitive or unreadable. It cannot reliably answer new questions, browse, generate images, or compete with ChatGPT/Claude. This is a proof that our model can learn patterns from data.

## Train from the GitHub website (no terminal)

**Training runs automatically when original model code under `mars_mini/` changes**, and can also be started manually. The current default is **2,500 response-focused training steps**, using `mars_mini/train_dialogue.py`. Ordinary website changes do not retrain the model. To train another checkpoint manually:\n\n1. Open the repo's [Actions tab](https://github.com/CORMACTHEGOD/mars-ai/actions).
2. Select **Train original Mars-1 Mini (manual)** on the left.
3. Click **Run workflow** and choose **2,500** steps for the current experimental setup.
4. Open the new run to watch its training-loss and validation-loss logs.
5. After the run succeeds, open **Artifacts** and download `mars-1-mini-trained-weights`.
6. The ZIP includes `mars1-mini.pt` (original PyTorch weights), `mars1-mini.onnx` (browser-ready weights), and `sample.txt` (genuine generated text).

The training workflow runs on changes to the original model code or when **you click Run workflow**; it does not run for ordinary website updates. GitHub Actions has account usage limits; check those in GitHub before scheduling long runs.

**Note:** The workflow must be on the repository's default branch for GitHub to offer the manual button. It will appear after we merge the feature branch.

## Optional local training (for later)

Python and CPU-compatible PyTorch must be installed. You can optionally run:

```bash
python -m unittest discover -s tests -v
python -m mars_mini.train --steps 150
python -m mars_mini.generate --prompt "hello"
python -m mars_mini.serve
```

The local API listens on `127.0.0.1:8765` and returns JSON for `POST /api/chat` when the model has been trained. It is a **development-only API** with no accounts, authentication or production safeguards. Never expose it directly to the public Internet.

## Run our original Mars model directly on GitHub Pages

We can run the model **in the visitor's browser**, without buying a server or calling someone else's AI API.

1. GitHub Actions trains the same original Mars-1 Mini PyTorch model.
2. `python -m mars_mini.export_onnx` converts its trained weights into `models/mars1-mini.onnx`.
3. An ONNX Runtime CPU test verifies that the exported model's results match our PyTorch model at sequence lengths 1, 8, 32 and 96.
4. The workflow commits the public model weights to `models/`.
5. `mars-mini-browser.js` downloads the weights and runs them through ONNX Runtime Web in the browser.
6. The regular [Mars AI website](https://cormacthegod.github.io/mars-ai/) generates a genuine experimental answer. The prompt and output are handled locally on the device.

This is **not** Python running on GitHub Pages; the static website loads a portable ONNX model and runs numerical operations through WebAssembly.

The model remains extremely small and was only trained on a few examples, so its responses may be nonsense. Search, image generation, plugins and 3D are still separate future projects. The older optional Hugging Face/Netlify backend is unused by this GitHub Pages experiment.

## What we improve after this experiment

1. Get proof that training and saving weights work.
2. Evaluate real answers versus memorized examples.
3. Collect more high-quality, appropriately licensed training examples.
4. Improve model architecture, datasets, tokenizer and evaluation tests.
5. Move to better compute and larger models when justified.

A competitive general-purpose model is a much bigger research-and-compute project. We can learn and make meaningful progress without pretending a tiny first model has those capabilities.
