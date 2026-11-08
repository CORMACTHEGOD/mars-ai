# Mars-1 Mini — our first ORIGINAL language model

**Status:** Original code is written. The training workflow is available but the initial model weights have not been trained or deployed yet.

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

1. Open the repo's [Actions tab](https://github.com/CORMACTHEGOD/mars-ai/actions).
2. Select **Train original Mars-1 Mini (manual)** on the left.
3. Click **Run workflow** and choose **150** steps first.
4. Open the new run to watch its training-loss and validation-loss logs.
5. After the run succeeds, open **Artifacts** and download `mars-1-mini-trained-weights`.
6. The ZIP includes `mars1-mini.pt` (the actual trained weights) and `sample.txt` (a genuine generated text sample).

The workflow only starts when **you click Run workflow**, never on every commit. GitHub Actions has account usage limits; check those in GitHub before scheduling long runs.

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

## When can Mars use our model on the website?

The public GitHub Pages frontend **cannot run PyTorch**. To use Mars-1 Mini online, we would need to:

1. Train and save a working checkpoint.
2. Host the Python model in a private CPU/GPU service.
3. Add secure authentication, user quotas, moderation, and a browser-facing chat endpoint.
4. Point the Mars website to that endpoint.

The previous Netlify/Hugging Face beta code is **separate**. It remains inactive unless you configure provider credentials. Mars-1 Mini does not use it, and this code does not silently replace it.

## What we improve after this experiment

1. Get proof that training and saving weights work.
2. Evaluate real answers versus memorized examples.
3. Collect more high-quality, appropriately licensed training examples.
4. Improve model architecture, datasets, tokenizer and evaluation tests.
5. Move to better compute and larger models when justified.

A competitive general-purpose model is a much bigger research-and-compute project. We can learn and make meaningful progress without pretending a tiny first model has those capabilities.
