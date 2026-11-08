# Mars-2 Research: Measuring More Than Memorization

Mars-2 Research is a **new, original experimental neural network** written in PyTorch. It does not use an outside AI model, a paid inference API, or scraped third-party content. Its weights begin randomly.

Our first Mars-1 Mini could repeat a few short answers but overfit its training examples: its held-out loss rose to **4.64** after 2,500 steps. That is not useful generalization.

## What's different

- **Larger original Transformer:** 3 layers, 128-dimensional token embeddings, four attention heads, 160-byte context (the exact parameter count appears in each GitHub training log).
- **More original practice questions:** multiple styles for the existing self-authored facts, plus new combinations of small arithmetic questions.
- **Three distinct groups:** training examples; validation prompts with *unseen wording for known facts*; separately held-out topics and addition pairs.
- **Early stopping:** saves the checkpoint with the lowest paraphrase-validation loss, not automatically the final training step.
- **Transparent reporting:** publishes `metrics.json` with the training and validation curve and `samples.json` containing real model-generated answers and correctness for each group.
- **Separate model file:** `models/mars2-mini.onnx`. Training Mars-2 does not replace the existing Mars-1 website.

## Important limits

Rephrasing a question about **a fact the model already saw** is a limited kind of generalization. It does **not** prove the model has learned unknown facts or can reliably reason.

A genuine held-out topic is never present in the training data, so the tiny model should not be expected to answer it correctly. Held-out addition operand combinations are also tested. We report failure honestly.

This is still much smaller than a useful general-purpose chatbot; no web search, image AI, browser access, or outside knowledge is provided to the model.

## Train and inspect

The model trains on a push to `main` that changes its own source, or via the manual **Train Mars-2 Research (original model)** GitHub Actions workflow.

1. Open [Mars AI GitHub Actions](https://github.com/CORMACTHEGOD/mars-ai/actions).
2. Choose **Train Mars-2 Research (original model)**.
3. Use the **Run workflow** menu if you want to retrain manually.
4. Open its completed run. Check `Train original Mars-2 with early stopping` and `Evaluate familiar, paraphrased and truly held-out questions`.
5. Download **mars2-research-weights-and-evaluation** from Artifacts to inspect `metrics.json`, `samples.json`, original PyTorch checkpoint, and exported ONNX weights.

Don't judge improvements only by training loss or three familiar example questions. Check its unfamiliar test prompts.

## Site deployment

Mars-1 continues serving the site until we have a verified reason to switch models. Mars-2 is a separate research checkpoint, which is intentional. The existing real browser-inference pipeline can later be adapted to Mars-2.

The manual training workflow consumes GitHub Actions compute allowance. It is not an unlimited free GPU. The model remains original even though PyTorch/ONNX provide tensor math.

## Code

- `mars_mini/mars2_data.py` — deterministic original examples and non-overlapping question splits
- `mars_mini/train_mars2.py` — train from random weights; early stop and save best checkpoint
- `mars_mini/evaluate_mars2.py` — separate, auditable generated-answer samples
- `tests/test_mars2.py` — regression tests for held-out splits, context limits and checkpoint saving
- `.github/workflows/train-mars2.yml` — reproducible GitHub Actions experiment
