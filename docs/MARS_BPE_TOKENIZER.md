# Mars BPE tokenizer V1 — original language-model experiment

**Status:** Research code and model training workflow. This is not a new capable general chatbot and does not replace the existing Mars-1 or Mars-2 live models.

## What changed?

Earlier Mars models used a 256-token **byte tokenizer**. Each UTF-8 byte was one model token. Even common pieces of English could consume many tokens.

We've now built an **original byte-pair encoding (BPE)** tokenizer in Python that:

- Starts with all 256 possible raw bytes, retaining the ability to represent unfamiliar Unicode strings and emojis.
- Learns up to **192 repeated adjacent token pairs** from **Mars Data V3's training split only** (no validation/test data used to fit its vocabulary).
- Creates a deterministic, auditable vocabulary with at most 448 tokens.
- Encodes and decodes sentences, including novel text, with no downloaded tokenizer, proprietary data or model API.
- Records the fraction of raw byte tokens saved for the train, validation and held-out test texts.

**Fewer tokens do not necessarily mean better answers.** The benefit is packing more of a question and answer into a limited context and letting the network learn some frequent text fragments as units.

## Controlled research experiment

A separate new experiment trains two small original Transformers **from random weights** on exactly the same Mars Data V3 training examples:

| Variant | Text encoding |
|---|---|
| `byte-baseline` | The original 256-byte tokenizer |
| `bpe-experiment` | Our new learned Mars BPE vocabulary |

Both use the same seed, hidden width, layer count, batch size, and maximum training steps. Because their vocabulary sizes differ, their parameter counts are **not identical**—we report that difference openly.

Model selection uses **validation tasks only**; the held-out test split is evaluated **once after checkpoint selection**. Scores report **answer bits per raw UTF-8 byte**, because loss per BPE token cannot be directly compared with loss per byte token.

We also save a few genuine generated responses from both checkpoints. An improvement in bits/byte does not prove the generated responses are correct.

## Run from GitHub — no download or PowerShell required

Open the [GitHub Actions page](https://github.com/CORMACTHEGOD/mars-ai/actions) and choose **Compare Mars original byte vs BPE tokenizer**. The first comparison also launches automatically when the experiment code is merged.

A completed run produces an optional artifact named `mars-original-bpe-and-byte-comparison` containing:

- `tokenizer.json` — our exact learned vocabulary and merge rules.
- `tokenizer-report.json` — token compression by dataset split.
- `comparison.json` — the real matched compute-budget comparison, validation loss in bits/byte, held-out scores and generated samples.
- Separate `weights.pt` and `results.json` files for each original model.

You **do not need to download any of these files**. GitHub's Actions logs show the important results. Downloading is only for inspecting or archiving them yourself.

## Why the normal Mars website won't change yet

The new BPE-trained model must **not** be pointed at Mars-1's byte decoder. That would produce incorrect text. A browser version would require the matching vocabulary file, a BPE JavaScript encoder/decoder, and an ONNX export of **the same** trained weights.

We will only attempt that after comparing actual generated answers and deciding whether the experiment merits deployment. Until then:
- [Mars-1](https://cormacthegod.github.io/mars-ai/) remains the normal live site.
- [Mars-2 Lab](https://cormacthegod.github.io/mars-ai/?model=mars2) remains an optional research site.
- The new tokenizer and checkpoints are research files only.

## Security and honesty

There is no scraping, third-party model, API token, paid inference request, or personal-data collection in this experiment. Only our previously authored examples and algorithmically generated simple tasks are used.

Our dataset is still tiny. Training these models for a few hundred steps will likely yield mostly wrong answers. Substantially larger, licensed datasets and more compute are necessary for broadly capable AI.

## Code

- `mars_mini/tokenizer_bpe.py` — deterministic original UTF-8 BPE and training-only vocabulary.
- `mars_mini/experiment_tokenizers.py` — CPU-trained comparison and checkpoint selection.
- `data_tests/test_tokenizer_bpe.py` — tokenizer roundtrip and data provenance tests.
- `tests/test_tokenizer_experiment.py` — PyTorch model/tokenization integration.
- `.github/workflows/compare-mars-tokenizers.yml` — automated, no-terminal experiment.
