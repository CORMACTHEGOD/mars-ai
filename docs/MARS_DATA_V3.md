# Mars Data V3 — building better learning and honest evaluation

This is the **next dataset/evaluation project**, not a new trained AI model. Existing Mars-1 and Mars-2 browser models are unchanged.

## Why

Mars-2 learned a few memorized answers (such as “hello”) but gave gibberish or wrong arithmetic answers for unfamiliar questions. More training on the same 716 examples is unlikely to fix that.

We now make a *reproducible, readable data pack* and tests that separate learning example phrases from actually applying a rule to new inputs.

## What is in the pack?

All content is authored directly in `mars_mini/dataset_v3.py` or drawn from our previous original `mars_mini/corpus.py`. No datasets are downloaded, scraped, or copied from proprietary chat logs.

- **Known chat dialogues:** Original Mars responses, with variations of how to phrase the question. Their validation answers are already seen in training; this checks wording robustness only.
- **Original short definitions:** Compact, readable facts. Four concept families (`latency`, `recursion`, `biodiversity`, `asteroid`) are entirely held out from this dataset's training examples.
- **Addition:** Deterministic sums of values 0–20. The pair `3 + 8` always stays in the same split as `8 + 3`; this prevents accidental train/test leakage from reversing operands.
- **Text operations:** Uppercase and reverse individual words. The *entire input word* is held out in validation or test, not just a different phrasing of the same word.
- **Metadata:** Every example includes an ID, original provenance, task label and family ID.

The test and validation sets are public and inspectable, not secret benchmarks. **A low validation loss, even with this dataset, is not evidence of general intelligence.**

## Generate the pack — without a terminal

GitHub Actions automatically checks the pack when code changes are merged. You can also start the **Check Mars data V3 (no model training)** workflow manually on [GitHub Actions](https://github.com/CORMACTHEGOD/mars-ai/actions).

The artifact **`mars-data-v3-audited`** contains:

| File | What it shows |
|---|---|
| `train.jsonl` | Original practice questions and expected answers |
| `validation.jsonl` | Reworded familiar facts and held-out task inputs |
| `test.jsonl` | Unseen definition families, addition pairs, and word inputs |
| `manifest.json` | Counts by task, context length, hashes, data provenance and limitations |
| `memorization-baseline.json` | How an exact prompt-lookup system scores against each split |

The baseline deliberately returns a known answer only when the *same normalized question* appears in training. It should score perfectly on its own training questions and fail on genuinely unseen inputs, showing why memorization is not the same as learning.

## Optional Python commands

The data tools themselves use the Python standard library — **no PyTorch, GPU or API keys are required**:

```bash
python -m unittest discover -s data_tests -v
python -m mars_mini.dataset_v3 --output artifacts/mars-data-v3
python -m mars_mini.score_v3 --baseline
```

When we eventually train a new model and collect predictions, write one JSON object per line as `{"id":"...","response":"model answer"}`, then run:

```bash
python -m mars_mini.score_v3 --predictions predictions.jsonl
```

The evaluator checks exact and normalized text matches per task, counts missing answers as incorrect, and rejects duplicate or unknown prediction IDs. It **does not** pretend that exact text matching fully measures language understanding.

## Safety and privacy

Do not add names, phone numbers, home addresses, school records, passwords, private chats, or other personal material to public training data. New examples should be original, age-appropriate, short, relevant, and accurately answered. Do not copy copyrighted books or model responses into the repository as training data.

## Next engineering stage

After inspecting the generated dataset and baseline, we can experiment with a more efficient tokenizer and a training-only model that consumes these examples. That will require new training code, a separate evaluation of held-out accuracy, and training compute.

**No model training or browser model replacement is triggered by this data-pack workflow.** Mars-1 remains the normal chat; Mars-2 remains the opt-in research experiment.
