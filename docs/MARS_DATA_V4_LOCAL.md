# Mars Data V4 — local Windows training

This is a **separate research model**, not a new public Mars chatbot.
It does not modify the weights used by chatmarsai.com.

## What is in Data V4?

- Deterministically generated addition, subtraction, comparison, parity, and multiplication questions.
- Short reading passages with names, objects, and locations, followed by answer questions.
- Word reversal and capitalization tasks.
- Short, originally written definitions and conversations from the Mars project.

**Limitation:** Many data rows come from the same small set of templates.
The much larger row count does not mean Mars has seen tens of thousands
of independent natural conversations. It is still unlikely to give fluent
general-purpose answers. This experiment measures whether we can begin
learning compositional tasks with properly held-out input families.

## Start on the Surface laptop

Open PowerShell in the cloned Mars project folder, then run these
commands one at a time:

    cd "$HOME\mars-ai"
    git pull
    & "$HOME\.venv\Scripts\python.exe" -m mars_mini.train_v4 --minutes 55

The first time, it prepares the training data and learns its BPE vocabulary
from the training split only. The 55-minute limit begins when actual
training starts. The model is a three-layer, 128-width, original Transformer.

Keep Windows awake and the laptop plugged in with unobstructed ventilation.
You can use other programs, but training will be slower.

## Continue after stopping or completing

    cd "$HOME\mars-ai"
    & "$HOME\.venv\Scripts\python.exe" -m mars_mini.train_v4 --minutes 55 --resume

The script saves files inside artifacts/mars-v4-local:

- last.pt — your latest model, optimizer state, and position for resuming.
- best.pt — the checkpoint with the lowest validation loss.
- tokenizer.json — the matching learned BPE vocabulary.
- report.json — the training history and final held-out loss.

Ctrl+C saves a resumable checkpoint. Do not delete these files if you
want to continue training. This directory is ignored by Git.

## Interpreting results

The validation/test scores are **prediction losses**, not pass rates or
proof of correct answers. The held-out tasks are structurally separate
from training by arithmetic operands or the story inputs, but share
many sentence templates and vocabulary with training. Compare actual
outputs to expected answers before deploying anything.

Future work: licensed, varied, genuinely authored natural-language
conversation corpora, better context handling, and more evaluation
of unfamiliar prompts.
