"""Train Mars' original BPE experiment and export its paired vocabulary.

This creates a FRESH model from random weights, not weights from an API.
Weights and tokenizer are published together for an opt-in GitHub Pages
browser demonstration; regular Mars-1/Mars-2 remain unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from .dataset_v3 import build_pack
from .experiment_tokenizers import run_one
from .tokenizer_bpe import MarsBPETokenizer, fit_bpe, training_texts


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    data = build_pack(seed=args.seed)
    tokenizer = fit_bpe(training_texts(), max_merges=args.merges)
    root = Path(args.output)
    metrics = run_one("bpe-experiment", tokenizer, data, args, root)
    checkpoint_path = root / "bpe-experiment" / "weights.pt"

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if saved["tokenizer"] != tokenizer.to_dict():
        raise RuntimeError("Model's tokenizer differs from published BPE vocabulary")
    if saved["model_config"]["vocab_size"] != tokenizer.vocab_size:
        raise RuntimeError("Model vocabulary size differs from tokenizer")

    vocabulary_path = Path(args.vocabulary_output)
    vocabulary_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(vocabulary_path)
    (root / "browser-model-report.json").write_text(
        json.dumps({
            "model_name": "Mars BPE Lab — original research model",
            "model_vocab_size": tokenizer.vocab_size,
            "training": "Original Mars Data V3 training split",
            "evaluation": metrics,
            "limitations": [
                "This is not a generally capable conversational AI.",
                "Generated answers on unfamiliar prompts are often incorrect.",
                "Previous Mars models remain unchanged.",
            ],
        }, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Matching original BPE checkpoint: {checkpoint_path}", flush=True)
    print(f"Matching BPE browser vocabulary: {vocabulary_path}", flush=True)
    return checkpoint_path, vocabulary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train original Mars BPE browser model")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--merges", type=int, default=192)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.0006)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts/mars-bpe-browser")
    parser.add_argument("--vocabulary-output", default="models/mars-bpe-mini.tokenizer.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
