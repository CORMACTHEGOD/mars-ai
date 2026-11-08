"""Transparent Mars-2 qualitative evaluation, including unfamiliar prompts.

Only a small, deterministic sample. This is not a formal intelligence benchmark.
All generated answers come from our model's trained weights, not hard-coded replies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .generate import load_checkpoint
from .mars2_data import make_mars2_splits
from .model import encode, decode


@torch.inference_mode()
def generate_greedy(model, question: str, max_new_tokens: int = 96) -> str:
    model.eval()
    prefix = encode(f"User: {question.strip()}\nMars:")
    tokens = torch.tensor([prefix], dtype=torch.long)
    completed = model.generate(tokens, max_new_tokens=max_new_tokens, top_k=1, temperature=1.0)
    generated = decode(completed[0].tolist()[len(prefix):])
    return generated.split("\n", 1)[0].strip()


def collect_samples(checkpoint: str, output: str) -> dict:
    torch.set_num_threads(2)
    model = load_checkpoint(checkpoint)
    data = make_mars2_splits()
    known = [
        ("hello", "Hello! I am Mars, an experimental AI."),
        ("who are you?", "I am Mars-1 Mini, a tiny model trained from scratch."),
        ("what is Python?", "Python is a programming language."),
        ("why is Mars red?", "Iron oxide dust makes Mars look reddish."),
    ]
    paraphrase = data.validation[:4]
    novel = data.novel_test[:8]
    results = []
    for section, pairs in (
        ("seen_in_training", known),
        ("unseen_wording_known_facts", paraphrase),
        ("heldout_task_or_topic", novel),
    ):
        for question, expected in pairs:
            reply = generate_greedy(model, question)
            results.append({
                "group": section,
                "question": question,
                "expected": expected,
                "generated": reply,
                "exact_match": reply.casefold() == expected.casefold(),
            })
            print(f"[{section}] {question} -> {reply!r}; expected {expected!r}", flush=True)
    summary = {}
    for group in {x["group"] for x in results}:
        items = [x for x in results if x["group"] == group]
        summary[group] = {
            "correct": sum(x["exact_match"] for x in items),
            "total": len(items),
        }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"samples": results, "exact_match_summary": summary}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved explicit evaluation samples to {path}", flush=True)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Mars-2 sample generations")
    parser.add_argument("--checkpoint", default="artifacts/mars2-mini/mars2-mini.pt")
    parser.add_argument("--output", default="artifacts/mars2-mini/samples.json")
    args = parser.parse_args()
    collect_samples(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
