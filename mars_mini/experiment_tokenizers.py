"""Controlled Mars V3 experiment: byte tokenizer versus original Mars BPE.

Both models start from RANDOM weights and see the SAME original training rows.
We compare held-out answer bits-per-byte (not incomparable token losses),
without using test examples to tune training or vocabulary.

This is an experiment, not a model upgrade or new general chatbot.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from .dataset_v3 import Example, build_pack
from .model import MarsConfig, MarsMini, encode as encode_bytes
from .tokenizer_bpe import MarsBPETokenizer, fit_bpe, training_texts


class ByteTokenizer:
    vocab_size = 256

    @staticmethod
    def encode(text: str) -> list[int]:
        return encode_bytes(text)

    @staticmethod
    def decode(ids: list[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")

    @staticmethod
    def to_dict() -> dict:
        return {"version": "mars-byte-v1", "vocab_size": 256}


@dataclass
class TensorExample:
    inputs: list[int]
    targets: list[int]
    answer_bytes: int
    task: str


def encode_example(example: Example, tokenizer, block_size: int) -> TensorExample:
    prefix = tokenizer.encode(f"User: {example.prompt}\nMars:")
    answer = tokenizer.encode(f" {example.answer}\n")
    sequence = prefix + answer
    if len(sequence) < 2 or len(sequence) - 1 > block_size:
        raise ValueError(
            f"Row {example.id} needs {len(sequence)-1} tokens; max {block_size}"
        )
    targets = [-100] * (len(prefix) - 1) + sequence[len(prefix):]
    if len(targets) != len(sequence) - 1:
        raise RuntimeError("Answer mask does not match model tokens")
    return TensorExample(
        inputs=sequence[:-1],
        targets=targets,
        answer_bytes=len(f" {example.answer}\n".encode("utf-8")),
        task=example.task,
    )


def pack_batch(examples: list[TensorExample]):
    longest = max(len(row.inputs) for row in examples)
    x = torch.zeros((len(examples), longest), dtype=torch.long)
    y = torch.full((len(examples), longest), -100, dtype=torch.long)
    for i, row in enumerate(examples):
        n = len(row.inputs)
        x[i, :n] = torch.tensor(row.inputs, dtype=torch.long)
        y[i, :n] = torch.tensor(row.targets, dtype=torch.long)
    return x, y


@torch.inference_mode()
def measure_bits_per_byte(model: MarsMini, data: list[TensorExample],
                          batch_size: int = 16) -> dict[str, float]:
    """Compare the same raw UTF-8 answers using normalized log loss."""
    model.eval()
    per_task: dict[str, list[float]] = {}
    try:
        for start in range(0, len(data), batch_size):
            group = data[start:start+batch_size]
            x, y = pack_batch(group)
            logits, _ = model(x)
            # Sum NLL only for real answer tokens; no prompt or pad loss.
            raw = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1), ignore_index=-100, reduction="none",
            ).reshape_as(y)
            for i, row in enumerate(group):
                task = per_task.setdefault(row.task, [0.0, 0.0])
                task[0] += float(raw[i].sum())
                task[1] += row.answer_bytes
    finally:
        model.train()

    return {
        task: (value[0] / max(1, value[1])) / math.log(2)
        for task, value in per_task.items()
    }


def balanced_score(metrics: dict[str, float]) -> float:
    weights = {
        "chat_paraphrase": 0.40,
        "definition_paraphrase": 0.25,
        "validation_addition_pair": 0.20,
        "word_transform": 0.15,
    }
    if not all(task in metrics for task in weights):
        raise RuntimeError("Validation tasks are incomplete")
    return sum(weight * metrics[task] for task, weight in weights.items())


@torch.inference_mode()
def generate_greedy(model: MarsMini, tokenizer, question: str,
                    limit: int = 90) -> str:
    """Inspect only a short original output; do not substitute expected text."""
    model.eval()
    prefix = tokenizer.encode(f"User: {question}\nMars:")
    sequence = torch.tensor([prefix], dtype=torch.long)
    completion = model.generate(
        sequence, max_new_tokens=limit,
        temperature=1.0, top_k=1,
    )
    generated = completion[0].tolist()[len(prefix):]
    return tokenizer.decode(generated).split("\n", 1)[0].strip()


def run_one(name: str, tokenizer, data, args, root: Path) -> dict:
    torch.manual_seed(args.seed)
    cfg = MarsConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=160,
        embedding_dim=args.width,
        num_layers=args.layers,
        num_heads=4,
        dropout=0.13,
    )
    cfg.validate()
    training = [encode_example(e, tokenizer, cfg.max_seq_len) for e in data.train]
    validation = [encode_example(e, tokenizer, cfg.max_seq_len) for e in data.validation]
    heldout = [encode_example(e, tokenizer, cfg.max_seq_len) for e in data.test]

    model = MarsMini(cfg)
    count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.05
    )
    generator = torch.Generator().manual_seed(args.seed + 100)
    history = []
    best_state = None
    best_step = 0
    best_score = math.inf
    stale_checks = 0

    print(f"{name}: {count:,} ORIGINAL parameters; vocab {tokenizer.vocab_size}", flush=True)
    for step in range(1, args.steps + 1):
        ids = torch.randint(
            0, len(training), (args.batch_size,), generator=generator
        ).tolist()
        x, y = pack_batch([training[i] for i in ids])
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            val_metrics = measure_bits_per_byte(model, validation)
            score = balanced_score(val_metrics)
            entry = {
                "step": step,
                "train_token_loss": float(loss),
                "weighted_validation_bits_per_byte": score,
                "validation_by_task_bits_per_byte": val_metrics,
            }
            history.append(entry)
            if score < best_score - 0.002:
                best_score = score
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
                stale_checks = 0
            else:
                stale_checks += 1
            print(f"{name} step {step:4d}: validation bits/byte={score:.3f}, "
                  f"best step={best_step}", flush=True)
            if step >= args.steps // 2 and stale_checks >= args.patience:
                print(f"{name}: early stop to limit memorization", flush=True)
                break

    if best_state is None:
        raise RuntimeError("No checkpoint selected")
    model.load_state_dict(best_state)
    # The held-out split is evaluated ONCE after selection, never for tuning.
    test_metrics = measure_bits_per_byte(model, heldout)
    report = {
        "model": name,
        "steps_selected": best_step,
        "maximum_steps": args.steps,
        "parameters": count,
        "vocab_size": tokenizer.vocab_size,
        "validation_bits_per_byte_weighted": best_score,
        "heldout_bits_per_byte_by_task": test_metrics,
        "caution": (
            "Bits/byte on held-out reference answers measures prediction, not "
            "whether generated responses are correct or generally intelligent."
        ),
        "history": history,
    }

    # Inspect a few held-out generations. Avoid interpreting a single
    # generated example as a generalization score.
    sampled = []
    for group_name, rows in (
        ("known", data.train[:3]),
        ("unseen_wording", data.validation[:3]),
        ("heldout", data.test[:5]),
    ):
        for item in rows:
            predicted = generate_greedy(model, tokenizer, item.prompt, limit=64)
            sampled.append({
                "group": group_name,
                "task": item.task,
                "prompt": item.prompt,
                "expected": item.answer,
                "actual": predicted,
                "exact_match": predicted.casefold() == item.answer.casefold(),
            })
            print(f"{name} [{group_name}] {item.prompt} -> {predicted!r}", flush=True)
    report["generated_samples"] = sampled

    save_root = root / name
    save_root.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_config": vars(cfg),
        "model_state": model.state_dict(),
        "tokenizer": tokenizer.to_dict(),
        "dataset": "mars-data-v3",
        "best_step": best_step,
    }, save_root / "weights.pt")
    (save_root / "results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{name}: saved research checkpoint and results", flush=True)
    return report


def run(args: argparse.Namespace) -> dict:
    if not 1 <= args.steps <= 3000 or not 1 <= args.batch_size <= 64:
        raise ValueError("Invalid steps or batch-size")
    if args.eval_every < 1 or args.patience < 1 or args.width % 4:
        raise ValueError("Invalid evaluation settings or width")
    if args.layers < 1 or args.learning_rate <= 0:
        raise ValueError("Invalid model training configuration")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    pack = build_pack(seed=args.seed)
    bpe = fit_bpe(training_texts(), max_merges=args.merges)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    bpe.save(output / "mars-bpe-tokenizer.json")

    results = {
        "dataset": "mars-data-v3",
        "comparison": "same dataset, seed, architecture width/depth, steps and batch size",
        "warning": (
            "Different vocabulary sizes mean different parameter counts. "
            "Do not directly compare token cross-entropy across tokenizers; "
            "we normalize answer log likelihood to bits per raw UTF-8 byte."
        ),
        "models": {},
    }
    for name, tokenizer in (
        ("byte-baseline", ByteTokenizer()),
        ("bpe-experiment", bpe),
    ):
        results["models"][name] = run_one(
            name, tokenizer, pack, args, output
        )
    (output / "comparison.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print("Controlled Mars byte-vs-BPE comparison completed", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Train byte/BPE Mars research baseline")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--merges", type=int, default=192)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.0006)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts/mars-tokenizer-comparison")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
