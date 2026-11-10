"""Train a fresh Mars V4 Transformer locally on CPU, with resumable checkpoints.

Usage (Windows PowerShell; from repository root):
  & "$HOME\.venv\Scripts\python.exe" -m mars_mini.train_v4 --minutes 55
  & "$HOME\.venv\Scripts\python.exe" -m mars_mini.train_v4 --minutes 55 --resume

This is an ORIGINAL model trained from random weights. No API/pretrained
weights are used. A large synthetic dataset is not a substitute for broad,
licensed natural-language text; generation will likely remain limited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import torch
from torch.nn import functional as F

from .dataset_v4 import build_pack, summary
from .model import MarsConfig, MarsMini
from .tokenizer_bpe import MarsBPETokenizer, fit_bpe


def pick_balanced(records, per_task: int, seed: int):
    groups = defaultdict(list)
    for item in records:
        groups[item.task].append(item)
    rng = random.Random(seed)
    picked = []
    for key in sorted(groups):
        rows = groups[key]
        if len(rows) > per_task:
            rows = rng.sample(rows, per_task)
        picked.extend(rows)
    return picked


def make_tokenizer(train, merges: int, seed: int) -> MarsBPETokenizer:
    # Fit on TRAIN ONLY, using a balanced sample for quick CPU setup.
    samples = pick_balanced(train, per_task=100, seed=seed)
    texts = []
    for row in samples:
        texts.append(f"User: {row.prompt}\nMars:")
        texts.append(f" {row.answer}\n")
    print(f"Learning {merges} original BPE merges from {len(samples)} TRAIN rows...",
          flush=True)
    return fit_bpe(texts, max_merges=merges)


def encode_example(row, tokenizer: MarsBPETokenizer, context: int):
    prefix = tokenizer.encode(f"User: {row.prompt}\nMars:")
    answer = tokenizer.encode(f" {row.answer}\n")
    sequence = prefix + answer
    if len(sequence) - 1 > context:
        raise ValueError(
            f"Example too long ({len(sequence)-1} tokens): {row.id}; "
            f"increase --context"
        )
    targets = [-100] * (len(prefix) - 1) + sequence[len(prefix):]
    return tuple(sequence[:-1]), tuple(targets)


def make_batch(rows, encode):
    pairs = [encode(row) for row in rows]
    size = max(len(x) for x, _ in pairs)
    x = torch.zeros((len(rows), size), dtype=torch.long)
    y = torch.full((len(rows), size), -100, dtype=torch.long)
    for i, (inputs, targets) in enumerate(pairs):
        length = len(inputs)
        x[i, :length] = torch.tensor(inputs, dtype=torch.long)
        y[i, :length] = torch.tensor(targets, dtype=torch.long)
    return x, y


@torch.inference_mode()
def evaluate(model: MarsMini, rows, encode, batch_size: int = 16):
    """Compute per-task token NLL, then a task-balanced validation score."""
    model.eval()
    sums = defaultdict(lambda: [0.0, 0])
    try:
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            x, y = make_batch(batch_rows, encode)
            logits, _ = model(x)
            raw = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                y.reshape(-1), ignore_index=-100, reduction="none",
            ).reshape_as(y)
            for i, row in enumerate(batch_rows):
                sums[row.task][0] += float(raw[i].sum())
                sums[row.task][1] += int((y[i] != -100).sum())
    finally:
        model.train()
    by_task = {task: total / max(tokens, 1)
               for task, (total, tokens) in sorted(sums.items())}
    return sum(by_task.values()) / len(by_task), by_task


def atomic_save(path: Path, payload: dict) -> None:
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


@torch.inference_mode()
def sample_answer(model: MarsMini, tokenizer: MarsBPETokenizer, question: str) -> str:
    model.eval()
    prefix = tokenizer.encode(f"User: {question}\nMars:")
    model_input = torch.tensor([prefix], dtype=torch.long)
    prediction = model.generate(model_input, max_new_tokens=48, top_k=1)
    return tokenizer.decode(prediction[0].tolist()[len(prefix):]).split("\n")[0].strip()


def run(args) -> None:
    if args.minutes <= 0 or args.steps < 1 or args.batch_size < 1:
        raise ValueError("minutes, steps, and batch-size must be positive")
    if args.eval_every < 1 or args.merges < 0 or args.width % 4:
        raise ValueError("Invalid evaluation, tokenizer, or width settings")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(args.seed)
    data = build_pack(seed=args.seed)
    counts = summary(data)
    print("Mars Data V4:", {k: v["rows"] for k, v in counts["splits"].items()}, flush=True)
    print("Synthetic structured examples are not equivalent to a broad text corpus.", flush=True)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    last_path, best_path = output / "last.pt", output / "best.pt"
    signature = hashlib.sha256(
        "|".join(item.id for item in data.train + data.validation + data.test).encode()
    ).hexdigest()

    saved = None
    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(f"No checkpoint to resume: {last_path}")
        saved = torch.load(last_path, map_location="cpu", weights_only=True)
        if saved["dataset_signature"] != signature:
            raise ValueError("Dataset/seed changed; resume would be unsafe")
        tokenizer = MarsBPETokenizer.from_dict(saved["tokenizer"])
    else:
        tokenizer = make_tokenizer(data.train, args.merges, args.seed)
        tokenizer.save(output / "tokenizer.json")

    cfg = MarsConfig(
        vocab_size=tokenizer.vocab_size, max_seq_len=args.context,
        embedding_dim=args.width, num_heads=4, num_layers=args.layers,
        dropout=0.12,
    )
    cfg.validate()
    if saved and saved["model_config"] != asdict(cfg):
        raise ValueError("Resume settings differ: keep width/layers/context/merges unchanged")
    model = MarsMini(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.03)
    if saved:
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        start_step = int(saved["step"])
        best_score, best_step = saved["best_score"], saved["best_step"]
        rng = random.Random()
        rng.setstate(saved["sampler_rng_state"])
        history = saved["history"]
        print(f"Resuming saved local model from step {start_step}", flush=True)
    else:
        start_step = 0
        best_score, best_step = math.inf, 0
        history = []
        rng = random.Random(args.seed + 500)

    groups = defaultdict(list)
    for item in data.train:
        groups[item.task].append(item)
    tasks = sorted(groups)
    # Do not let the huge arithmetic tables crowd out other tasks.
    task_weights = {
        "conversation": 3, "definition": 2, "reading": 3,
        "word_rules": 2, "addition": 1, "subtraction": 1,
        "comparison": 1, "multiplication": 1, "parity": 1,
    }
    weights = [task_weights.get(task, 1) for task in tasks]
    validation = pick_balanced(data.validation, per_task=50, seed=args.seed + 1)
    test = pick_balanced(data.test, per_task=50, seed=args.seed + 2)

    @lru_cache(maxsize=10000)
    def encode(row):
        return encode_example(row, tokenizer, cfg.max_seq_len)

    # Validate the held-out encoding lengths without revealing test answers to
    # the optimizer. Test metrics are computed only at the end.
    for row in validation + test:
        encode(row)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} original parameters; "
          f"vocabulary {cfg.vocab_size}; context {cfg.max_seq_len}", flush=True)
    print(f"Training up to {args.minutes:g} minutes or {args.steps:,} total steps; "
          "saving checkpoints every evaluation.", flush=True)

    def snapshot(step: int):
        return {
            "step": step, "best_score": best_score, "best_step": best_step,
            "model_config": asdict(cfg),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "sampler_rng_state": rng.getstate(),
            "tokenizer": tokenizer.to_dict(),
            "dataset_signature": signature,
            "history": history,
        }

    start = time.monotonic()
    end_step = start_step
    try:
        for step in range(start_step + 1, args.steps + 1):
            if time.monotonic() - start >= args.minutes * 60:
                print("Training time limit reached.", flush=True)
                break
            selected_tasks = rng.choices(tasks, weights=weights, k=args.batch_size)
            batch_rows = [rng.choice(groups[task]) for task in selected_tasks]
            x, y = make_batch(batch_rows, encode)
            model.train()
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            end_step = step

            if step == 1 or step % args.eval_every == 0:
                score, by_task = evaluate(model, validation, encode)
                improved = score < best_score - 0.002
                if improved:
                    best_score, best_step = score, step
                elapsed = (time.monotonic() - start) / 60
                history.append({
                    "step": step, "train_batch_loss": float(loss.detach()),
                    "validation_score": score, "validation_by_task": by_task,
                    "elapsed_minutes_this_run": round(elapsed, 2),
                })
                atomic_save(last_path, snapshot(step))
                if improved:
                    atomic_save(best_path, {
                        "model_config": asdict(cfg),
                        "model_state": model.state_dict(),
                        "tokenizer": tokenizer.to_dict(),
                        "step": step,
                        "validation_score": score,
                        "dataset_signature": signature,
                    })
                print(
                    f"step {step:6d} | train loss {float(loss.detach()):.3f} | "
                    f"balanced validation {score:.3f} | best {best_score:.3f} "
                    f"at step {best_step} | elapsed {elapsed:.1f}m",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\nTraining interrupted; saving progress safely.", flush=True)
    finally:
        # Save even if time limit or Ctrl+C occurred between evaluations.
        atomic_save(last_path, snapshot(end_step))
        print(f"Last resumable checkpoint: {last_path}", flush=True)

    if not best_path.exists():
        score, _ = evaluate(model, validation, encode)
        best_score, best_step = score, end_step
        atomic_save(best_path, {
            "model_config": asdict(cfg), "model_state": model.state_dict(),
            "tokenizer": tokenizer.to_dict(), "step": end_step,
            "validation_score": score, "dataset_signature": signature,
        })

    best = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best["model_state"])
    test_score, per_task = evaluate(model, test, encode)
    report = {
        "dataset": counts,
        "model_config": asdict(cfg),
        "completed_step": end_step,
        "best_step": best["step"],
        "best_validation_score": best["validation_score"],
        "test_score_once_after_training": test_score,
        "test_by_task": per_task,
        "history": history,
        "warning": "Loss is not answer correctness; trained on templated data, not fluent language.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n",
                                        encoding="utf-8")
    print(f"HELD-OUT test average loss: {test_score:.3f} (lower is better)", flush=True)
    for q in ("Hello, can we chat?", "What is 18 plus 7?",
              "Read: Ada left the blue book in the library. Where did Ada leave the blue book?"):
        print(f"Mars V4 sample: {q} -> {sample_answer(model, tokenizer, q)!r}", flush=True)
    print(f"Best model: {best_path}", flush=True)
    print(f"Report: {output / 'report.json'}", flush=True)
    print("This does NOT change the live chatmarsai.com model.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Original Mars V4 local CPU trainer")
    parser.add_argument("--minutes", type=float, default=55)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--merges", type=int, default=96)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.00055)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts/mars-v4-local")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
