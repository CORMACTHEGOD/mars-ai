"""Train Mars-1 Mini from random initialization.

Example: python -m mars_mini.train --steps 150
No internet, API key, external dataset, or GPU is required for the toy run.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path

import torch

from .corpus import build_corpus
from .model import MarsConfig, MarsMini, encode


def sample_batch(data: torch.Tensor, batch_size: int, block_size: int):
    # Sample random contiguous character windows and next-byte targets.
    maximum_start = data.numel() - block_size
    if maximum_start <= 0:
        raise ValueError("Training corpus is shorter than the sequence length")
    starts = torch.randint(0, maximum_start, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x, y


@torch.no_grad()
def evaluate(model: MarsMini, validation: torch.Tensor, block_size: int, batches: int = 4) -> float:
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = sample_batch(validation, 8, block_size)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def run(args: argparse.Namespace) -> Path:
    if args.steps < 1 or args.steps > 10_000:
        raise ValueError("steps must be between 1 and 10,000")
    if args.batch_size < 1 or args.batch_size > 128:
        raise ValueError("batch-size must be between 1 and 128")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")

    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    config = MarsConfig(
        max_seq_len=args.block_size,
        embedding_dim=args.width,
        num_heads=args.heads,
        num_layers=args.layers,
        dropout=0.1,
    )
    config.validate()

    text = build_corpus(seed=args.seed)
    tokens = torch.tensor(encode(text), dtype=torch.long)
    split = int(tokens.numel() * 0.90)
    train = tokens[:split]
    val = tokens[split:]
    if min(train.numel(), val.numel()) <= config.max_seq_len:
        raise ValueError("Training or validation split is too short")

    model = MarsMini(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"Original Mars-1 Mini | parameters: {parameters:,}")
    print(f"Corpus: {tokens.numel():,} UTF-8 bytes | training steps: {args.steps}")
    print("CPU training from random weights; not a production chat model.")

    model.train()
    last_loss = math.nan
    for step in range(1, args.steps + 1):
        x, y = sample_batch(train, args.batch_size, config.max_seq_len)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.item())

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            val_loss = evaluate(model, val, config.max_seq_len)
            print(f"step {step:>5} | train loss {last_loss:.3f} | val loss {val_loss:.3f}", flush=True)

    checkpoint_dir = Path(args.output)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target = checkpoint_dir / "mars1-mini.pt"
    torch.save(
        {
            "model_config": vars(config),
            "model_state": model.state_dict(),
            "training_steps": args.steps,
            "training_seed": args.seed,
            "training_corpus_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "last_training_loss": last_loss,
        },
        target,
    )
    print(f"Saved original weights to {target}", flush=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Mars-1 Mini (experimental)")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--block-size", type=int, default=96)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.0006)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=30)
    parser.add_argument("--output", default="artifacts/mars1-mini")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
