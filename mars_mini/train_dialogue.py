"""Conversation-focused training for Mars-1 Mini, from entirely random weights.

The previous experiment predicted ALL bytes of a long concatenated corpus,
including headers and questions. This training objective focuses gradient
updates on the ANSWER bytes after 'User: ...\\nMars:' so that the same tiny
model has a better chance to learn question-to-answer patterns.

This is still a tiny experimental model, not a reliable chatbot.
It uses original example pairs from mars_mini.corpus; no downloaded
pretrained weights, proprietary data, inference providers, or API keys.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
from pathlib import Path

import torch

from .corpus import DIALOGUES, TOPICS
from .model import MarsConfig, MarsMini, encode


def make_examples() -> list[tuple[str, str]]:
    examples = list(DIALOGUES)
    for topic, definition in TOPICS:
        answer = f"A {topic} is {definition}."
        examples.extend((
            (f"What is a {topic}?", answer),
            (f"Define {topic}.", answer),
        ))
    # A few arithmetic examples help demonstrate learned answer patterns.
    for a in range(1, 8):
        for b in range(1, 8):
            examples.append((f"What is {a} plus {b}?", f"{a + b}."))
    return examples


def example_tensors(
    question: str, answer: str, context_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce next-byte inputs, with loss only on the answer bytes.

    The prefix tokens are conditioning context, not target text. Every
    pad position gets -100 (cross-entropy ignore_index).
    """
    prefix = encode(f"User: {question.strip()}\nMars:")
    full = prefix + encode(f" {answer.strip()}\n")
    if len(full) <= len(prefix) or len(full) < 2:
        raise ValueError("Answer must contain at least one token")
    if len(full) - 1 > context_size:
        raise ValueError("Example is too long for the model context")

    x = full[:-1] + [0] * (context_size - len(full) + 1)
    y = full[1:] + [-100] * (context_size - len(full) + 1)
    # The token at index len(prefix)-1 predicts the first answer byte.
    for i in range(len(prefix) - 1):
        y[i] = -100
    assert any(token != -100 for token in y)
    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def create_splits(
    context_size: int, seed: int
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    # Keep the core conversation prompts (hello, who are you, etc.) in the
    # training set. Hold out a fraction of generated definitions/arithmetic
    # for a genuine, non-overlapping generalization check.
    rng = random.Random(seed)
    core = list(DIALOGUES)
    derived = make_examples()[len(core):]
    rng.shuffle(derived)
    split = int(0.85 * len(derived))
    training = core + derived[:split]
    validation = derived[split:]

    def tensors(examples):
        items = [example_tensors(q, a, context_size) for q, a in examples]
        return torch.stack([x for x, _ in items]), torch.stack([y for _, y in items])

    return tensors(training), tensors(validation)


def batch(data: tuple[torch.Tensor, torch.Tensor], batch_size: int):
    xs, ys = data
    indexes = torch.randint(xs.shape[0], (batch_size,))
    return xs[indexes], ys[indexes]


@torch.no_grad()
def evaluate(model: MarsMini, data, n_batches: int = 8) -> float:
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = batch(data, min(16, data[0].shape[0]))
        _, loss = model(x, y)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


def run(args: argparse.Namespace) -> Path:
    if args.steps < 1 or args.steps > 10000:
        raise ValueError("steps must be between 1 and 10000")
    if args.batch_size < 1 or args.batch_size > 128:
        raise ValueError("batch-size must be between 1 and 128")
    if args.learning_rate <= 0 or args.log_every < 1:
        raise ValueError("Invalid learning rate or log frequency")

    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    cfg = MarsConfig(
        max_seq_len=128, embedding_dim=96, num_heads=4, num_layers=2, dropout=0.1
    )
    cfg.validate()
    training, validation = create_splits(cfg.max_seq_len, args.seed)
    model = MarsMini(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    count = sum(p.numel() for p in model.parameters())
    print(f"Mars-1 Mini | original random weights | {count:,} parameters", flush=True)
    print(f"Train dialogue pairs: {training[0].shape[0]} | held out: {validation[0].shape[0]}", flush=True)
    print("Objective: predict the response bytes conditioned on the user question.", flush=True)

    model.train()
    for step in range(1, args.steps + 1):
        x, y = batch(training, args.batch_size)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            val_loss = evaluate(model, validation)
            print(f"step {step:5} | answer loss {loss.item():.4f} | "
                  f"held-out answer loss {val_loss:.4f}", flush=True)

    path = Path(args.output) / "mars1-mini.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_config": vars(cfg),
        "model_state": model.state_dict(),
        "training_steps": args.steps,
        "training_seed": args.seed,
        "objective": "question-conditioned answer-only next-byte prediction",
        "training_examples": int(training[0].shape[0]),
        "heldout_examples": int(validation[0].shape[0]),
        "corpus_sha256": hashlib.sha256(repr(make_examples()).encode("utf-8")).hexdigest(),
    }, path)
    print(f"Saved trained Mars weights to {path}", flush=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train original Mars-1 Mini on short dialogues")
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0006)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--output", default="artifacts/mars1-mini")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
