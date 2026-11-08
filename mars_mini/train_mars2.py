"""Mars-2 research training: original random-weight Transformer with real held-out checks.

NOT a capable general chatbot. The validation split rephrases facts seen in
training; separate held-out problems test whether it learned more than lookup.
The best checkpoint is chosen by validation loss, never final training loss.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from .mars2_data import make_mars2_splits, format_summary
from .model import MarsConfig, MarsMini
from .train_dialogue import example_tensors


def tensorize(examples: list[tuple[str, str]], block_size: int):
    pairs = [example_tensors(q, a, block_size) for q, a in examples]
    return torch.stack([p[0] for p in pairs]), torch.stack([p[1] for p in pairs])


@torch.inference_mode()
def measure(model: MarsMini, tensors, batch_size: int = 32) -> float:
    model.eval()
    x, y = tensors
    total = 0.0
    tokens = 0
    for start in range(0, len(x), batch_size):
        inputs = x[start:start + batch_size]
        targets = y[start:start + batch_size]
        _, loss = model(inputs, targets)
        n = int((targets != -100).sum())
        total += float(loss) * n
        tokens += n
    model.train()
    return total / max(tokens, 1)


def run(args: argparse.Namespace) -> Path:
    if not (1 <= args.steps <= 6000 and 1 <= args.batch_size <= 64):
        raise ValueError("steps must be 1..6000 and batch-size must be 1..64")
    if args.eval_every < 1 or args.patience < 1 or args.learning_rate <= 0:
        raise ValueError("Invalid training settings")

    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    cfg = MarsConfig(
        max_seq_len=160, embedding_dim=128, num_heads=4,
        num_layers=3, dropout=0.15,
    )
    cfg.validate()
    splits = make_mars2_splits(args.seed)
    training = tensorize(splits.train, cfg.max_seq_len)
    validation = tensorize(splits.validation, cfg.max_seq_len)
    novel = tensorize(splits.novel_test, cfg.max_seq_len)

    model = MarsMini(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.04
    )
    count = sum(p.numel() for p in model.parameters())
    print(f"Mars-2 Research | {count:,} ORIGINAL parameters | no pretrained weights", flush=True)
    print(format_summary(splits), flush=True)
    print("Validation tests new *wording* of familiar facts; held-out test checks unseen tasks.", flush=True)

    best_val = math.inf
    best_step = 0
    best_state = None
    stale_checks = 0
    samples = []
    model.train()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, training[0].shape[0], (args.batch_size,))
        x, y = training[0][idx], training[1][idx]
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            val_loss = measure(model, validation)
            train_loss = float(loss)
            samples.append({"step": step, "train_loss": train_loss, "paraphrase_val_loss": val_loss})
            if val_loss < best_val - 0.005:
                best_val = val_loss
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
                stale_checks = 0
            else:
                stale_checks += 1
            print(f"step {step:5} | train loss {train_loss:.3f} | "
                  f"unseen-phrasing loss {val_loss:.3f} | best step {best_step}", flush=True)
            if stale_checks >= args.patience:
                print("Early stopping: no better validation results; avoid further memorization.", flush=True)
                break

    if best_state is None:
        raise RuntimeError("No valid checkpoint selected")
    model.load_state_dict(best_state)
    test_loss = measure(model, novel)

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / "mars2-mini.pt"
    torch.save({
        "model_config": vars(cfg),
        "model_state": model.state_dict(),
        "model_name": "Mars-2 Research",
        "best_step": best_step,
        "best_paraphrase_val_loss": best_val,
        "heldout_task_loss": test_loss,
        "training_examples": len(splits.train),
        "validation_examples": len(splits.validation),
        "heldout_test_examples": len(splits.novel_test),
        "source_sha256": hashlib.sha256(
            repr((splits.train, splits.validation, splits.novel_test)).encode("utf-8")
        ).hexdigest(),
        "objective": "question-conditioned answer bytes, early-stopped on paraphrases",
    }, model_path)
    metrics = {
        "model_name": "Mars-2 Research",
        "parameters": count,
        "chosen_step": best_step,
        "requested_steps": args.steps,
        "best_paraphrase_val_loss": best_val,
        "heldout_task_loss": test_loss,
        "note": "Low rephrase loss is NOT proof of knowledge of new facts.",
        "history": samples,
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"BEST checkpoint step {best_step}, unseen-phrasing loss {best_val:.3f}, "
          f"held-out task loss {test_loss:.3f}", flush=True)
    print(f"Saved original model to {model_path}", flush=True)
    return model_path


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate original Mars-2 mini")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.00055)
    parser.add_argument("--eval-every", type=int, default=150)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts/mars2-mini")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
