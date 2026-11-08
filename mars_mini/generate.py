"""Generate experimental text from Mars-1 Mini weights.

Small models with toy data can produce repetitive or incoherent output. A
checkpoint is required; we do NOT quietly substitute someone else's model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import MarsConfig, MarsMini, decode, encode


def load_checkpoint(path: str | Path) -> MarsMini:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError("Not a Mars-1 Mini checkpoint")
    model = MarsMini(MarsConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def complete(
    model: MarsMini,
    prompt: str,
    max_new_tokens: int = 160,
    temperature: float = 0.85,
) -> str:
    prefix = f"User: {prompt.strip()}\nMars:"
    original_tokens = encode(prefix)
    input_tokens = torch.tensor([original_tokens], dtype=torch.long)
    result = model.generate(
        input_tokens, max_new_tokens=max_new_tokens, temperature=temperature
    )
    completion = decode(result[0].tolist()[len(original_tokens) :])
    # Don't present text from a fictional next turn as the same answer.
    for stop in ("\nUser:", "\nConversation:", "\nMars chat example:"):
        completion = completion.split(stop, 1)[0]
    return completion.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the original tiny Mars model")
    parser.add_argument("--checkpoint", default="artifacts/mars1-mini/mars1-mini.pt")
    parser.add_argument("--prompt", default="hello")
    parser.add_argument("--new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.85)
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint)
    print("Mars-1 Mini (original toy model, not a production assistant):")
    print(complete(model, args.prompt, args.new_tokens, args.temperature))


if __name__ == "__main__":
    main()
