"""Export the trained original Mars-1 Mini model to ONNX for browser inference.

This conversion does NOT retrain or replace the model with a third-party AI.
Its own trained PyTorch weights are embedded into a portable ONNX graph.

Usage:
  python -m mars_mini.export_onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from .generate import load_checkpoint


class LogitsOnly(nn.Module):
    """Wrap the original model to export just the next-token logits."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(tokens)
        return logits


def export_model(checkpoint: str, target: str) -> Path:
    torch.set_num_threads(2)
    model = load_checkpoint(checkpoint)
    wrapper = LogitsOnly(model).eval()
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)

    example = torch.tensor([[85, 115, 101, 114, 58, 32, 104, 105]], dtype=torch.long)
    # The traditional TorchScript exporter supports ONNX opset 17.
    # Dynamic sequence length is essential: a model with a fixed 8-token
    # example input would not work with messages of varying lengths.
    torch.onnx.export(
        wrapper,
        (example,),
        str(path),
        opset_version=17,
        input_names=["tokens"],
        output_names=["logits"],
        dynamic_axes={
            "tokens": {1: "sequence"},
            "logits": {1: "sequence"},
        },
        do_constant_folding=True,
        dynamo=False,
    )

    onnx.checker.check_model(str(path))
    session = ort.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )
    assert session.get_inputs()[0].name == "tokens"
    assert session.get_outputs()[0].name == "logits"
    # Validate both short and full-context inputs, not only the export sample.
    lengths = sorted({1, min(8, model.cfg.max_seq_len),
                      min(32, model.cfg.max_seq_len), model.cfg.max_seq_len})
    for length in lengths:
        tokens = torch.randint(0, 256, (1, length), dtype=torch.long)
        with torch.inference_mode():
            expected, _ = model(tokens)
        predicted = session.run(None, {"tokens": tokens.numpy()})[0]
        np.testing.assert_allclose(
            predicted, expected.numpy(), rtol=1e-3, atol=1e-3
        )
        print(f"Validated ONNX logits against Mars PyTorch: sequence={length}")

    print(f"Exported original Mars-1 Mini to {path} ({path.stat().st_size:,} bytes)")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Mars Mini to browser ONNX")
    parser.add_argument("--checkpoint", default="artifacts/mars1-mini/mars1-mini.pt")
    parser.add_argument("--output", default="models/mars1-mini.onnx")
    args = parser.parse_args()
    export_model(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
