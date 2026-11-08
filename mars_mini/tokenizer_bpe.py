"""Mars BPE tokenizer V1: original deterministic byte-pair encoding.

Trains only on our original TRAINING split, never validation or test data.
Starts with 256 raw byte symbols, learns frequent adjacent byte/token pairs,
and can represent ANY UTF-8 string, including unfamiliar words and emoji.
No downloaded tokenizer, pretrained vocabulary, external package or API key.

This is a research tokenizer, not by itself evidence of language understanding.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .dataset_v3 import build_pack


@dataclass(frozen=True)
class MarsBPETokenizer:
    merges: tuple[tuple[int, int], ...]
    version: str = "mars-bpe-v1"

    def __post_init__(self):
        if self.version != "mars-bpe-v1":
            raise ValueError("Unsupported Mars tokenizer version")
        vocab_size = 256
        for left, right in self.merges:
            if not (0 <= left < vocab_size and 0 <= right < vocab_size):
                raise ValueError("Invalid or forward-referencing BPE merge")
            vocab_size += 1

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    @property
    def id_to_bytes(self) -> tuple[bytes, ...]:
        vocab = [bytes([i]) for i in range(256)]
        for left, right in self.merges:
            vocab.append(vocab[left] + vocab[right])
        return tuple(vocab)

    def encode(self, text: str) -> list[int]:
        sequence = list(text.encode("utf-8"))
        if len(sequence) < 2:
            return sequence
        # Earliest learned merge has precedence. Applying merges in ranked
        # order is deterministic and reproduces BPE used during learning.
        for new_id, pair in enumerate(self.merges, 256):
            sequence = _merge_pair(sequence, pair, new_id)
        return sequence

    def decode(self, ids: Iterable[int]) -> str:
        vocab = self.id_to_bytes
        raw = bytearray()
        for token in ids:
            if not isinstance(token, int) or not 0 <= token < len(vocab):
                raise ValueError(f"Unknown Mars token ID: {token!r}")
            raw.extend(vocab[token])
        return raw.decode("utf-8", errors="replace")

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "vocab_size": self.vocab_size,
            "merges": [list(pair) for pair in self.merges],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> MarsBPETokenizer:
        if not isinstance(payload, dict) or payload.get("version") != "mars-bpe-v1":
            raise ValueError("Invalid Mars BPE tokenizer metadata")
        pairs = payload.get("merges")
        if not isinstance(pairs, list) or any(
            not isinstance(p, list) or len(p) != 2
            or any(type(i) is not int for i in p) for p in pairs
        ):
            raise ValueError("Invalid BPE merge list")
        tokenizer = cls(tuple((p[0], p[1]) for p in pairs))
        if payload.get("vocab_size") != tokenizer.vocab_size:
            raise ValueError("Tokenizer vocabulary size mismatch")
        return tokenizer

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> MarsBPETokenizer:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _merge_pair(sequence: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    output = []
    index = 0
    while index < len(sequence):
        if index + 1 < len(sequence) and (
            sequence[index], sequence[index + 1]
        ) == pair:
            output.append(new_id)
            index += 2
        else:
            output.append(sequence[index])
            index += 1
    return output


def fit_bpe(texts: Iterable[str], max_merges: int = 192,
            min_frequency: int = 2) -> MarsBPETokenizer:
    if not 0 <= max_merges <= 1024 or min_frequency < 2:
        raise ValueError("Invalid tokenizer training settings")
    sequences = [list(text.encode("utf-8")) for text in texts]
    if not sequences:
        raise ValueError("Cannot train BPE on an empty corpus")
    merges: list[tuple[int, int]] = []

    for _ in range(max_merges):
        frequencies: Counter[tuple[int, int]] = Counter()
        for sequence in sequences:
            frequencies.update(zip(sequence, sequence[1:]))
        if not frequencies:
            break
        # Tie break by token IDs so repeated training yields identical merges.
        pair, freq = sorted(frequencies.items(), key=lambda x: (-x[1], x[0]))[0]
        if freq < min_frequency:
            break
        new_id = 256 + len(merges)
        sequences = [_merge_pair(seq, pair, new_id) for seq in sequences]
        merges.append(pair)

    return MarsBPETokenizer(tuple(merges))


def training_texts() -> list[str]:
    """Never include validation/test strings in vocabulary fitting."""
    data = build_pack()
    # Keep the BPE boundaries consistent with prefix/answer encoding later.
    lines = []
    for example in data.train:
        lines.append(f"User: {example.prompt}\nMars:")
        lines.append(f" {example.answer}\n")
    return lines


def describe(tokenizer: MarsBPETokenizer) -> dict:
    data = build_pack()
    def lengths(examples):
        old = new = 0
        for e in examples:
            text = f"User: {e.prompt}\nMars: {e.answer}\n"
            old += len(text.encode("utf-8"))
            new += len(tokenizer.encode(text))
        return {
            "examples": len(examples),
            "original_byte_tokens": old,
            "bpe_tokens": new,
            "token_reduction_percent": round(100 * (1 - new / old), 2) if old else 0,
        }

    return {
        "version": tokenizer.version,
        "vocab_size": tokenizer.vocab_size,
        "merges_learned": len(tokenizer.merges),
        "fit_data": "Mars Data V3 train split ONLY",
        "training": lengths(data.train),
        "validation": lengths(data.validation),
        "test": lengths(data.test),
        "limitation": "Token compression alone does not create factual understanding.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Mars BPE from original train split")
    parser.add_argument("--merges", type=int, default=192)
    parser.add_argument("--output", default="artifacts/mars-tokenizer-v1")
    args = parser.parse_args()
    tokenizer = fit_bpe(training_texts(), max_merges=args.merges)
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer.save(directory / "tokenizer.json")
    report = describe(tokenizer)
    (directory / "tokenizer-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
