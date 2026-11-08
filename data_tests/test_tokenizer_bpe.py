"""Tests for Mars BPE that require only Python's standard library."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mars_mini.tokenizer_bpe import (
    MarsBPETokenizer, describe, fit_bpe, training_texts,
)


class OriginalMarsBPETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.texts = training_texts()
        cls.tokenizer = fit_bpe(cls.texts, max_merges=80)

    def test_deterministic_training(self):
        self.assertEqual(
            self.tokenizer.merges,
            fit_bpe(self.texts, max_merges=80).merges,
        )

    def test_vocab_starts_with_all_utf8_bytes(self):
        self.assertGreater(self.tokenizer.vocab_size, 256)
        self.assertLessEqual(self.tokenizer.vocab_size, 336)
        for value in range(256):
            self.assertEqual(self.tokenizer.id_to_bytes[value], bytes([value]))

    def test_roundtrip_including_unseen_unicode(self):
        examples = (
            "hello",
            "Mars-1 Mini is original.",
            "Why is Mars red?",
            "Completely unseen zzzzzwords!",
            "Emoji: 🌠🚀🪐",
            "Newlines\ncount\ttoo",
            "Café — 東京",
            "",
        )
        for text in examples:
            self.assertEqual(text, self.tokenizer.decode(self.tokenizer.encode(text)))

    def test_minimizes_tokens_on_repeated_training_text(self):
        sentence = "User: what is Python?\nMars: Python is a programming language."
        self.assertLess(len(self.tokenizer.encode(sentence)), len(sentence.encode("utf-8")))

    def test_token_serialization_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokenizer.json"
            self.tokenizer.save(path)
            restored = MarsBPETokenizer.load(path)
            self.assertEqual(restored.to_dict(), self.tokenizer.to_dict())
            self.assertEqual(
                restored.encode("A brand new unseen sentence"),
                self.tokenizer.encode("A brand new unseen sentence"),
            )

    def test_rejects_corrupt_vocabulary(self):
        data = self.tokenizer.to_dict()
        data["vocab_size"] += 1
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            MarsBPETokenizer.from_dict(data)
        with self.assertRaises(ValueError):
            MarsBPETokenizer(((256, 0),))

    def test_description_discloses_separate_splits(self):
        report = describe(self.tokenizer)
        self.assertEqual(report["fit_data"], "Mars Data V3 train split ONLY")
        self.assertGreater(report["training"]["examples"], 600)
        self.assertGreater(report["validation"]["examples"], 50)
        self.assertGreater(report["test"]["examples"], 30)
        self.assertGreater(report["training"]["token_reduction_percent"], 0)

    def test_cannot_train_on_empty_or_invalid_settings(self):
        with self.assertRaises(ValueError):
            fit_bpe([])
        with self.assertRaises(ValueError):
            fit_bpe(["hello"], min_frequency=1)


if __name__ == "__main__":
    unittest.main()
