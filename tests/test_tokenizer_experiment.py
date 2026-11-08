"""Small PyTorch smoke tests for original BPE Mars experiment."""
from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from mars_mini.dataset_v3 import build_pack
from mars_mini.experiment_tokenizers import (
    ByteTokenizer, balanced_score, encode_example,
    pack_batch, measure_bits_per_byte, run,
)
from mars_mini.model import MarsConfig, MarsMini
from mars_mini.tokenizer_bpe import fit_bpe, training_texts, MarsBPETokenizer


class MarsTokenizerModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack = build_pack()
        cls.tokenizer = fit_bpe(training_texts(), max_merges=16)

    def test_model_supports_legacy_and_new_vocabularies(self):
        for size in (256, self.tokenizer.vocab_size, 512):
            cfg = MarsConfig(vocab_size=size, max_seq_len=160,
                             embedding_dim=32, num_heads=4, num_layers=1,
                             dropout=0.0)
            model = MarsMini(cfg)
            logits, loss = model(
                torch.tensor([[0, min(size - 1, 256)]], dtype=torch.long),
                torch.tensor([[0, 5]], dtype=torch.long),
            )
            self.assertEqual(logits.shape, (1, 2, size))
            self.assertTrue(torch.isfinite(loss))
        with self.assertRaises(ValueError):
            MarsConfig(vocab_size=255).validate()

    def test_bpe_answer_mask_aligns_with_prompt(self):
        question = self.pack.train[0]
        e = encode_example(question, self.tokenizer, 160)
        prefix = self.tokenizer.encode(f"User: {question.prompt}\nMars:")
        self.assertEqual(
            e.targets[:len(prefix)-1],
            [-100] * (len(prefix)-1),
        )
        self.assertEqual(
            e.targets[len(prefix)-1],
            self.tokenizer.encode(f" {question.answer}\n")[0],
        )

    def test_batched_padding_is_ignored(self):
        samples = [encode_example(item, self.tokenizer, 160)
                   for item in self.pack.train[:2]]
        x, y = pack_batch(samples)
        self.assertEqual(x.shape, y.shape)
        for index, row in enumerate(samples):
            self.assertTrue(torch.all(y[index, len(row.targets):] == -100))

    def test_bits_per_byte_metrics_finite(self):
        model = MarsMini(MarsConfig(
            vocab_size=self.tokenizer.vocab_size, max_seq_len=160,
            embedding_dim=32, num_layers=1, num_heads=4, dropout=0.0,
        ))
        examples = [encode_example(item, self.tokenizer, 160)
                    for item in self.pack.validation[:8]]
        metrics = measure_bits_per_byte(model, examples)
        self.assertTrue(metrics)
        self.assertTrue(all(value > 0 for value in metrics.values()))

    def test_controlled_smoke_stores_both_original_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run(Namespace(
                steps=2, batch_size=2, width=32,
                layers=1, merges=8, eval_every=1,
                patience=2, learning_rate=0.0006,
                seed=42, output=tmp,
            ))
            self.assertEqual(
                set(report["models"]), {"byte-baseline", "bpe-experiment"},
            )
            root = Path(tmp)
            for name in ("byte-baseline", "bpe-experiment"):
                path = root / name / "weights.pt"
                self.assertTrue(path.is_file())
                checkpoint = torch.load(
                    path, weights_only=True, map_location="cpu"
                )
                self.assertIn("tokenizer", checkpoint)
                self.assertGreater(checkpoint["best_step"], 0)
            token_file = root / "mars-bpe-tokenizer.json"
            self.assertEqual(MarsBPETokenizer.load(token_file).vocab_size, 264)

    def test_balanced_score_requires_all_validation_tasks(self):
        with self.assertRaises(RuntimeError):
            balanced_score({"chat_paraphrase": 2.0})


if __name__ == "__main__":
    unittest.main()
