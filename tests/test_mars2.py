"""Mars-2 model research checks, especially evaluation integrity."""
from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from mars_mini.mars2_data import make_mars2_splits
from mars_mini.train_dialogue import example_tensors
from mars_mini.train_mars2 import run, tensorize


class Mars2Tests(unittest.TestCase):
    def test_no_exact_train_validation_or_test_prompt_overlap(self):
        data = make_mars2_splits(seed=42)
        groups = [{question.lower() for question, _ in group}
                  for group in (data.train, data.validation, data.novel_test)]
        self.assertFalse(groups[0] & groups[1])
        self.assertFalse(groups[0] & groups[2])
        self.assertFalse(groups[1] & groups[2])
        self.assertGreater(len(data.train), 500)
        self.assertGreater(len(data.validation), 100)
        self.assertGreater(len(data.novel_test), 20)

    def test_novel_arithmetic_pair_is_unseen(self):
        data = make_mars2_splits()
        self.assertIn(("What is 0 plus 0?", "0."), data.novel_test)
        self.assertNotIn(("What is 0 plus 0?", "0."), data.train)

    def test_training_inputs_fit_context_and_answer_loss_is_masked(self):
        data = make_mars2_splits()
        for examples in (data.train, data.validation, data.novel_test):
            x, y = tensorize(examples, 160)
            self.assertEqual(x.shape, y.shape)
            self.assertEqual(x.shape[1], 160)
            self.assertTrue(torch.all(torch.any(y != -100, dim=1)))

        x, y = example_tensors("What is Python?", "Python is a language.", 160)
        self.assertTrue(torch.all(y[:len("User: What is Python?\nMars:") - 1] == -100))

    def test_smoke_training_saves_checkpoint_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = run(Namespace(
                steps=2, batch_size=2, learning_rate=0.00055,
                eval_every=1, patience=2, seed=42, output=tmp,
            ))
            self.assertTrue(Path(checkpoint).is_file())
            self.assertTrue((Path(tmp) / "metrics.json").is_file())
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertGreater(saved["best_step"], 0)
            self.assertTrue(torch.isfinite(torch.tensor(saved["best_paraphrase_val_loss"])))


if __name__ == "__main__":
    unittest.main()
