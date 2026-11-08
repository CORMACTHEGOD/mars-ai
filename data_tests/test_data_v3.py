"""Standard-library-only Mars data V3 tests: leakage, reproducibility, scoring."""
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from mars_mini.dataset_v3 import (
    DataPack, Example, UNSEEN_FACTS, build_pack,
    manifest, normalize_prompt, verify_pack, write_pack,
)
from mars_mini.score_v3 import (
    normalize_answer, prompt_lookup_baseline,
    read_predictions, score,
)


class DataV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack = build_pack(seed=42)

    def test_reproducible_generation_and_manifest(self):
        self.assertEqual(self.pack, build_pack(seed=42))
        self.assertEqual(manifest(self.pack), manifest(build_pack(seed=42)))

    def test_original_data_pack_has_nontrivial_splits(self):
        self.assertGreater(len(self.pack.train), 600)
        self.assertGreater(len(self.pack.validation), 50)
        self.assertGreater(len(self.pack.test), 30)
        self.assertIn("sha256", manifest(self.pack)["splits"]["test"])

    def test_no_duplicate_ids_or_normalized_prompts(self):
        examples = self.pack.train + self.pack.validation + self.pack.test
        self.assertEqual(len({e.id for e in examples}), len(examples))
        self.assertEqual(len({normalize_prompt(e.prompt) for e in examples}), len(examples))

    def test_math_commutative_families_do_not_leak(self):
        for prefix in ("addition:", "word:"):
            groups = [
                {e.family for e in items if e.family.startswith(prefix)}
                for items in (self.pack.train, self.pack.validation, self.pack.test)
            ]
            self.assertFalse(groups[0] & groups[1])
            self.assertFalse(groups[0] & groups[2])
            self.assertFalse(groups[1] & groups[2])
        # Critically, '3 + 8' may not be train while '8 + 3' is test.
        for row in self.pack.test:
            if row.task == "unseen_addition_pair":
                a, b = row.family.removeprefix("addition:").split(":")
                self.assertLessEqual(int(a), int(b))

    def test_heldout_facts_absent_from_training(self):
        train_families = {e.family for e in self.pack.train}
        heldout_families = {e.family for e in self.pack.test if e.task == "unseen_fact"}
        self.assertEqual(heldout_families, {f"definition:{name}" for name in UNSEEN_FACTS})
        self.assertTrue(heldout_families.isdisjoint(train_families))

    def test_all_examples_fit_original_model_context(self):
        verify_pack(self.pack, max_context_bytes=160)
        with self.assertRaises(ValueError):
            verify_pack(self.pack, max_context_bytes=20)

    def test_malformed_split_with_reversed_math_leak_is_rejected(self):
        original = next(e for e in self.pack.train if e.family.startswith("addition:"))
        slipped = Example(
            "leaked-example", "Do this new-looking math expression.",
            original.answer, "unseen_addition_pair", original.family,
            "computed_arithmetic",
        )
        broken = DataPack(self.pack.train, self.pack.validation, self.pack.test + (slipped,))
        with self.assertRaisesRegex(ValueError, "family leakage"):
            verify_pack(broken)

    def test_repeated_prompt_detected_even_when_punctuation_changes(self):
        original = self.pack.train[0]
        duplicate = replace(
            original, id="clone", prompt=original.prompt.upper().rstrip("?!") + "!"
        )
        broken = DataPack(self.pack.train, self.pack.validation, self.pack.test + (duplicate,))
        with self.assertRaisesRegex(ValueError, "prompt"):
            verify_pack(broken)

    def test_baseline_exposes_memorization(self):
        predictions = prompt_lookup_baseline(self.pack)
        report = score(self.pack, predictions)
        train_scores = report["splits"]["train"]["normalized_exact_match"]
        self.assertTrue(all(item["correct"] == item["total"] for item in train_scores.values()))
        # An exact lookup cannot answer differently phrased or heldout questions.
        for split in ("validation", "test"):
            scores = report["splits"][split]["normalized_exact_match"]
            self.assertTrue(all(item["correct"] == 0 for item in scores.values()))

    def test_scoring_accepts_missing_and_rejects_unknown_predictions(self):
        first = self.pack.test[0]
        partial = score(self.pack, {first.id: first.answer})
        self.assertEqual(partial["splits"]["test"]["predicted"], 1)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            score(self.pack, {"made-up-id": "hello"})

    def test_normalized_scoring_not_semantic_claim(self):
        self.assertEqual(normalize_answer("  Hello!  "), normalize_answer("hello"))
        self.assertNotEqual(normalize_answer("Earth"), normalize_answer("Mars"))

    def test_export_and_prediction_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            details = write_pack(directory, self.pack)
            self.assertEqual(
                sum(1 for _ in (directory / "train.jsonl").open(encoding="utf-8")),
                details["splits"]["train"]["rows"],
            )
            self.assertEqual(
                json.loads((directory / "manifest.json").read_text())["dataset"],
                "mars-data-v3",
            )
            predictions = directory / "predictions.jsonl"
            predictions.write_text('{"id":"a","response":"one"}\n{"id":"a","response":"two"}\n')
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_predictions(predictions)
            predictions.write_text('{"id":"a","response":"one"}\n')
            self.assertEqual(read_predictions(predictions), {"a": "one"})


if __name__ == "__main__":
    unittest.main()
