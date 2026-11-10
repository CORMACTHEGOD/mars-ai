"""Fast structural tests for Mars Data V4. No PyTorch dependency."""
import unittest
from collections import Counter

from mars_mini.dataset_v4 import build_pack, summary, verify_pack
from mars_mini.dataset_v3 import normalize_prompt


class DataV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack = build_pack(seed=42)

    def test_much_larger_than_v3(self):
        self.assertGreater(len(self.pack.train), 20_000)
        self.assertGreater(len(self.pack.validation), 1_000)
        self.assertGreater(len(self.pack.test), 1_000)

    def test_no_duplicate_prompts_or_family_leakage(self):
        verify_pack(self.pack)
        seen = {}
        for split in ("train", "validation", "test"):
            for row in getattr(self.pack, split):
                key = normalize_prompt(row.prompt)
                self.assertNotIn(key, seen)
                seen[key] = split
        train_families = {row.family for row in self.pack.train}
        valid_families = {row.family for row in self.pack.validation}
        test_families = {row.family for row in self.pack.test}
        self.assertTrue(train_families.isdisjoint(valid_families))
        self.assertTrue(train_families.isdisjoint(test_families))
        self.assertTrue(valid_families.isdisjoint(test_families))

    def test_all_task_families_represented(self):
        for name in ("train", "validation", "test"):
            task_counts = Counter(row.task for row in getattr(self.pack, name))
            for expected in ("addition", "subtraction", "comparison",
                             "multiplication", "parity", "reading",
                             "word_rules", "conversation", "definition"):
                self.assertIn(expected, task_counts, name)

    def test_arithmetic_answers_are_derived_not_hardcoded(self):
        for row in self.pack.train[:5000]:
            if row.task == "addition":
                import re
                digits = [int(d) for d in re.findall(r"\d+", row.prompt)]
                self.assertEqual(int(row.answer.rstrip(".")), sum(digits))

    def test_manifest_is_honest(self):
        data = summary(self.pack)
        self.assertIn("synthetic", data["type"])
        self.assertEqual(data["splits"]["train"]["rows"], len(self.pack.train))


if __name__ == "__main__":
    unittest.main()
