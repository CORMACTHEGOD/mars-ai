"""Fast tests for the original Mars model (no training download required)."""
import tempfile
import unittest
from pathlib import Path

import torch

from mars_mini.corpus import build_corpus
from mars_mini.generate import complete, load_checkpoint
from mars_mini.model import MarsConfig, MarsMini, decode, encode
from mars_mini.train_dialogue import create_splits, example_tensors, make_examples


class MarsMiniTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        cls.cfg = MarsConfig(max_seq_len=24, embedding_dim=32, num_heads=4, num_layers=1, dropout=0.0)

    def test_byte_tokenizer(self):
        self.assertEqual(decode(encode("Hello from Mars! 🚀")), "Hello from Mars! 🚀")
        self.assertTrue(all(0 <= token < 256 for token in encode("Mars")))

    def test_original_corpus_is_long_enough(self):
        text = build_corpus()
        self.assertIn("Mars-1 Mini", text)
        self.assertGreater(len(encode(text)), 15_000)

    def test_answer_only_training_masks_question_text(self):
        x, targets = example_tensors("hello", "Hello!", 48)
        prefix = encode("User: hello\nMars:")
        self.assertEqual(tuple(x.shape), (48,))
        self.assertTrue(torch.all(targets[:len(prefix)-1] == -100))
        self.assertEqual(targets[len(prefix)-1].item(), ord(" "))
        self.assertEqual(targets[len(prefix)].item(), ord("H"))
        self.assertTrue(torch.all(targets[len(encode("User: hello\nMars: Hello!\n"))-1:] == -100))

    def test_training_dialogues_have_common_prompts(self):
        self.assertIn(("hello", "Hello! I am Mars, an experimental AI."), make_examples())
        training, validation = create_splits(128, 42)
        self.assertEqual(training[0].shape[1], 128)
        self.assertGreater(training[0].shape[0], validation[0].shape[0])
        self.assertTrue(torch.all((training[1] == -100) | ((training[1] >= 0) & (training[1] < 256))))

    def test_forward_and_backward(self):
        model = MarsMini(self.cfg)
        x = torch.randint(0, 256, (2, 12), dtype=torch.long)
        y = torch.randint(0, 256, (2, 12), dtype=torch.long)
        logits, loss = model(x, y)
        self.assertEqual(tuple(logits.shape), (2, 12, 256))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_causal_attention_cannot_peek_ahead(self):
        model = MarsMini(self.cfg)
        model.eval()
        first = torch.randint(0, 256, (1, 10), dtype=torch.long)
        second = first.clone()
        second[:, -1] = (second[:, -1] + 1) % 256
        with torch.no_grad():
            earlier_a, _ = model(first)
            earlier_b, _ = model(second)
        self.assertTrue(torch.allclose(earlier_a[:, :-1], earlier_b[:, :-1], atol=1e-6))

    def test_save_reload_and_generate(self):
        model = MarsMini(self.cfg)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "mini.pt"
            torch.save(
                {"model_config": vars(self.cfg), "model_state": model.state_dict()},
                checkpoint,
            )
            restored = load_checkpoint(checkpoint)
            result = complete(restored, "hello", max_new_tokens=4)
            self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
