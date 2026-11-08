"""CPU tests for denoising alignment, padding, unbiased weighting and checkpoints."""
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

from diffusion_training import (atomic_json, canonical_sha256, capture_rng,
    collate_examples, corrupt_completion, encode_training_example, file_sha256,
    load_training_examples, native_attention_mask, resolve_resume,
    restore_rng, shifted_masked_diffusion_loss,
    verify_checkpoint, write_checkpoint_manifest, write_language_balanced_jsonl)


class TinyTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        text = "".join("<" + m["role"] + ">" + m["content"] + "</>" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def encode(self, text, *, add_special_tokens):
        return [ord(c) for c in text]


class DiffusionObjectiveTests(unittest.TestCase):
    def test_shifted_logits_predict_masked_next_token_only(self):
        ids = torch.tensor([[0, 1, 2, 3]])
        targets = torch.tensor([[False, False, True, True]])
        masked = torch.tensor([[False, False, True, False]])
        logits = torch.zeros(1, 4, 4, requires_grad=True)
        with torch.no_grad():
            logits[0, 1, 2] = 8
            logits[0, 2, 2] = -8
        loss = shifted_masked_diffusion_loss(logits, ids, targets, masked, torch.tensor([0.5]))
        expected = F.cross_entropy(logits[0, 1:2], torch.tensor([2]))
        self.assertAlmostEqual(float(loss), float(expected), places=6)
        loss.backward()
        self.assertGreater(float(logits.grad[0, 1].abs().sum()), 0)
        self.assertEqual(float(logits.grad[0, [0, 2, 3]].abs().sum()), 0)

    def test_loss_is_sample_mean_with_inverse_probability_and_all_target_denominator(self):
        ids = torch.tensor([[0, 1, 2, 0], [0, 1, 2, 3]])
        target = torch.tensor([[False, True, True, False], [False, True, True, True]])
        masked = torch.tensor([[False, True, False, False], [False, True, True, False]])
        logits = torch.zeros(2, 4, 4)
        loss = shifted_masked_diffusion_loss(logits, ids, target, masked, torch.tensor([.5, .25]))
        expected = torch.log(torch.tensor(4.)) * (1/(.5*2) + 2/(.25*3)) / 2
        self.assertAlmostEqual(float(loss), float(expected), places=6)

    def test_inverse_probability_expectation_over_all_bernoulli_masks(self):
        ids = torch.tensor([[0, 1, 2]])
        target = torch.tensor([[False, True, True]])
        logits = torch.tensor([[[0., 2., 1.], [0., 1., 3.], [0., 0., 0.]]])
        probability = .25
        expectation = 0.
        for left in (False, True):
            for right in (False, True):
                n = int(left) + int(right)
                weight = probability**n * (1-probability)**(2-n)
                expectation += weight * shifted_masked_diffusion_loss(logits, ids, target,
                    torch.tensor([[False, left, right]]), torch.tensor([probability]))
        expected = F.cross_entropy(logits[0, :2], ids[0, 1:])
        self.assertAlmostEqual(float(expectation), float(expected), places=6)

    def test_zero_masks_produces_connected_zero_instead_of_biased_forced_mask(self):
        logits = torch.randn(1, 3, 4, requires_grad=True)
        target = torch.tensor([[False, True, True]])
        loss = shifted_masked_diffusion_loss(logits, torch.tensor([[0, 1, 2]]), target,
                                             torch.zeros_like(target), torch.tensor([.2]))
        loss.backward()
        self.assertEqual(float(loss), 0)
        self.assertEqual(float(logits.grad.abs().sum()), 0)

    def test_corruption_preserves_prompt_and_batch_padding(self):
        ids = torch.tensor([[1, 2, 3, 4, 9]])
        target = torch.tensor([[False, False, True, True, False]])
        noisy, mask, probs = corrupt_completion(ids, target, mask_token_id=8, min_mask_prob=1)
        self.assertEqual(noisy.tolist(), [[1, 2, 8, 8, 9]])
        self.assertTrue(torch.equal(mask, target))
        self.assertEqual(probs.tolist(), [1])

    def test_invalid_mask_cannot_supervise_prompt(self):
        with self.assertRaisesRegex(ValueError, "Only completion"):
            shifted_masked_diffusion_loss(torch.zeros(1, 3, 4), torch.tensor([[0, 1, 2]]),
                torch.tensor([[False, False, True]]), torch.tensor([[False, True, False]]), torch.tensor([.5]))


class DataAndCheckpointTests(unittest.TestCase):
    def row(self):
        return {"example_id": "example", "source_id": "source", "language": "py",
                "messages": [{"role": "user", "content": "Write f"}], "completion": "def f(): return 7"}

    def test_target_includes_native_terminal_once_and_padding_is_excluded(self):
        example = encode_training_example(self.row(), TinyTokenizer(), max_length=300, terminal_token_id=999)
        self.assertEqual(example["input_ids"][-1], 999)
        self.assertTrue(example["target_mask"][-1])
        short = {**example, "input_ids": example["input_ids"][:-5], "target_mask": example["target_mask"][:-5]}
        batch = collate_examples([example, short], 999)
        self.assertFalse(bool(batch["target_mask"][1, -5:].any()))
        attention = native_attention_mask(batch["valid_mask"])
        self.assertTrue(bool(attention[0, 0, 0, -1]))  # Future visible: not causal.
        self.assertFalse(bool(attention[1, 0, :, -1].any()))

    def test_length_limit_drops_whole_example_instead_of_changing_target(self):
        self.assertIsNone(encode_training_example(self.row(), TinyTokenizer(), max_length=8, terminal_token_id=999))

    def test_training_completion_cannot_inject_special_control_tokens(self):
        row = self.row(); row["completion"] += "<|dlm_pad|>"
        with self.assertRaisesRegex(ValueError, "control tokens"):
            encode_training_example(row, TinyTokenizer(), max_length=300, terminal_token_id=999)

    def test_language_balancing_is_grouped_cyclic_and_byte_stable(self):
        import hashlib
        import json
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "train.jsonl"
            rows = []
            for language, count in (("py", 2), ("js", 1), ("cpp", 3), ("rs", 1)):
                for index in range(count):
                    rows.append({"example_id": f"{language}-{index}", "source_id": f"{language}-{index}",
                                 "language": language, "messages": [{"role": "user", "content": "f"}],
                                 "completion": "return 1"})
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            destination = Path(temporary) / "balanced.jsonl"
            receipt = write_language_balanced_jsonl(source, destination)
            self.assertEqual(receipt["balanced_rows"], 12)
            balanced = [json.loads(line) for line in destination.read_text().splitlines()]
            self.assertEqual([row["language"] for row in balanced],
                             ["py"] * 3 + ["js"] * 3 + ["cpp"] * 3 + ["rs"] * 3)
            self.assertEqual(hashlib.sha256(destination.read_bytes()).hexdigest(),
                             receipt["balanced_sha256"])

    def test_balanced_duplicate_rows_are_collapsed_and_conflicts_rejected(self):
        import json
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "balanced.jsonl"
            row = self.row()
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            examples, receipt = load_training_examples(path, TinyTokenizer(), max_length=300,
                                                        terminal_token_id=999)
            self.assertEqual(len(examples), 1)
            self.assertEqual(receipt["input_examples"], 1)
            self.assertEqual(receipt["duplicate_rows_collapsed"], 1)
            changed = dict(row, completion="return 8")
            path.write_text(json.dumps(row) + "\n" + json.dumps(changed) + "\n")
            with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
                load_training_examples(path, TinyTokenizer(), max_length=300, terminal_token_id=999)

    def test_checkpoint_detects_corruption_and_latest_pointer_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); checkpoint = root / "checkpoints/step-00000001"; checkpoint.mkdir(parents=True)
            (checkpoint / "adapter_model.safetensors").write_bytes(b"weight fixture")
            write_checkpoint_manifest(checkpoint, {"training_fingerprint": "abc"})
            self.assertTrue(verify_checkpoint(checkpoint, expected_fingerprint="abc"))
            atomic_json(root / "latest.json", {"checkpoint": "checkpoints/step-00000001",
                "manifest_sha256": file_sha256(checkpoint / "checkpoint_manifest.json")})
            self.assertEqual(resolve_resume(root, "latest"), checkpoint)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                verify_checkpoint(checkpoint, expected_fingerprint="different")
            (checkpoint / "adapter_model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                verify_checkpoint(checkpoint)

    def test_rng_resume_reproduces_next_mask_draw_and_python_rng(self):
        random.seed(123); torch.manual_seed(456)
        state = capture_rng(torch.device("cpu"))
        expected = torch.rand(10); expected_python = random.random()
        restore_rng(state, torch.device("cpu"))
        self.assertTrue(torch.equal(expected, torch.rand(10)))
        self.assertEqual(expected_python, random.random())

    def test_fingerprint_ignores_object_key_order_not_hyperparameter_changes(self):
        self.assertEqual(canonical_sha256({"a": 1, "b": 2}), canonical_sha256({"b": 2, "a": 1}))
        self.assertNotEqual(canonical_sha256({"lr": 1e-5}), canonical_sha256({"lr": 2e-5}))

    def test_checkpoint_restores_optimizer_scheduler_rng_and_next_update(self):
        from scripts.train_diffusion_lora import save_checkpoint
        class Adapter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1., 2.]))
            def save_pretrained(self, path, safe_serialization):
                torch.save(self.state_dict(), Path(path) / "adapter_model.safetensors")
                atomic_json(Path(path) / "adapter_config.json", {"fixture": True})
        class Tokenizer:
            def save_pretrained(self, path):
                atomic_json(Path(path) / "tokenizer_config.json", {"fixture": True})
        def make(model):
            opt = torch.optim.AdamW(model.parameters(), lr=.01)
            return opt, torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=.9)
        def update(model, optimizer, scheduler):
            optimizer.zero_grad()
            loss = (model.weight * torch.rand(2)).sum()
            loss.backward(); optimizer.step(); scheduler.step()
        with tempfile.TemporaryDirectory() as temporary:
            model = Adapter(); optimizer, scheduler = make(model)
            torch.manual_seed(823)
            update(model, optimizer, scheduler)
            context = SimpleNamespace(rank=0, world_size=1, device=torch.device("cpu"), barrier=lambda: None)
            args = SimpleNamespace(output=Path(temporary), base_model="fixture", revision="pinned")
            counters = {"global_step": 1, "epoch": 0, "next_batch": 1}
            checkpoint = save_checkpoint(args, context, model, Tokenizer(), optimizer, scheduler, counters, {"data": "hash"})
            update(model, optimizer, scheduler)
            expected = model.weight.detach().clone()
            expected_lr = scheduler.get_last_lr()
            resumed = Adapter(); resumed_opt, resumed_scheduler = make(resumed)
            resumed.load_state_dict(torch.load(checkpoint / "adapter_model.safetensors", weights_only=True))
            saved = torch.load(checkpoint / "training_state.pt", weights_only=True)
            resumed_opt.load_state_dict(saved["optimizer"])
            resumed_scheduler.load_state_dict(saved["scheduler"])
            restore_rng(torch.load(checkpoint / "rng-rank-0000.pt", weights_only=True), torch.device("cpu"))
            update(resumed, resumed_opt, resumed_scheduler)
            self.assertTrue(torch.equal(expected, resumed.weight.detach()))
            self.assertEqual(expected_lr, resumed_scheduler.get_last_lr())
            self.assertEqual(saved["counters"], counters)
            self.assertEqual(resolve_resume(args.output, "latest"), checkpoint)


if __name__ == "__main__":
    unittest.main()
