"""CPU-only checks for bounded, complete and immutable candidate preparation."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from iterative_tuning import DREAM_CODER, candidate, validate_candidate
from scripts import prepare_dream_coder_candidate as prepare


class DreamCandidateTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.snapshot = self.root / "snapshots" / prepare.REVISION
        self.snapshot.mkdir(parents=True)
        (self.root / "blobs").mkdir()
        self.expected = {}
        self.add_file("config.json", json.dumps({
            "model_type": "Dream", "architectures": ["DreamModel"],
            "mask_token_id": 151666, "pad_token_id": 151643, "eos_token_id": 151643,
            "auto_map": {"AutoModel": "modeling_dream.DreamModel"}}).encode())
        self.add_file("tokenizer_config.json", b"{}")
        self.add_file("configuration_dream.py", b"# Fixture only\n")
        self.add_file("modeling_dream.py", b"class DreamModel: pass\nclass DreamSdpaAttention: pass\n")
        self.add_file("generation_utils.py", b"class DreamGenerationConfig: pass\nclass DreamGenerationMixin: pass\nlogits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)\n")
        header = json.dumps({"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
        header += b" " * (-len(header) % 8)
        self.add_file("model.safetensors", struct.pack("<Q", len(header)) + header + struct.pack("<ff", 1, 2), lfs=True)
        self.add_file("model.safetensors.index.json", json.dumps({
            "metadata": {"total_size": 8}, "weight_map": {"weight": "model.safetensors"}}).encode())

    def add_file(self, name, content, *, lfs=False):
        digest = (hashlib.sha256(content) if lfs else hashlib.sha1(
            b"blob " + str(len(content)).encode() + b"\0" + content)).hexdigest()
        blob = self.root / "blobs" / digest
        blob.write_bytes(content)
        path = self.snapshot / name
        if path.is_symlink():
            path.unlink()
        path.symlink_to(blob)
        self.expected[name] = len(content)
        return blob

    def tokenizer_result(self):
        tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **k: "<|im_start|>assistant\n")
        return tokenizer, {"status": "verified", "chat_template": {"policy": "checkpoint_native"},
                           "special_token_ids": {"<|endoftext|>": 151643, "<|im_end|>": 151645}}

    def verify(self):
        with patch.object(prepare, "load_verified_tokenizer", return_value=self.tokenizer_result()), patch.object(
                prepare, "check_generation_config", return_value={"status": "verified"}):
            return prepare.verify_snapshot(self.snapshot, self.expected, started=1, deadline=1e20)

    def test_deadline_is_capped_by_wall_budget_and_campaign(self):
        self.assertEqual(prepare.effective_deadline(100, 10000, 10), 110)
        self.assertEqual(prepare.effective_deadline(prepare.PREPARATION_CUTOFF_UNIX - 5, 1e20, 10), prepare.PREPARATION_CUTOFF_UNIX)
        for value in (float("nan"), float("inf"), 0, 21601):
            with self.assertRaises(ValueError):
                prepare.effective_deadline(100, 10000, value)
        with self.assertRaises(TimeoutError):
            prepare.effective_deadline(100, 99, 10)

    def test_download_inventory_is_revision_bound_and_complete(self):
        info = SimpleNamespace(sha=prepare.REVISION, siblings=[
            SimpleNamespace(rfilename=k, size=v) for k, v in self.expected.items()])
        self.assertEqual(prepare.select_expected_files(info), self.expected)
        info.sha = "b" * 40
        with self.assertRaisesRegex(ValueError, "different revision"):
            prepare.select_expected_files(info)
        info.sha = prepare.REVISION
        info.siblings.pop()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            prepare.select_expected_files(info)

    def test_missing_download_file_is_rejected(self):
        (self.snapshot / "generation_utils.py").unlink()
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.verify()

    def test_complete_real_shards_have_receipt_but_gpu_remains_pending(self):
        report = self.verify()
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["gpu_compatibility"], "pending")
        self.assertEqual(len(report["files"]), len(self.expected))
        self.assertEqual(report["shards"][0]["tensor_count"], 1)
        self.assertEqual(report["termination"]["terminal_token_ids"], [151643])

    def test_corrupted_weights_are_rejected_before_tokenizer_import(self):
        path = self.snapshot / "model.safetensors"
        content = path.read_bytes()
        path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
        with patch.object(prepare, "load_verified_tokenizer", side_effect=AssertionError("premature import")):
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                prepare.verify_snapshot(self.snapshot, self.expected, started=1, deadline=1e20)

    def test_unsupported_prediction_coordinates_are_rejected(self):
        self.add_file("generation_utils.py", b"class DreamGenerationConfig: pass\nclass DreamGenerationMixin: pass\n")
        with self.assertRaisesRegex(ValueError, "shifted prediction"):
            self.verify()

    def test_missing_native_template_blocks_candidate(self):
        with patch.object(prepare, "load_verified_tokenizer", side_effect=ValueError("Tokenizer has no native chat template")):
            with self.assertRaisesRegex(ValueError, "native chat template"):
                prepare.verify_snapshot(self.snapshot, self.expected, started=1, deadline=1e20)

    def test_non_native_template_is_rejected(self):
        tokenizer, receipt = self.tokenizer_result()
        receipt["chat_template"]["policy"] = "explicit_sha256_bound_override"
        with patch.object(prepare, "load_verified_tokenizer", return_value=(tokenizer, receipt)):
            with self.assertRaisesRegex(ValueError, "checkpoint-native"):
                prepare.verify_snapshot(self.snapshot, self.expected, started=1, deadline=1e20)

    def test_expired_verification_never_returns_receipt(self):
        with patch.object(prepare, "load_verified_tokenizer", return_value=self.tokenizer_result()), patch.object(
                prepare, "check_generation_config", return_value={"status": "verified"}):
            with self.assertRaises(TimeoutError):
                prepare.verify_snapshot(self.snapshot, self.expected, started=0, deadline=1)

    def test_only_pinned_candidate_is_admitted(self):
        config = candidate(DREAM_CODER)
        validate_candidate(config)
        config["revision"] = "a" * 40
        with self.assertRaisesRegex(ValueError, "Unpinned"):
            validate_candidate(config)


if __name__ == "__main__":
    unittest.main()
