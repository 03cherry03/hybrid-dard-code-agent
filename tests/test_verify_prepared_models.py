"""Integrity checks use tiny real content-addressed checkpoints, without GPU access."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from scripts import verify_prepared_models as verify


class PreparedModelTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.revision = "a" * 40
        self.snapshot = self.root / "cache/snapshots" / self.revision
        self.snapshot.mkdir(parents=True)
        (self.root / "cache/blobs").mkdir()
        (self.root / "models").mkdir()
        self.alias = self.root / "models/model"
        self.alias.symlink_to(self.snapshot)
        self.enterContext(patch.object(verify, "MODELS", {"fixture/model": self.revision}))
        self.manifest = {"fixture/model": {"revision": self.revision,
                         "snapshot": str(self.snapshot), "alias": str(self.alias)}}
        (self.root / "models/manifest.json").write_text(json.dumps(self.manifest))
        for name in ("config.json", "tokenizer_config.json", "vocab.json"):
            self.add_file(name, b"{}")
        self.add_file("merges.txt", b"#version: 0.2\n")
        header = json.dumps({"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
        header += b" " * (-len(header) % 8)
        self.add_file("model.safetensors", struct.pack("<Q", len(header)) + header + struct.pack("<ff", 1, 2), lfs=True)
        self.index = {"metadata": {"total_size": 8}, "weight_map": {"weight": "model.safetensors"}}
        self.add_file("model.safetensors.index.json", json.dumps(self.index).encode())
        self.receipt = self.root / "receipt.json"

    def add_file(self, name, content, *, lfs=False):
        digest = (hashlib.sha256(content) if lfs else
                  hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content)).hexdigest()
        target = self.root / "cache/blobs" / digest
        target.write_bytes(content)
        (self.snapshot / name).symlink_to(target)
        return target

    def make_receipt(self):
        with contextlib.redirect_stdout(io.StringIO()):
            verify.verify_models(self.root, self.receipt)
        return hashlib.sha256(self.receipt.read_bytes()).hexdigest()

    def test_full_verification_supports_custom_tokenizer_and_fast_receipt(self):
        digest = self.make_receipt()
        with patch.object(verify, "_hash_file", side_effect=AssertionError("Must not reread payloads")):
            result = verify.verify_receipt(self.root, self.receipt, digest)
        self.assertEqual((result["status"], result["files"], result["weight_shards"]), ("verified", 6, 1))
        rows = json.loads(self.receipt.read_text())["files"]
        self.assertEqual({row["hash_algorithm"] for row in rows}, {"sha256", "git_blob_sha1"})

    def test_corrupt_payload_rejected_by_content_hash(self):
        path = self.snapshot / "model.safetensors"
        content = path.read_bytes()
        path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
        with self.assertRaisesRegex(ValueError, "content hash mismatch"):
            verify._hash_file(path)

    def test_changed_file_rejected_without_rehash(self):
        digest = self.make_receipt()
        (self.snapshot / "merges.txt").write_bytes(b"#version: 0.3\n")
        with self.assertRaisesRegex(ValueError, "changed after integrity"):
            verify.verify_receipt(self.root, self.receipt, digest)

    def test_edited_receipt_rejected(self):
        digest = self.make_receipt()
        self.receipt.write_text(self.receipt.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "receipt changed"):
            verify.verify_receipt(self.root, self.receipt, digest)

    def test_receipt_coverage_must_include_every_file(self):
        self.make_receipt()
        report = json.loads(self.receipt.read_text())
        report["files"].pop()
        self.receipt.write_text(json.dumps(report))
        digest = hashlib.sha256(self.receipt.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "incomplete file coverage"):
            verify.verify_receipt(self.root, self.receipt, digest)

    def test_wrong_revision_rejected(self):
        self.manifest["fixture/model"]["revision"] = "b" * 40
        (self.root / "models/manifest.json").write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            verify.manifest_identity(self.root)

    def test_index_key_mismatch_rejected(self):
        self.index["weight_map"] = {"wrong": "model.safetensors"}
        with self.assertRaisesRegex(ValueError, "keys disagree with index"):
            verify._check_shards(self.snapshot, self.index)

    def test_index_byte_count_mismatch_rejected(self):
        self.index["metadata"]["total_size"] = 9
        with self.assertRaisesRegex(ValueError, "Total weight bytes"):
            verify._check_shards(self.snapshot, self.index)


if __name__ == "__main__":
    unittest.main()
