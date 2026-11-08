"""Probe only local fixture files; no network, package installs, or GPU allocation."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import environment_probe as readiness
import prepare_data
from environment_probe import check_safetensors_size, inspect_catalog, inspect_model


def safetensors_fixture():
    header = json.dumps({"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    return len(header).to_bytes(8, "little") + header + b"\x00" * 4


class TestEnvironmentProbe(unittest.TestCase):
    def test_catalog_checks_root_relative_paths_and_detects_modified_payload(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            datasets = root / "datasets"
            datasets.mkdir()
            payload = datasets / "fixture.jsonl"
            data = b'{"task_id":"one"}\n'
            payload.write_bytes(data)
            (datasets / "catalog.json").write_text(json.dumps({"files": {
                "datasets/fixture.jsonl": {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}}}))
            self.assertTrue(inspect_catalog(root)["complete"])
            payload.write_bytes(data.replace(b"one", b"two"))
            result = inspect_catalog(root)
            self.assertFalse(result["complete"])
            self.assertTrue(result["files"][0]["size_matches"])
            self.assertFalse(result["files"][0]["hash_matches"])

    def test_missing_hash_and_outside_root_cannot_certify_completeness(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "datasets").mkdir()
            (root / "datasets" / "empty").write_text("data")
            catalog = root / "datasets" / "catalog.json"
            catalog.write_text(json.dumps({"files": {"datasets/empty": {"bytes": 4}}}))
            self.assertFalse(inspect_catalog(root)["complete"])
            catalog.write_text(json.dumps({"files": {"../outside": {"sha256": "dummy"}}}))
            self.assertIn("outside data root", inspect_catalog(root)["errors"][0])

    def test_model_missing_shard_and_dream_code_are_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "config.json").write_text('{"model_type":"Dream"}')
            (root / "tokenizer_config.json").write_text('{}')
            (root / "tokenizer.json").write_text('{}')
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"a": "part-1.safetensors", "b": "part-2.safetensors"}}))
            (root / "part-1.safetensors").write_bytes(safetensors_fixture())
            result = inspect_model(root)
            self.assertFalse(result["structurally_complete"])
            self.assertIn("part-2.safetensors", result["missing"])
            self.assertIn("generation_utils.py", result["missing"])
            (root / "part-2.safetensors").write_bytes(safetensors_fixture())
            for name in ("modeling_dream.py", "configuration_dream.py", "generation_utils.py"):
                (root / name).write_text("structural fixture")
            result = inspect_model(root)
            self.assertTrue(result["structurally_complete"])
            self.assertEqual(result["weight_integrity"], "not_checked")

    def test_nonempty_truncated_weight_file_is_not_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "shard.safetensors"
            path.write_bytes(safetensors_fixture()[:-1])
            with self.assertRaisesRegex(ValueError, "Incomplete safetensors payload"):
                check_safetensors_size(path)

    def test_probe_only_inspects_environments_selected_by_scope(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("inference", "function", "evalplus", "swebench", "bigcodebench"):
                path = root / "envs" / name / "bin/python"
                path.parent.mkdir(parents=True)
                path.touch()
            def readonly(argv):
                if argv[0] == "nvidia-smi":
                    return {"ok": False, "error": "fixture has no GPU"}
                return {"ok": True, "stdout": '{"python":"fixture","versions":{}}'}
            for scope in readiness.ENVIRONMENTS:
                with self.subTest(scope=scope), patch.object(readiness, "run_readonly", side_effect=readonly) as run, \
                     patch.object(readiness, "installed_versions", return_value={}), \
                     patch.object(readiness, "inspect_model", return_value={}), \
                     patch.object(readiness, "inspect_catalog", return_value={}) as catalog:
                    report = readiness.probe(root, scope=scope)
                self.assertEqual(tuple(report["environments"]), readiness.ENVIRONMENTS[scope])
                self.assertEqual(report["scope"], scope)
                self.assertNotIn("bigcodebench", report["environments"])
                self.assertEqual(run.call_count, 1 + len(readiness.ENVIRONMENTS[scope]))
                catalog.assert_called_once_with(root, verify_hashes=True, scope=scope)
            with patch.object(readiness, "run_readonly", side_effect=readonly), \
                 patch.object(readiness, "installed_versions", return_value={}), \
                 patch.object(readiness, "inspect_model", return_value={}), \
                 patch.object(readiness, "inspect_catalog", return_value={}):
                self.assertEqual(readiness.probe(root)["scope"], "functions")

    def test_scoped_catalog_uses_shared_verifier_without_unselected_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "datasets").mkdir()
            (root / "datasets/catalog.json").write_text("{}")
            catalog = {"datasets": {suite: {"count": 1} for suite in prepare_data.selected_suites("all")},
                       "files": {relative: {} for relative in prepare_data.required_files("all")}}
            catalog["datasets"]["bigcodebench_instruct"] = {"count": 1}
            catalog["files"]["datasets/functions/raw/bigcodebench_instruct.jsonl"] = {}
            with patch.object(readiness, "verify", return_value=catalog) as verify:
                report = readiness.inspect_catalog(root, verify_hashes=False, scope="functions")
            verify.assert_called_once_with(root, "functions", quiet=True, verify_hashes=False)
            self.assertTrue(report["complete"])
            self.assertFalse(report["hashes_verified"])
            self.assertEqual(set(report["datasets"]), set(prepare_data.FUNCTION_SUITES))
            self.assertTrue(all("bigcodebench" not in item["path"] and "swebench" not in item["path"]
                                for item in report["files"]))
            shared = next(item for item in report["files"] if item["path"].endswith("functions/manifest.json"))
            self.assertEqual(shared["validation"], "selected_entries")

    def test_scoped_catalog_reports_selected_integrity_error(self):
        with patch.object(readiness, "verify", side_effect=ValueError("Integrity failure: selected fixture")):
            report = readiness.inspect_catalog(Path("/fixture"), scope="functions")
        self.assertFalse(report["complete"])
        self.assertEqual(report["errors"], ["Integrity failure: selected fixture"])


if __name__ == "__main__":
    unittest.main()
