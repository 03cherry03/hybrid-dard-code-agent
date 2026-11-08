"""Integrity and oracle-isolation checks for the preparation boundary."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_data as preparation
from prepare_data import Snapshot, public_repo_task, validate_page


class PreparationTests(unittest.TestCase):
    def test_incomplete_or_truncated_viewer_pages_rejected(self):
        base = {'num_rows_total': 1, 'rows': [{'row_idx': 0, 'row': {'id': 'x'}, 'truncated_cells': []}]}
        self.assertEqual(validate_page(base, 0), ([{'id': 'x'}], 1))
        for bad in [dict(base, partial=True), dict(base, rows=[]),
                    dict(base, rows=[dict(base['rows'][0], truncated_cells=['test'])]),
                    dict(base, rows=[dict(base['rows'][0], row_idx=2)])]:
            with self.assertRaises(ValueError):
                validate_page(bad, 0)
        with self.assertRaises(ValueError):
            validate_page(base, 0, total=2)

    def test_public_repository_record_has_no_oracles_or_test_metadata(self):
        row = dict(instance_id='abc', repo='apache/druid', base_commit='a'*40,
                   problem_statement='Fix a bug', image='official/image', patch='SECRET',
                   test_patch='SECRET', eval_script='SECRET', FAIL_TO_PASS=['SECRET'])
        public = public_repo_task(row, 'swebench_multilingual')
        self.assertEqual(public['language'], 'java')
        self.assertNotIn('SECRET', json.dumps(public))
        self.assertEqual(set(public), {'dataset','instance_id','repo','base_commit','problem_statement','language'})

    def test_corrupt_cached_file_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Snapshot(Path(tmp))
            path = Path(tmp) / 'datasets/x.json'
            snapshot.store(path, b'{}', url='https://fixture')
            path.write_bytes(b'corrupt')
            with patch('prepare_data.request') as remote:
                with self.assertRaisesRegex(ValueError, 'Corrupt'):
                    snapshot.fetch('https://fixture', path)
                remote.assert_not_called()

    def test_viewer_revision_mismatch_is_not_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Snapshot(Path(tmp))
            path = Path(tmp) / 'datasets/x.json'
            with patch('prepare_data.request', return_value=(b'{}', {'x-revision':'wrong'})):
                with self.assertRaisesRegex(ValueError, 'revision mismatch'):
                    snapshot.fetch('https://fixture', path, revision='expected')
            self.assertFalse(path.exists())

    def test_repository_download_reuses_valid_pinned_data_and_preserves_legacy_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Snapshot(Path(tmp))
            row = dict(instance_id="fixture-1", repo="example/example", base_commit="a" * 40,
                       problem_statement="Fix fixture", patch="sealed gold", test_patch="sealed test")
            with patch.object(preparation, "hf_rows", return_value=[row]), \
                 contextlib.redirect_stdout(io.StringIO()):
                preparation.prepare_swe(snapshot, "swebench_verified", "SWE-bench/SWE-bench_Verified", 1)
            target = Path(tmp) / "datasets/swebench_verified"
            public_path = target / "tasks.jsonl"
            public = json.loads(public_path.read_text())
            public["unused_legacy_metadata"] = "preserve without exposing to the agent"
            public_path.write_bytes(preparation.jsonl([public]))
            manifest_path = target / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest.pop("local_preparation_status")
            manifest["unused_legacy_status"] = "preserve"
            manifest["tasks_sha256"] = preparation.sha256(public_path.read_bytes())
            manifest_path.write_bytes(preparation.json_bytes(manifest))
            for path in (public_path, manifest_path):
                snapshot.catalog["files"][str(path.relative_to(snapshot.root))] = {
                    "sha256": preparation.sha256(path.read_bytes()), "bytes": path.stat().st_size}
            snapshot.save()
            originals = {path: path.read_bytes() for path in (public_path, manifest_path, snapshot.catalog_path)}
            with patch.object(preparation, "hf_rows") as download, contextlib.redirect_stdout(io.StringIO()):
                preparation.prepare_swe(snapshot, "swebench_verified", "SWE-bench/SWE-bench_Verified", 1)
            download.assert_not_called()
            self.assertEqual({path: path.read_bytes() for path in originals}, originals)
            public_path.write_bytes(b"corrupted")
            with patch.object(preparation, "hf_rows") as download:
                with self.assertRaisesRegex(ValueError, "Corrupt repository snapshot"):
                    preparation.prepare_swe(snapshot, "swebench_verified", "SWE-bench/SWE-bench_Verified", 1)
            download.assert_not_called()


class ScopedPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.snapshot = Snapshot(self.root)
        for suite in preparation.selected_suites("all"):
            self.snapshot.catalog["datasets"][suite] = {"count": 1}
        for relative in preparation.required_files("all") - {"datasets/functions/manifest.json"}:
            self.snapshot.store(self.root / relative, (relative + " fixture").encode())
        self.entries = {suite: {"path": f"raw/{suite}.jsonl",
            "sha256": self.snapshot.catalog["files"][f"datasets/functions/raw/{suite}.jsonl"]["sha256"],
            "revision": spec[1], "evaluator_version": "0.3.1"} for suite, spec in preparation.PLUS.items()}
        self.manifest = self.root / "datasets/functions/manifest.json"
        document = {"schema_version": 1, "suites": {**self.entries,
                    "bigcodebench_instruct": {"path": "raw/bigcodebench_instruct.jsonl", "sha256": "unused"}}}
        self.snapshot.store(self.manifest, preparation.json_bytes(document))
        self.snapshot.catalog["datasets"]["bigcodebench_instruct"] = {"count": 1}
        self.snapshot.store(self.root / "datasets/functions/raw/bigcodebench_instruct.jsonl", b"library fixture")
        self.snapshot.store(self.root / "datasets/downloads/bigcodebench_0.json", b"download fixture")
        self.snapshot.save()

    def verify(self, scope="functions"):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            catalog = preparation.verify(self.root, scope)
        return catalog, json.loads(output.getvalue())

    def test_default_functions_ignore_missing_repository_and_library_files(self):
        for relative in list(self.snapshot.catalog["files"]):
            if relative.startswith("datasets/swebench_") or "bigcodebench" in relative:
                (self.root / relative).unlink()
        catalog, report = self.verify()
        self.assertEqual(set(report["datasets"]), set(preparation.FUNCTION_SUITES))
        self.assertEqual(len(report["datasets"]), 10)
        self.assertEqual(report["scope"], "functions")
        self.assertIn("bigcodebench_instruct", catalog["datasets"])
        with self.assertRaises(FileNotFoundError):
            self.verify("all")

    def test_repository_scope_does_not_read_function_files_or_shared_manifest(self):
        for relative in list(self.snapshot.catalog["files"]):
            if not preparation.selected_file(relative, "repositories"):
                (self.root / relative).unlink()
        _, report = self.verify("repositories")
        self.assertEqual(set(report["datasets"]), set(preparation.REPOSITORY_SUITES))

    def test_selected_raw_checksum_and_manifest_identity_remain_strict(self):
        raw = self.root / "datasets/functions/raw/humaneval_plus.jsonl"
        original = raw.read_bytes()
        raw.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Integrity failure"):
            self.verify()
        raw.write_bytes(original)
        document = json.loads(self.manifest.read_text())
        document["suites"]["humaneval_plus"]["sha256"] = "changed"
        self.manifest.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "Selected official manifest integrity"):
            self.verify()

    def test_excluded_shared_manifest_entry_can_change_without_invalidating_selected_data(self):
        document = json.loads(self.manifest.read_text())
        document["suites"]["bigcodebench_instruct"] = "corrupt unrelated metadata"
        self.manifest.write_text(json.dumps(document))
        _, report = self.verify("all")
        self.assertEqual(len(report["datasets"]), 12)
        self.assertNotIn("bigcodebench_instruct", report["datasets"])

    def test_missing_selected_dataset_or_file_catalog_is_rejected(self):
        self.snapshot.catalog["datasets"].pop("mbpp_plus")
        self.snapshot.save()
        with self.assertRaisesRegex(ValueError, "Incomplete functions preparation"):
            self.verify()
        self.snapshot.catalog["datasets"]["mbpp_plus"] = {"count": 1}
        self.snapshot.catalog["files"].pop("datasets/function-cache/multiple_rs_0.json")
        self.snapshot.save()
        with self.assertRaisesRegex(ValueError, "Incomplete functions file catalog"):
            self.verify()

    def test_manifest_merge_preserves_unselected_and_legacy_metadata_without_rewriting(self):
        document = json.loads(self.manifest.read_text())
        document["suites"]["humaneval_plus"]["legacy_evaluator_metadata"] = "preserve"
        self.manifest.write_text(json.dumps(document))
        before = self.manifest.read_bytes()
        original_catalog = json.loads(self.snapshot.catalog_path.read_text())
        preparation.merge_official_manifest(self.snapshot, self.entries)
        self.assertEqual(self.manifest.read_bytes(), before)
        self.assertEqual(json.loads(self.snapshot.catalog_path.read_text()), original_catalog)
        with self.assertRaisesRegex(ValueError, "snapshot changed"):
            preparation.merge_official_manifest(self.snapshot,
                {"humaneval_plus": {**self.entries["humaneval_plus"], "sha256": "changed"}})
        self.assertEqual(self.manifest.read_bytes(), before)

    def test_manifest_merge_adds_selected_suite_and_keeps_unrelated_catalog_records(self):
        document = json.loads(self.manifest.read_text())
        document["suites"].pop("mbpp_plus")
        self.manifest.write_text(json.dumps(document))
        unrelated = dict(self.snapshot.catalog["files"]["datasets/functions/raw/bigcodebench_instruct.jsonl"])
        preparation.merge_official_manifest(self.snapshot, {"mbpp_plus": self.entries["mbpp_plus"]})
        merged = json.loads(self.manifest.read_text())
        self.assertEqual(merged["suites"]["mbpp_plus"], self.entries["mbpp_plus"])
        self.assertEqual(merged["suites"]["bigcodebench_instruct"], document["suites"]["bigcodebench_instruct"])
        self.assertEqual(self.snapshot.catalog["files"]["datasets/functions/raw/bigcodebench_instruct.jsonl"], unrelated)
        self.verify()

    def test_download_dispatches_selected_scope_and_never_prepares_library_suite(self):
        for scope, functions, repositories in (("functions", 1, 0), ("repositories", 0, 2), ("all", 1, 2)):
            with self.subTest(scope=scope), patch.object(preparation, "Snapshot", return_value=self.snapshot), \
                 patch.object(preparation, "prepare_plus", return_value=self.entries) as plus, \
                 patch.object(preparation, "merge_official_manifest") as merge, \
                 patch.object(preparation, "prepare_legacy") as legacy, \
                 patch.object(preparation, "prepare_swe") as swe, \
                 patch.object(preparation, "verify") as verify:
                preparation.main(["--data-root", str(self.root), "--scope", scope, "--download"])
            self.assertEqual(plus.call_count, functions)
            self.assertEqual(legacy.call_count, functions)
            self.assertEqual(merge.call_count, functions)
            self.assertEqual(swe.call_count, repositories)
            verify.assert_called_once_with(self.root, scope)

    def test_cli_defaults_to_function_scope_and_rejects_unknown_scope(self):
        with patch.object(preparation, "verify") as verify:
            preparation.main(["--data-root", str(self.root), "--verify"])
        verify.assert_called_once_with(self.root, "functions")
        with self.assertRaisesRegex(ValueError, "Unknown preparation scope"):
            preparation.verify(self.root, "library")

    def test_quiet_verification_and_explicit_size_only_probe(self):
        raw = self.root / "datasets/functions/raw/humaneval_plus.jsonl"
        raw.write_bytes(b"x" * raw.stat().st_size)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            preparation.verify(self.root, quiet=True, verify_hashes=False)
        self.assertEqual(output.getvalue(), "")
        with self.assertRaisesRegex(ValueError, "Integrity failure"):
            preparation.verify(self.root, quiet=True)


if __name__ == '__main__':
    unittest.main()
