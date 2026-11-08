"""Native loader imports pinned official leaf modules without optional services."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import local_repository_preparation as preparation
import local_swe_worker as worker


class OfficialLeafLoaderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.original_modules = {key: value for key, value in sys.modules.items()
                                 if key == "swebench" or key.startswith("swebench.")}
        self.remove_modules()
        self.addCleanup(self.restore_modules)
        sources = {
            "swebench/__init__.py": 'PACKAGE_SENTINEL = "pinned package"\n',
            "swebench/types.py": 'class TestSpec:\n    pass\n',
            "swebench/harness/__init__.py":
                'raise AssertionError("Optional execution service initializer must not run")\n',
            "swebench/harness/constants/__init__.py": 'TEST_SENTINEL = "unchanged constant"\n',
            "swebench/harness/log_parsers/__init__.py":
                'def parse_fixture(text):\n    return {"fixture": text}\n',
            "swebench/harness/grading.py": (
                'from swebench.types import TestSpec\n'
                'from swebench.harness.constants import TEST_SENTINEL\n'
                'from swebench.harness.log_parsers import parse_fixture\n'
                'def grade_fixture(text):\n'
                '    return TEST_SENTINEL, parse_fixture(text)\n'),
        }
        for relative, text in sources.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        self.enterContext(patch.object(sys, "path", [str(self.root), *sys.path]))
        importlib.invalidate_caches()

    @staticmethod
    def remove_modules():
        for key in list(sys.modules):
            if key == "swebench" or key.startswith("swebench."):
                sys.modules.pop(key)

    def restore_modules(self):
        self.remove_modules()
        sys.modules.update(self.original_modules)
        importlib.invalidate_caches()

    def distribution(self, *, version=preparation.HARNESS_VERSION,
                     commit=preparation.HARNESS_COMMIT):
        return SimpleNamespace(version=version,
            read_text=lambda name: json.dumps({"vcs_info": {"commit_id": commit}})
                                  if name == "direct_url.json" else None,
            locate_file=lambda name: self.root / name)

    def source_hashes(self):
        return {str(path.relative_to(self.root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in self.root.rglob("*.py")}

    def test_loader_skips_initializer_but_loads_original_leaf_files(self):
        # The ordinary dotted import demonstrably executes the unavailable
        # optional-service initializer; the native loader must avoid only it.
        with self.assertRaisesRegex(AssertionError, "initializer must not run"):
            importlib.import_module("swebench.harness.grading")
        self.remove_modules()
        before = self.source_hashes()
        with patch.object(worker.importlib.metadata, "distribution", return_value=self.distribution()):
            api = worker.load_official_api()
            again = worker.load_official_api()
        self.assertEqual(api["identity"], {"version": preparation.HARNESS_VERSION,
                                           "commit": preparation.HARNESS_COMMIT})
        self.assertEqual(api["grading"].grade_fixture("PASSED"),
                         ("unchanged constant", {"fixture": "PASSED"}))
        self.assertIs(api["TestSpec"], sys.modules["swebench.types"].TestSpec)
        self.assertIs(again["grading"], api["grading"])
        self.assertEqual(Path(api["grading"].__file__), self.root / "swebench/harness/grading.py")
        self.assertEqual(sys.modules["swebench.harness"].__path__, [str(self.root / "swebench/harness")])
        self.assertEqual(before, self.source_hashes())

    def test_wrong_distribution_version_or_commit_is_rejected_before_import(self):
        for distribution in (self.distribution(version="wrong"), self.distribution(commit="0" * 40),
                             self.distribution(commit=None)):
            with self.subTest(identity=distribution.read_text("direct_url.json")), \
                 patch.object(worker.importlib.metadata, "distribution", return_value=distribution):
                with self.assertRaisesRegex(ValueError, "pinned official SWE parser revision"):
                    worker.load_official_api()
            self.assertNotIn("swebench", sys.modules)
            self.assertNotIn("swebench.harness.grading", sys.modules)

    def test_missing_pinned_leaf_module_is_rejected_without_initializer(self):
        (self.root / "swebench/harness/grading.py").unlink()
        with patch.object(worker.importlib.metadata, "distribution", return_value=self.distribution()):
            with self.assertRaisesRegex(ValueError, "grading module is unavailable"):
                worker.load_official_api()
        self.assertNotIn("swebench.harness", sys.modules)


class NativeCandidateOutcomeTests(unittest.TestCase):
    def test_candidate_collection_error_is_unresolved_unless_environment_fault_is_confirmed(self):
        row = {"instance_id": "fixture-1", "repo": "fixture/repo", "version": "1",
               "FAIL_TO_PASS": ["test_issue"], "PASS_TO_PASS": ["test_existing"],
               "log_parser": "fixture", "eval_type": "pass_and_fail"}
        for infrastructure, expected in ((False, False), (True, None)):
            official_report = {"resolved": False, "infra_failure": infrastructure,
                               "patch_successfully_applied": False}
            grading = SimpleNamespace(
                get_logs_eval=lambda spec, path: ({}, False),
                get_eval_report=lambda *args, **kwargs: {row["instance_id"]: official_report},
                _resolve_case=lambda case, statuses: None)
            api = {"grading": grading, "TestSpec": lambda **fields: SimpleNamespace(**fields)}
            with self.subTest(infrastructure=infrastructure), \
                 patch.object(worker, "load_official_api", return_value=api):
                result = worker.parse_official_report(row, Path("syntax-error.log"), "candidate patch")
            self.assertIs(result["resolved"], expected)
            self.assertFalse(result["parser_output_valid"])
            self.assertEqual(result["observed_expected_tests"], {"test_issue": None, "test_existing": None})

    def test_worker_abnormal_exit_cannot_publish_success_and_can_be_retried(self):
        # Reuse the paired orchestration fixture; every generation and command
        # is simulated, while manifests and submission/report checks are real.
        from tests import test_repository_runner as fixtures
        import repository_runner as runner
        fixture = fixtures.RunnerTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        fixture.run_mock()
        successful_grader = fixture.fake_grader()
        def abnormal_grader(argv, **kwargs):
            result = successful_grader(argv, **kwargs)
            if "-c" not in argv:
                result.returncode = 7  # The report was written before cleanup failed.
            return result
        runner.grade_experiment(fixture.grade_args(), command_runner=abnormal_grader)
        grades = runner.read_jsonl(fixture.output / "grades.jsonl")
        self.assertEqual(len(grades), 6)
        self.assertTrue(all(row["resolved"] is None and row["status"] == "native_worker_abnormal_exit"
                            for row in grades))
        runner.grade_experiment(fixture.grade_args(), command_runner=successful_grader)
        latest = {runner.row_key(row): row for row in runner.read_jsonl(fixture.output / "grades.jsonl")}
        self.assertEqual(len(latest), 6)
        self.assertTrue(all(row["resolved"] is True for row in latest.values()))


if __name__ == "__main__":
    unittest.main()
