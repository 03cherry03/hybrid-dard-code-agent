"""Handwritten temporary Git fixtures for native source and file boundaries.

No model programs, environment installation or network access is used.
"""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import repository_agent as agent


class RepoBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="repo-boundary-fixture_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / "checkout"
        self.repo.mkdir()
        self.run_git("init", "-q")
        self.run_git("config", "user.name", "Fixture")
        self.run_git("config", "user.email", "fixture@example.invalid")
        (self.repo / ".gitignore").write_text("ignored/\n*.cache\n")
        (self.repo / "source.py").write_text("BASE_CONTENT = 1\n")
        (self.repo / "delete_me.txt").write_text("base deletion fixture\n")
        (self.repo / "eval.sh").write_text("Tracked repository file, not injected evaluation data\n")
        (self.repo / "ignored").mkdir()
        (self.repo / "ignored/tracked.py").write_text("BASE_IGNORED_BUT_TRACKED = 2\n")
        self.run_git("add", "-A")
        self.run_git("add", "-f", "ignored/tracked.py")
        self.run_git("commit", "-qm", "fixture base")
        self.base = self.run_git("rev-parse", "HEAD").stdout.strip()
        self.base_tree = self.run_git("rev-parse", "HEAD^{tree}").stdout.strip()
        self.base_files = self.run_git("ls-files", "-z").stdout
        # Simulate runtime dependencies installed in an ignored repository path.
        (self.repo / "ignored/dependency.bin").write_bytes(b"PREPARED_DEPENDENCY\x00\xff")
        (self.repo / "runtime.cache").write_text("PREPARED_CACHE\n")

    def run_git(self, *args, cwd=None, input_text=None, check=True):
        return subprocess.run(["git", *args], cwd=cwd or self.repo, input=input_text,
                              capture_output=True, text=True, check=check)

    def add_future_history(self):
        (self.repo / "source.py").write_text("FUTURE_SOLUTION = 999\n")
        (self.repo / "future_only.txt").write_text("FUTURE_SECRET\n")
        self.run_git("add", "-A")
        self.run_git("commit", "-qm", "future solution")
        future = self.run_git("rev-parse", "HEAD").stdout.strip()
        self.run_git("tag", "future-answer", future)
        self.run_git("update-ref", "refs/remotes/origin/future", future)
        self.run_git("reset", "--hard", self.base)
        (self.repo / "test.patch").write_text("Injected artifact\n")
        return future

    def bootstrap(self):
        script = agent.BOOTSTRAP_SCRIPT.replace("/tmp/agent-tracked.XXXXXX", str(self.root / "agent-tracked.XXXXXX"))
        result = subprocess.run(["bash", "-c", script, "fixture-bootstrap", self.base], cwd=self.repo,
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        baseline = result.stdout.strip().splitlines()[-1]
        self.assertRegex(baseline, r"^[0-9a-f]{40}$")
        return baseline

    def test_bootstrap_preserves_exact_base_tree_and_ignored_dependencies_but_erases_future_history(self):
        future = self.add_future_history()
        baseline = self.bootstrap()
        self.assertNotEqual(baseline, future)
        self.assertEqual(self.run_git("rev-parse", "HEAD^{tree}").stdout.strip(), self.base_tree)
        self.assertEqual(self.run_git("ls-files", "-z").stdout, self.base_files)
        self.assertEqual((self.repo / "ignored/tracked.py").read_text(), "BASE_IGNORED_BUT_TRACKED = 2\n")
        self.assertEqual((self.repo / "ignored/dependency.bin").read_bytes(), b"PREPARED_DEPENDENCY\x00\xff")
        self.assertEqual((self.repo / "runtime.cache").read_text(), "PREPARED_CACHE\n")
        self.assertFalse((self.repo / "future_only.txt").exists())
        self.assertFalse((self.repo / "test.patch").exists())
        self.assertEqual(self.run_git("rev-list", "--all", "--count").stdout.strip(), "1")
        self.assertNotEqual(self.run_git("cat-file", "-e", future, check=False).returncode, 0)
        self.assertEqual(self.run_git("tag", "--list").stdout, "")
        self.assertEqual(self.run_git("status", "--porcelain").stdout, "")

    def test_final_patch_applies_to_original_base_and_excludes_ignored_runtime_dependencies(self):
        self.add_future_history()
        original = self.root / "original-base"
        self.run_git("clone", "--no-hardlinks", "--quiet", str(self.repo), str(original), cwd=self.root)
        self.run_git("reset", "--hard", self.base, cwd=original)
        baseline = self.bootstrap()
        (self.repo / "source.py").write_text("BASE_CONTENT = 2\n")
        (self.repo / "ignored/tracked.py").write_text("BASE_IGNORED_BUT_TRACKED = 3\n")
        (self.repo / "new_file.py").write_text("HANDWRITTEN_FIXTURE = True\n")
        (self.repo / "delete_me.txt").unlink()
        (self.repo / "ignored/dependency.bin").write_bytes(b"CHANGED_RUNTIME_CACHE")
        task = agent.RepoTask("fixture", "fixture-1", "fixture/repo", self.base, "fixture issue", "py")
        sandbox = agent.LocalSandbox(task, agent.AgentConfig())
        sandbox.baseline = baseline
        def local_handwritten_git_script(argv, **kwargs):
            self.assertEqual(argv[:2], ["/bin/bash", "-c"])
            result = subprocess.run(argv, cwd=self.repo, capture_output=True, text=True, timeout=10)
            return agent.CommandResult(result.returncode, result.stdout + result.stderr)
        with patch.object(sandbox, "_exec", side_effect=local_handwritten_git_script):
            diff = sandbox.patch(timeout=10)
        self.assertIn("ignored/tracked.py", diff)
        self.assertIn("new_file.py", diff)
        self.assertIn("delete_me.txt", diff)
        self.assertNotIn("dependency.bin", diff)
        self.assertNotIn("runtime.cache", diff)
        self.assertNotIn("FUTURE_SECRET", diff)
        self.run_git("apply", "--check", "-", cwd=original, input_text=diff)
        self.run_git("apply", "-", cwd=original, input_text=diff)
        self.assertEqual((original / "source.py").read_text(), "BASE_CONTENT = 2\n")
        self.assertEqual((original / "ignored/tracked.py").read_text(), "BASE_IGNORED_BUT_TRACKED = 3\n")
        self.assertFalse((original / "delete_me.txt").exists())

    def test_preparation_edits_and_untracked_build_metadata_become_baseline(self):
        (self.repo / "source.py").write_text("BASE_CONTENT = 2\n")
        (self.repo / "build-metadata.txt").write_text("public preparation metadata\n")
        self.bootstrap()
        self.assertEqual((self.repo / "source.py").read_text(), "BASE_CONTENT = 2\n")
        self.assertEqual(self.run_git("status", "--porcelain").stdout, "")
        self.assertIn("build-metadata.txt", self.run_git("ls-files").stdout)
        self.assertTrue((self.repo / "ignored/dependency.bin").exists())

    def test_wrong_head_is_rejected_without_rewriting_source(self):
        (self.repo / "source.py").write_text("WRONG_HEAD = 99\n")
        self.run_git("add", "-A")
        self.run_git("commit", "-qm", "not the requested base")
        result = subprocess.run(["bash", "-c", agent.BOOTSTRAP_SCRIPT, "fixture-bootstrap", self.base],
                                cwd=self.repo, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 42)
        self.assertEqual((self.repo / "source.py").read_text(), "WRONG_HEAD = 99\n")

    def file_action(self, request):
        script = agent.FILE_TOOL_SCRIPT
        payload = {**request, "_max_file_bytes": 10000, "_max_output_chars": 10000}
        result = subprocess.run([sys.executable, "-c", script], input=json.dumps(payload),
                                capture_output=True, text=True, timeout=10, cwd=self.repo)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_file_tools_reject_external_symlinks_for_read_edit_and_search(self):
        outside = self.root / "outside.txt"
        outside.write_text("OUTSIDE_SENTINEL\n")
        (self.repo / "external.txt").symlink_to(outside)
        (self.repo / "external-dir").symlink_to(self.root)
        for request in [
                {"action": "read", "path": "external.txt"},
                {"action": "edit", "path": "external.txt", "old": "OUTSIDE", "new": "CHANGED"},
                {"action": "edit", "path": "external-dir/created.txt", "old": None, "new": "bad"},
                {"action": "search", "path": "external.txt", "pattern": "SENTINEL"}]:
            with self.subTest(request=request):
                self.assertFalse(self.file_action(request)["ok"])
        result = self.file_action({"action": "search", "path": ".", "pattern": "OUTSIDE_SENTINEL"})
        self.assertTrue(result["ok"])
        self.assertNotIn("OUTSIDE_SENTINEL", result["output"])
        self.assertEqual(outside.read_text(), "OUTSIDE_SENTINEL\n")
        self.assertFalse((self.root / "created.txt").exists())

    def test_vcs_metadata_symlink_aliases_are_rejected(self):
        (self.repo / "metadata").symlink_to(self.repo / ".git")
        before = (self.repo / ".git/config").read_text()
        for request in [
                {"action": "read", "path": "metadata/config"},
                {"action": "edit", "path": "metadata/config", "old": "Fixture", "new": "Changed"},
                {"action": "search", "path": "metadata", "pattern": "Fixture"}]:
            with self.subTest(request=request):
                self.assertFalse(self.file_action(request)["ok"])
        self.assertEqual((self.repo / ".git/config").read_text(), before)


if __name__ == "__main__":
    unittest.main()
