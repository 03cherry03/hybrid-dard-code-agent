"""Offline repository-agent contract tests; no models, installation or network required."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import os
import shutil
import subprocess
import time
import tempfile
import unittest
from unittest.mock import patch

from repository_agent import (
    AgentConfig, CommandResult, LocalSandbox, RepoTask, bounded_messages,
    file_hash, load_repo_tasks, parse_action, run_bounded_command,
    run_repository_task, SYSTEM_PROMPT,
)
from dard_decoder import DARDConfig


ENVIRONMENT_ID = "2" * 64
BASELINE = "3" * 40
PATCH = "diff --git a/source.py b/source.py\n--- a/source.py\n+++ b/source.py\n@@ -1 +1 @@\n-old\n+new\n"


def task_record(**changes):
    return {
        "dataset": "fixture_suite", "instance_id": "org__repo-1", "repo": "org/repo",
        "base_commit": "1" * 40, "problem_statement": "Fix the public issue.",
        "language": "py", **changes,
    }


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeBackend:
    def __init__(self, responses, effect=None):
        self.responses = iter(responses)
        self.calls = []
        self.effect = effect

    def generate_messages(self, messages, decoder, **kwargs):
        self.calls.append({"messages": messages, "decoder": decoder, "kwargs": kwargs})
        if self.effect:
            self.effect()
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, str):
            response = {"text": response}
        return {"output_tokens": 7, "model_forward_passes": 2,
                "attention_fallback_steps": 0, "gpu_peak_gib": 1.25, **response}


class FakeSandbox:
    def __init__(self, task, config, *, observations=None, start_error=None, close_error=None):
        self.task, self.config = task, config
        self.environment_id = ENVIRONMENT_ID
        self.observations = iter(observations or [])
        self.start_error, self.close_error = start_error, close_error
        self.started = False
        self.closed = False
        self.actions = []
        self.patch_calls = 0

    def start(self, *, timeout):
        self.started = True
        self.start_timeout = timeout
        if self.start_error:
            raise self.start_error

    def action(self, action, *, timeout):
        self.actions.append((action, timeout))
        return next(self.observations, {"ok": True, "output": "public tool observation"})

    def patch(self, *, timeout):
        self.patch_calls += 1
        self.patch_timeout = timeout
        return PATCH

    def close(self):
        self.closed = True
        if self.close_error:
            raise self.close_error


class LoaderTests(unittest.TestCase):
    def test_unknown_gold_fields_cannot_enter_task_prompt_or_serialized_state(self):
        raw = task_record(patch="SECRET_GOLD_PATCH", test_patch="SECRET_TEST_PATCH",
                          eval_script="SECRET_EVAL_SCRIPT", FAIL_TO_PASS=["secret_test"])
        task = RepoTask.from_record(raw)
        serialized = json.dumps(asdict(task)) + task.prompt()
        for secret in ("SECRET_GOLD_PATCH", "SECRET_TEST_PATCH", "SECRET_EVAL_SCRIPT", "secret_test"):
            self.assertNotIn(secret, serialized)
        raw["problem_statement"] = "mutated source object"
        self.assertEqual(task.problem_statement, "Fix the public issue.")

    def test_loader_checks_manifest_and_environment_lock_without_opening_sealed_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks_path = root / "tasks.jsonl"
            tasks_path.write_text(json.dumps(task_record(patch="DO_NOT_RETAIN")) + "\n")
            # Deliberately unreadable as JSON. Loading tasks must never inspect it.
            (root / "sealed.jsonl").write_text("invalid secret sealed material")
            manifest = {"tasks_sha256": file_hash(tasks_path), "count": 1,
                        "dataset": "fixture_suite", "sealed_sha256": "sealed-provenance-only"}
            (root / "manifest.json").write_text(json.dumps(manifest))
            (root / "repositories.lock.json").write_text(json.dumps({"org__repo-1": {
                "base_commit": "1" * 40, "environment_id": ENVIRONMENT_ID}}))
            original_open = Path.open
            def public_files_only(path, *args, **kwargs):
                if path.name == "sealed.jsonl":
                    self.fail("Agent task loader opened sealed evaluation data")
                return original_open(path, *args, **kwargs)
            with patch.object(Path, "open", public_files_only):
                tasks, sources = load_repo_tasks([tasks_path])
            self.assertEqual(tasks[0].environment_id, ENVIRONMENT_ID)
            self.assertEqual(sources[0]["sealed_sha256"], "sealed-provenance-only")
            self.assertNotIn("DO_NOT_RETAIN", json.dumps([asdict(x) for x in tasks]))
            tasks_path.write_text(tasks_path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_repo_tasks([tasks_path])

    def test_loader_rejects_lock_mismatch_and_duplicate_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tasks.jsonl"
            path.write_text(json.dumps(task_record()) + "\n")
            lock = path.with_name("repositories.lock.json")
            lock.write_text(json.dumps({"org__repo-1": {
                "base_commit": "9" * 40, "environment_id": ENVIRONMENT_ID}}))
            with self.assertRaisesRegex(ValueError, "base commit"):
                load_repo_tasks([path])
            lock.unlink()
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_repo_tasks([path, path])
            with self.assertRaisesRegex(ValueError, "absent"):
                load_repo_tasks([path], instance_ids={"missing"})

    def test_task_rejects_mutable_commit_and_unsafe_identifiers(self):
        for changes in ({"base_commit": "main"}, {"instance_id": "../escape"},
                        {"dataset": "suite/path"}, {"source_path": "relative"},
                        {"environment_id": "mutable-tag"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                RepoTask.from_record(task_record(**changes))


class ProtocolTests(unittest.TestCase):
    def test_exactly_one_json_action_and_no_extra_fields(self):
        config = AgentConfig()
        for output in ('```json\n{"action":"finish"}\n```', '[]',
                       '{"action":"finish"}\n{"action":"finish"}',
                       '{"action":"run","command":"true","host":true}',
                       '{"action":"unknown"}'):
            with self.subTest(output=output), self.assertRaises((ValueError, TypeError)):
                parse_action(output, config)

    def test_path_range_command_and_edit_budgets(self):
        config = AgentConfig(max_file_bytes=100)
        invalid = [{"action": "read", "path": name} for name in (
            "/etc/passwd", "../secret", "a/../../secret", ".git/config", "a/.git/HEAD", "x\0y")]
        invalid.extend([
            {"action": "read", "path": "a", "start_line": 1, "end_line": 401},
            {"action": "read", "path": "a", "start_line": True},
            {"action": "edit", "path": "a", "old": "", "new": "x"},
            {"action": "edit", "path": "a", "old": None, "new": "é" * 51},
            {"action": "run", "command": "x\0y"},
            {"action": "search", "pattern": ""},
        ])
        for action in invalid:
            with self.subTest(action=action), self.assertRaises(ValueError):
                parse_action(json.dumps(action, ensure_ascii=False), config)
        valid = {"action": "edit", "path": "new.py", "old": None, "new": "x"}
        self.assertEqual(parse_action(json.dumps(valid), AgentConfig()), valid)

    def test_context_eviction_keeps_whole_pairs_and_initial_issue(self):
        initial = [{"role": "system", "content": "policy"}, {"role": "user", "content": "issue"}]
        history = [{"role": "assistant", "content": "old-action"},
                   {"role": "user", "content": "old-observation"},
                   {"role": "assistant", "content": "new"},
                   {"role": "user", "content": "observation"}]
        result = bounded_messages(initial, history, 25)
        self.assertEqual(result, initial + history[2:])
        self.assertEqual(len(history), 4)
        with self.assertRaisesRegex(ValueError, "context budget"):
            bounded_messages(initial, [], 3)


class AgentLoopTests(unittest.TestCase):
    def run_fixture(self, backend, *, config=None, sandbox_options=None, task=None):
        config = config or AgentConfig(max_steps=5)
        task = task or RepoTask.from_record(task_record())
        sandbox = FakeSandbox(task, config, **(sandbox_options or {}))
        result = run_repository_task(task, backend, "dard", config=config, seed=17,
                                     sandbox_factory=lambda *_: sandbox)
        return result, sandbox

    def test_shared_loop_edit_test_finish_extracts_patch_and_excludes_gold(self):
        task = RepoTask.from_record(task_record(patch="SECRET_GOLD", test_patch="SECRET_TEST"))
        backend = FakeBackend([
            json.dumps({"action": "edit", "path": "source.py", "old": "old", "new": "new"}),
            json.dumps({"action": "run", "command": "python -m unittest"}),
            json.dumps({"action": "finish", "message": "done"}),
        ])
        result, sandbox = self.run_fixture(backend, task=task)
        self.assertEqual(result["status"], "submitted")
        self.assertEqual(result["model_patch"], PATCH)
        self.assertEqual(result['patch_sha256'], hashlib.sha256(PATCH.encode()).hexdigest())
        self.assertEqual(result['patch_bytes'], len(PATCH.encode()))
        self.assertEqual(result['patch_changed_files'], 1)
        self.assertEqual((result["steps"], result["tool_calls"]), (3, 2))
        self.assertEqual(result["model_forward_passes"], 6)
        self.assertEqual(result["output_tokens"], 21)
        self.assertEqual(result["environment_id"], ENVIRONMENT_ID)
        self.assertEqual(sandbox.patch_calls, 1)
        self.assertTrue(sandbox.closed)
        self.assertTrue(result["speed_eligible"])
        transcript = json.dumps([call["messages"] for call in backend.calls])
        self.assertNotIn("SECRET_GOLD", transcript)
        self.assertNotIn("SECRET_TEST", transcript)
        self.assertEqual(asdict(sandbox.task), asdict(task))
        self.assertEqual([call["kwargs"]["seed"] for call in backend.calls], [17, 18, 19])
        for call in backend.calls:
            self.assertEqual(call["kwargs"]["max_input_tokens"], 8192)
            self.assertGreater(call["kwargs"]["max_time"], 0)
            self.assertIn("deadline", call["kwargs"])
            self.assertTrue(all(m["role"] in {"system", "user", "assistant"} for m in call["messages"]))

    def test_invalid_json_consumes_generation_budget_but_cannot_execute_tool(self):
        backend = FakeBackend(["not JSON", "[]", '{"action":"unknown"}'])
        result, sandbox = self.run_fixture(backend, config=AgentConfig(max_steps=3))
        self.assertEqual(result["status"], "step_budget")
        self.assertEqual(result["steps"], 3)
        self.assertEqual(result["tool_calls"], 0)
        self.assertEqual(result['invalid_action_count'], 3)
        self.assertIn('action_parse_error', result['trace'][0])
        self.assertEqual(sandbox.actions, [])
        self.assertIn("Invalid action", backend.calls[1]["messages"][-1]["content"])
        self.assertEqual(sandbox.patch_calls, 1)

    def test_tool_budget_blocks_extra_execution(self):
        action = json.dumps({"action": "read", "path": "source.py"})
        result, sandbox = self.run_fixture(FakeBackend([action, action]),
                                           config=AgentConfig(max_tool_calls=1, max_steps=3))
        self.assertEqual(result["status"], "tool_budget")
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(len(sandbox.actions), 1)
        self.assertEqual(sandbox.patch_calls, 1)

    def test_backend_exception_keeps_elapsed_generation_and_cleans_workspace(self):
        clock = Clock()
        backend = FakeBackend([RuntimeError("model failure")], effect=lambda: clock.advance(2.5))
        with patch("repository_agent.time.monotonic", clock):
            result, sandbox = self.run_fixture(backend)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["generation_wall_s"], 2.5)
        self.assertEqual(result["trace"][0]["generation_call_wall_s"], 2.5)
        self.assertIn("model failure", result["trace"][0]["generation_error"])
        self.assertFalse(result["speed_eligible"])
        self.assertEqual(result["model_patch"], "")
        self.assertEqual(sandbox.patch_calls, 0)
        self.assertTrue(sandbox.closed)

    def test_late_generation_response_is_discarded_without_executing_action(self):
        clock = Clock()
        backend = FakeBackend(['{"action":"run","command":"touch should-not-run"}'],
                              effect=lambda: clock.advance(2))
        with patch("repository_agent.time.monotonic", clock):
            result, sandbox = self.run_fixture(backend, config=AgentConfig(task_timeout_s=1))
        self.assertEqual(result["status"], "deadline")
        self.assertTrue(result["deadline_violation"])
        self.assertEqual(result["generation_wall_s"], 2)
        self.assertEqual(sandbox.actions, [])
        self.assertEqual(sandbox.patch_calls, 0)
        self.assertEqual(result["model_patch"], "")
        self.assertTrue(sandbox.closed)

    def test_tool_timeout_discards_submission_and_always_cleans(self):
        backend = FakeBackend(['{"action":"run","command":"python -m unittest"}'])
        result, sandbox = self.run_fixture(backend, sandbox_options={
            "observations": [{"ok": False, "timed_out": True}]})
        self.assertEqual(result["status"], "deadline")
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(sandbox.patch_calls, 0)
        self.assertTrue(sandbox.closed)

    def test_backend_timeout_exception_records_cost_without_partial_submission(self):
        clock = Clock()
        backend = FakeBackend([TimeoutError("forward deadline")], effect=lambda: clock.advance(1.5))
        with patch("repository_agent.time.monotonic", clock):
            result, sandbox = self.run_fixture(backend, config=AgentConfig(task_timeout_s=1))
        self.assertEqual(result["status"], "deadline")
        self.assertEqual(result["generation_wall_s"], 1.5)
        self.assertEqual(result["model_patch"], "")
        self.assertEqual(sandbox.patch_calls, 0)
        self.assertTrue(sandbox.closed)

    def test_oversized_initial_context_fails_before_start_or_generation(self):
        backend = FakeBackend([])
        result, sandbox = self.run_fixture(backend, config=AgentConfig(max_context_chars=10))
        self.assertEqual(result["status"], "error")
        self.assertIn("context budget", result["error"])
        self.assertEqual(backend.calls, [])
        self.assertFalse(sandbox.started)
        self.assertTrue(sandbox.closed)

    def test_start_and_cleanup_errors_are_recorded(self):
        result, sandbox = self.run_fixture(FakeBackend([]), sandbox_options={
            "start_error": RuntimeError("environment unavailable")})
        self.assertEqual(result["status"], "error")
        self.assertTrue(sandbox.closed)
        result, sandbox = self.run_fixture(FakeBackend(['{"action":"finish"}']), sandbox_options={
            "close_error": RuntimeError("cleanup unavailable")})
        self.assertEqual(result["status"], "cleanup_error")
        self.assertEqual(result["model_patch"], "")
        self.assertFalse(result["speed_eligible"])

    def test_attention_fallback_excludes_speed_measurement(self):
        result, sandbox = self.run_fixture(FakeBackend([
            {"text": '{"action":"finish"}', "attention_fallback_steps": 2}]))
        self.assertEqual(result["status"], "submitted")
        self.assertFalse(result["speed_eligible"])
        self.assertEqual(result["attention_fallback_steps"], 2)

    def test_context_and_tool_truncation_are_recorded_without_altering_messages(self):
        task = RepoTask.from_record(task_record())
        config = AgentConfig(max_steps=3, max_context_chars=len(SYSTEM_PROMPT)+len(task.prompt())+20,
                             max_tool_output_chars=40)
        backend = FakeBackend(['{"action":"read","path":"source.py"}', '{"action":"finish"}'])
        result, _ = self.run_fixture(backend, task=task, config=config,
            sandbox_options={'observations': [{'ok': False, 'output': 'x'*200, 'truncated': True}]})
        event = result['trace'][0]
        self.assertEqual(result['tool_error_count'], 1)
        self.assertEqual(result['tool_output_truncations'], 1)
        self.assertIn('tool_wall_s', event)
        self.assertTrue(event['tool_output']['source_truncated'])
        self.assertTrue(event['tool_output']['context_truncated'])
        self.assertEqual(event['tool_output']['delivered_chars'], 40)
        self.assertEqual(result['context_trim_events'], 1)
        self.assertEqual(result['trace'][1]['context']['dropped_history_messages'], 2)
        self.assertEqual(len(backend.calls[1]['messages']), 2)

    def test_generation_failure_retains_partial_telemetry(self):
        error = RuntimeError('late model failure')
        error.generation_diagnostics = {'model_forward_passes': 13, 'gpu_peak_gib': 2,
            'diagnostics': {'complete': False, 'phase': 'sampling'}}
        result, _ = self.run_fixture(FakeBackend([error]))
        self.assertEqual(result['model_forward_passes'], 13)
        self.assertEqual(result['max_gpu_peak_gib'], 2)
        self.assertFalse(result['measurement_complete'])
        self.assertEqual(result['trace'][0]['generation']['diagnostics']['phase'], 'sampling')

    def test_instrumented_run_cannot_be_used_for_controlled_speed(self):
        task = RepoTask.from_record(task_record())
        config = AgentConfig(max_steps=1)
        backend = FakeBackend(['{"action":"finish"}'])
        sandbox = FakeSandbox(task, config)
        result = run_repository_task(task, backend, 'dard', config=config, seed=17,
            dard_config=DARDConfig(diagnostics='profile'), sandbox_factory=lambda *_: sandbox)
        self.assertEqual(result['status'], 'submitted')
        self.assertTrue(result['instrumentation_affects_timing'])
        self.assertFalse(result['speed_eligible'])


class LocalSandboxTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "prepared"
        self.source.mkdir()
        self.prefix = self.root / "prefix"
        (self.prefix / "bin").mkdir(parents=True)
        (self.prefix / "bin/python").symlink_to(sys.executable)
        (self.source / "source.py").write_text("VALUE = 1\n")
        (self.source / ".gitignore").write_text("dependencies/\n")
        self.git("init", "-q")
        self.git("config", "user.name", "Fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        self.git("add", "-A")
        self.git("commit", "-qm", "base")
        self.commit = self.git("rev-parse", "HEAD").stdout.strip()
        (self.source / "dependencies").mkdir()
        (self.source / "dependencies/installed.txt").write_text("prepared dependency")
        self.task = RepoTask.from_record(task_record(base_commit=self.commit,
            source_path=str(self.source), environment_path=str(self.prefix),
            environment_id=ENVIRONMENT_ID))
        self.config = AgentConfig(work_root=str(self.root / "work"))
        self.verify = patch.object(LocalSandbox, "_verify", return_value=None)
        self.verify.start()
        self.addCleanup(self.verify.stop)
        # Fixtures contain no relocatable binaries. Production always invokes
        # conda --offline --copy to relocate actual prepared environments.
        self.cloner = patch("repository_agent.clone_environment", side_effect=
            lambda source, destination, **_: Path(shutil.copytree(source, destination, symlinks=True)))
        self.cloner.start()
        self.addCleanup(self.cloner.stop)

    def git(self, *argv):
        return subprocess.run(["git", *argv], cwd=self.source, check=True, text=True, capture_output=True)

    def test_native_read_edit_run_patch_and_cleanup_use_fresh_source_and_prefix(self):
        sandbox = LocalSandbox(self.task, self.config)
        sandbox.start(timeout=20)
        try:
            workspace = sandbox.workspace
            self.assertNotEqual(workspace, self.source)
            self.assertEqual(sandbox.environment_id, ENVIRONMENT_ID)
            self.assertEqual((workspace / "dependencies/installed.txt").read_text(), "prepared dependency")
            self.assertEqual(sandbox.action({"action":"read", "path":"source.py"}, timeout=3)["output"], "1: VALUE = 1")
            self.assertTrue(sandbox.action({"action":"edit", "path":"source.py", "old":"1", "new":"2"}, timeout=3)["ok"])
            result = sandbox.action({"action":"run", "command":"python -c 'import source; print(source.VALUE)'"}, timeout=3)
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["output"].strip(), "2")
            self.assertIn("+VALUE = 2", sandbox.patch(timeout=3))
            self.assertEqual((self.source / "source.py").read_text(), "VALUE = 1\n")
            self.assertNotEqual(str(sandbox.environment_prefix), str(self.prefix))
        finally:
            sandbox.close()
        self.assertFalse(workspace.exists())
        self.assertTrue(self.source.exists())

    def test_setup_failure_removes_only_owned_trial(self):
        sandbox = LocalSandbox(self.task, self.config)
        with patch("repository_agent.clone_environment", side_effect=RuntimeError("clone failed")):
            with self.assertRaisesRegex(RuntimeError, "clone failed"):
                sandbox.start(timeout=20)
        root = sandbox.trial_root
        sandbox.close()
        self.assertFalse(root.exists())
        self.assertTrue(self.source.exists())

    def test_unprepared_task_fails_before_work_directory_creation(self):
        sandbox = LocalSandbox(RepoTask.from_record(task_record()), self.config)
        with self.assertRaisesRegex(RuntimeError, "prepare_repositories"):
            sandbox.start(timeout=20)
        self.assertIsNone(sandbox.trial_root)

    def test_shared_environment_mutation_invalidates_patch_and_cleans_trial(self):
        sandbox = LocalSandbox(self.task, self.config)
        backend = FakeBackend(['{"action":"finish"}'])
        with patch.object(sandbox, "_verify", side_effect=[None, RuntimeError("environment changed")]):
            result = run_repository_task(self.task, backend, "ar", config=self.config,
                                        seed=1, sandbox_factory=lambda *_: sandbox)
        self.assertEqual(result["status"], "cleanup_error")
        self.assertEqual(result["model_patch"], "")
        self.assertFalse(result["speed_eligible"])
        self.assertIsNone(sandbox.trial_root)

    def test_patch_budget_is_enforced(self):
        sandbox = LocalSandbox(self.task, AgentConfig(work_root=str(self.root / "work"), max_patch_bytes=10))
        sandbox.start(timeout=20)
        try:
            sandbox.action({"action":"edit", "path":"source.py", "old":"1", "new":"2"}, timeout=3)
            with self.assertRaisesRegex(ValueError, "submission size"):
                sandbox.patch(timeout=3)
        finally:
            sandbox.close()


class BoundedCommandTests(unittest.TestCase):
    @staticmethod
    def process_running(stat):
        try:
            return stat.read_text().rsplit(") ", 1)[1].split()[0] != "Z"
        except OSError:
            return False

    def test_explicit_environment_working_directory_and_resource_exec_wrapper(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_bounded_command([sys.executable, "-c",
                "import os,resource; print(os.getcwd()); print(os.environ['FIXTURE_VALUE']); "
                "print(resource.getrlimit(resource.RLIMIT_CPU)[0])"],
                cwd=tmp, env={**os.environ, "FIXTURE_VALUE": "native-fixture"},
                timeout=3, cpu_seconds=2)
            self.assertEqual(result.returncode, 0, result.output)
            self.assertEqual(result.output.splitlines(), [tmp, "native-fixture", "2"])

    def test_successful_parent_does_not_leave_background_children_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            child_path = Path(tmp) / "child.pid"
            code = ("import subprocess,sys,pathlib; "
                    "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],"
                    "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
                    "pathlib.Path(sys.argv[1]).write_text(str(child.pid))")
            result = run_bounded_command([sys.executable, "-c", code, str(child_path)], timeout=3)
            self.assertEqual(result.returncode, 0, result.output)
            stat = Path("/proc") / child_path.read_text() / "stat"
            deadline = time.monotonic() + 2
            while self.process_running(stat) and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertFalse(self.process_running(stat))

    def test_outer_timeout_cleans_nested_native_evaluator(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid_path = Path(tmp) / "evaluator.pid"
            leaf = ("import os,pathlib,time; pathlib.Path(" + repr(str(pid_path)) +
                    ").write_text(str(os.getpid())); time.sleep(30)")
            worker = ("import sys; from local_execution import run_bounded_command; "
                      "run_bounded_command([sys.executable,'-c'," + repr(leaf) + "],timeout=60)")
            result = run_bounded_command([sys.executable, "-c", worker], timeout=.6)
            self.assertTrue(result.timed_out)
            self.assertTrue(pid_path.exists(), result.output)
            stat = Path("/proc") / pid_path.read_text() / "stat"
            deadline = time.monotonic() + 2
            while self.process_running(stat) and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertFalse(self.process_running(stat))

    def test_timeout_cleans_observed_descendant_that_started_another_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid_path = Path(tmp) / "detached.pid"
            code = ("import subprocess,sys,pathlib,time; child=subprocess.Popen("
                    "[sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True); "
                    "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(30)")
            result = run_bounded_command([sys.executable, "-c", code, str(pid_path)], timeout=.6)
            self.assertTrue(result.timed_out)
            stat = Path("/proc") / pid_path.read_text() / "stat"
            deadline = time.monotonic() + 2
            while self.process_running(stat) and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertFalse(self.process_running(stat))

    def test_offline_clone_requests_independent_package_files(self):
        from local_execution import clone_environment
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "source", Path(tmp) / "target"
            (source / "conda-meta").mkdir(parents=True)
            with patch("local_execution.run_bounded_command", return_value=CommandResult(0, "cloned")) as runner:
                with patch.dict(os.environ, {"CONDA_EXE": "/fixture/conda"}):
                    self.assertEqual(clone_environment(source, target, timeout=5), target)
            command = runner.call_args.args[0]
            for option in ("--offline", "--copy", "--clone", "--prefix"):
                self.assertIn(option, command)

    def test_environment_fingerprint_detects_mutations_but_ignores_bytecode(self):
        from local_execution import environment_fingerprint
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp)
            (prefix / "library.py").write_text("original")
            before = environment_fingerprint(prefix)
            (prefix / "__pycache__").mkdir()
            (prefix / "__pycache__/library.pyc").write_bytes(b"compiled")
            self.assertEqual(environment_fingerprint(prefix), before)
            (prefix / "library.py").write_text("modified contents")
            self.assertNotEqual(environment_fingerprint(prefix), before)

    def test_native_environment_relocates_prepared_paths_and_seeds_trial_caches(self):
        from local_execution import native_environment
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prefix, workspace, runtime = root / "prefix", root / "checkout", root / "runtime"
            (prefix / "native-home/.cargo").mkdir(parents=True)
            (prefix / "native-home/.cargo/prepared.txt").write_text("cached dependency")
            (prefix / "native-environment.json").write_text(json.dumps({
                "environment": {"RUSTUP_HOME": "${PREFIX}/rustup", "CARGO_HOME": "${RUNTIME}/home/.cargo"},
                "path_entries": ["${PREFIX}/go/bin"]}))
            env = native_environment(prefix, workspace, runtime)
            self.assertEqual(env["RUSTUP_HOME"], str(prefix / "rustup"))
            self.assertEqual(env["CARGO_HOME"], str(runtime / "home/.cargo"))
            self.assertEqual(env["PATH"].split(os.pathsep)[0], str(prefix / "go/bin"))
            self.assertEqual(env["PYTHONPATH"].split(os.pathsep), [str(workspace), str(workspace / "src")])
            cache = runtime / "home/.cargo/prepared.txt"
            self.assertEqual(cache.read_text(), "cached dependency")
            cache.write_text("changed trial cache")
            self.assertEqual((prefix / "native-home/.cargo/prepared.txt").read_text(), "cached dependency")

    def test_output_is_drained_without_retaining_unbounded_data(self):
        result = run_bounded_command([sys.executable, "-c", "import os; os.write(1,b'x'*200000)"],
                                     timeout=3, max_output_bytes=127)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.output, "x" * 127)
        self.assertTrue(result.truncated)
        self.assertFalse(result.timed_out)

    def test_unresponsive_process_and_blocked_stdin_respect_deadline(self):
        result = run_bounded_command([sys.executable, "-c", "import time; time.sleep(10)"],
                                     input_text="x" * 200000, timeout=.2, max_output_bytes=100)
        self.assertTrue(result.timed_out)
        self.assertNotEqual(result.returncode, 0)
        self.assertLess(result.wall_s, 3)


if __name__ == "__main__":
    unittest.main()
