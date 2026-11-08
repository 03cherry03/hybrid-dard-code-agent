"""Offline multi-GPU coordination contracts; no CUDA or model loading."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, call, patch

from datasets_code import Task
from scripts import four_gpu


GPU_ROWS = "\n".join(f"{i}, GPU-fixture-{i}, Fixture GPU" for i in range(4))


class FourGpuTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "data"
        self.output = Path(self.temporary.name) / "campaign"
        self.output.mkdir()
        self.gpus = [{"uuid": f"GPU-fixture-{i}", "name": "Fixture GPU"} for i in range(4)]
        self.tasks = [Task(dataset, language, f"{dataset}/{i}", i, "public prompt", "sealed test")
                      for dataset, language, count in [("humaneval", "py", 7),
                                                       ("multiple_js", "js", 3),
                                                       ("multiple_rs", "rs", 3)]
                      for i in range(count)]

    def plan(self, *, fail_fast=False, backends="ar,standard,dard", repairs=1, selection_manifest=None):
        args = four_gpu.parser().parse_args([
            "functions", "--output", str(self.output), "--data-root", str(self.root),
            "--suites", "humaneval,multiple_js,multiple_rs", "--repeats", "3", "--seed", "7",
            "--backends", backends, "--repairs", str(repairs)]
            + (["--fail-fast"] if fail_fast else [])
            + (["--selection-manifest", str(selection_manifest)] if selection_manifest else []))
        with patch("datasets_code.load_tasks", return_value=self.tasks):
            plan, tasks = four_gpu.make_plan(args, self.gpus)
        self.assertEqual(tasks, self.tasks)
        return plan

    def prepare_model_files(self, plan, *, manifest=True):
        python = Path(plan["workers"][0]["command"][0])
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_text("fixture executable; never invoked")
        if not manifest:
            return
        models = {}
        for index, (name, revision) in enumerate([
            ("Qwen/Qwen2.5-Coder-7B-Instruct", "c03e6d358207e414f1eca0bb1891e29f1db0e242"),
            ("apple/DiffuCoder-7B-Instruct", "4fdd4580064ca5d11808069ce78f88d068753c96"),
        ]):
            snapshot = self.root / "models" / f"snapshot-{index}"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}")
            (snapshot / "model.safetensors").write_bytes(b"fixture, not a real model")
            models[name] = {"revision": revision, "snapshot": str(snapshot)}
        (self.root / "models/manifest.json").write_text(json.dumps(models))

    def test_gpu_aliases_resolve_to_four_physical_identities(self):
        with patch.object(four_gpu, "run_checked", return_value=GPU_ROWS):
            selected = four_gpu.resolve_gpus("0,GPU-fixture-1,2,GPU-fixture-3")
        self.assertEqual(selected, self.gpus)

    def test_duplicate_index_uuid_alias_and_incomplete_gpu_selection_rejected(self):
        with patch.object(four_gpu, "run_checked", return_value=GPU_ROWS):
            with self.assertRaisesRegex(ValueError, "distinct"):
                four_gpu.resolve_gpus("0,GPU-fixture-0,2,3")
            for spec in ("0,1,2", "0,1,2,4"):
                with self.subTest(spec=spec), self.assertRaises(ValueError):
                    four_gpu.resolve_gpus(spec)

    def test_plan_is_balanced_disjoint_and_keeps_all_backends_and_repeat_seeds(self):
        plan = self.plan()
        keys = [tuple(key) for worker in plan["workers"] for key in worker["task_keys"]]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(set(keys), {tuple(four_gpu.task_key(task)) for task in self.tasks})
        self.assertEqual([worker["task_count"] for worker in plan["workers"]], [4, 3, 3, 3])
        self.assertEqual(plan["expected_cases"], len(self.tasks) * 3 * 3)
        self.assertEqual(plan["seed"], 7)
        outputs = set()
        for index, worker in enumerate(plan["workers"]):
            command = worker["command"]
            for flag, expected in [("--backends", "ar,standard,dard"), ("--repeats", "3"),
                                   ("--seed", "7"), ("--num-shards", "4"),
                                   ("--shard-index", str(index)), ("--device", "cuda:0"),
                                   ("--diagnostics", "summary"), ("--runner", "local"),
                                   ("--block-size", "512"),
                                   ("--tau-c", "0.5"), ("--tau-u", "0.8"),
                                   ("--transfer-limit", "0")]:
                self.assertEqual(command[command.index(flag) + 1], expected)
            self.assertIn("--local-files-only", command)
            self.assertEqual(worker["gpu"], self.gpus[index])
            self.assertEqual(worker["expected_cases"], worker["task_count"] * 9)
            self.assertEqual(worker["task_seeds"], [[7, 8, 9] for _ in worker["task_keys"]])
            outputs.add(worker["output"])
        self.assertEqual(len(outputs), 4)
        groups = {tuple(four_gpu.task_key(task)[:2]) for task in self.tasks}
        for dataset, language in groups:
            counts = [sum(key[:2] == [dataset, language] for key in worker["task_keys"])
                      for worker in plan["workers"]]
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_saved_plan_is_idempotent_and_rejects_any_changed_gpu_or_budget(self):
        plan = self.plan()
        four_gpu.save_plan(self.output, plan)
        original = (self.output / "campaign.json").read_bytes()
        four_gpu.save_plan(self.output, copy.deepcopy(plan))
        for changed in ("gpu", "budget"):
            edited = copy.deepcopy(plan)
            if changed == "gpu":
                edited["workers"][0]["gpu"]["uuid"] = "GPU-other"
            else:
                edited["repeats"] += 1
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "changed"):
                four_gpu.save_plan(self.output, edited)
            self.assertEqual((self.output / "campaign.json").read_bytes(), original)

    def test_priority_and_standard_campaigns_preserve_task_and_seed_pairing(self):
        priority = self.plan(backends="ar,dard")
        standard = self.plan(backends="standard")
        self.assertEqual(priority["expected_cases"], len(self.tasks) * 2 * 3)
        self.assertEqual(standard["expected_cases"], len(self.tasks) * 3)
        for left, right in zip(priority["workers"], standard["workers"], strict=True):
            self.assertEqual(left["task_keys"], right["task_keys"])
            self.assertEqual(left["task_seeds"], right["task_seeds"])
            self.assertEqual(left["command"][left["command"].index("--backends") + 1], "ar,dard")
            self.assertEqual(right["command"][right["command"].index("--backends") + 1], "standard")
        self.populate_results(priority)
        with patch("experiment_analysis.write_report"):
            coverage = four_gpu.audit(priority, self.output)
        self.assertTrue(coverage["coverage_complete"])
        monitor = four_gpu.CampaignProgress(priority, self.output, time.monotonic())
        snapshot = monitor.snapshot({str(i): 0 for i in range(4)})
        self.assertTrue(snapshot["workers_successful"])
        self.assertEqual(snapshot["completed_cases"], priority["expected_cases"])
        self.assertNotIn("standard", snapshot["workers"][0]["completed_by_decoder"])

    def test_invalid_backend_selection_is_rejected_before_loading_data(self):
        for backends in ("", "ar,ar", "ar,typo", "ar,"):
            with self.subTest(backends=backends), self.assertRaisesRegex(ValueError, "backends"):
                self.plan(backends=backends)

    def test_exact_manifest_selection_precedes_sharding_and_freezes_worker_hash(self):
        selected = [self.tasks[index] for index in (2, 4, 6, 8, 9, 10, 11, 12)]
        path = self.output / "selection.json"
        path.write_text(json.dumps({"schema_version": 1,
            "task_keys": [four_gpu.task_key(task) for task in reversed(selected)],
            "selection_rule": "fixture_public_id_intersection"}))
        # The plan helper normally checks its unfiltered fixture selection.
        args = four_gpu.parser().parse_args(["functions", "--output", str(self.output),
            "--data-root", str(self.root), "--selection-manifest", str(path),
            "--repairs", "0", "--repeats", "1", "--backends", "ar,dard,standard"])
        with patch("datasets_code.load_tasks", return_value=self.tasks):
            plan, actual = four_gpu.make_plan(args, self.gpus)
        self.assertEqual(actual, selected)
        self.assertEqual(plan["task_count"], 8)
        self.assertEqual(plan["expected_cases"], 24)
        self.assertEqual(plan["backends"], ["ar", "dard", "standard"])
        self.assertEqual(plan["selection_manifest"]["sha256"], four_gpu.hashlib.sha256(path.read_bytes()).hexdigest())
        seen = set()
        for worker in plan["workers"]:
            seen.update(tuple(key) for key in worker["task_keys"])
            command = worker["command"]
            self.assertEqual(command[command.index("--repairs") + 1], "0")
            self.assertEqual(command[command.index("--selection-manifest") + 1], str(path))
            self.assertEqual(command[command.index("--selection-manifest-sha256") + 1],
                             plan["selection_manifest"]["sha256"])
        self.assertEqual(seen, {tuple(four_gpu.task_key(task)) for task in selected})

    def test_manifest_rejects_duplicate_missing_and_invalid_scope_before_launch(self):
        keys = [four_gpu.task_key(task) for task in self.tasks[:4]]
        path = self.output / "selection.json"
        invalid = [
            {"schema_version": 1, "task_keys": [*keys, keys[0]]},
            {"schema_version": 1, "task_keys": [["humaneval", "py", "missing"]]},
            {"schema_version": 2, "task_keys": keys},
            {"schema_version": 1, "task_keys": []},
            {"schema_version": 1, "task_keys": [["humaneval", "py", 4]]},
        ]
        for manifest in invalid:
            path.write_text(json.dumps(manifest))
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                self.plan(selection_manifest=path)
        path.write_text(json.dumps({"schema_version": 1, "task_keys": keys}))
        args = four_gpu.parser().parse_args(["functions", "--output", str(self.output),
            "--selection-manifest", str(path), "--limit", "1"])
        with patch("datasets_code.load_tasks") as load, self.assertRaisesRegex(ValueError, "limit 0"):
            four_gpu.make_plan(args, self.gpus)
        load.assert_not_called()

    def test_preflight_missing_models_fails_before_any_external_command(self):
        plan = self.plan()
        self.prepare_model_files(plan, manifest=False)
        with patch.object(four_gpu, "run_checked") as external:
            with self.assertRaisesRegex(RuntimeError, "Model manifest missing"):
                four_gpu.preflight(plan, self.tasks)
        external.assert_not_called()

    def test_preflight_wrong_revision_fails_before_any_external_command(self):
        plan = self.plan()
        self.prepare_model_files(plan)
        path = self.root / "models/manifest.json"
        models = json.loads(path.read_text())
        models["apple/DiffuCoder-7B-Instruct"]["revision"] = "changed"
        path.write_text(json.dumps(models))
        with patch.object(four_gpu, "run_checked") as external:
            with self.assertRaisesRegex(RuntimeError, "Pinned model snapshot missing"):
                four_gpu.preflight(plan, self.tasks)
        external.assert_not_called()

    def test_preflight_busy_selected_gpu_blocks_worker_and_evaluator_launch(self):
        plan = self.plan()
        self.prepare_model_files(plan)
        def external(command):
            if command[0] == "nvidia-smi":
                return "GPU-unselected, 900\nGPU-fixture-2, 901\n"
            raise AssertionError(f"Unexpected external command: {command}")
        with patch.object(four_gpu, "run_checked", side_effect=external), \
                patch("evaluation.preflight_local") as evaluator, \
                patch.object(four_gpu.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(RuntimeError, "already have compute processes"):
                four_gpu.execute(plan, self.tasks, self.output)
        evaluator.assert_not_called()
        popen.assert_not_called()

    def test_native_function_preflight_uses_selected_tasks_and_data_root(self):
        plan = self.plan()
        self.prepare_model_files(plan)
        with patch.object(four_gpu, "run_checked", return_value="") as external, \
             patch("evaluation.preflight_local") as evaluator:
            four_gpu.preflight(plan, self.tasks)
        self.assertEqual(external.call_count, 1)
        self.assertEqual(external.call_args.args[0][0], "nvidia-smi")
        evaluator.assert_called_once_with({"py", "js", "rs"}, tasks=self.tasks, data_root=self.root)

    def test_execute_collects_all_four_exit_codes_including_failures(self):
        plan = self.plan()
        children = [Mock(pid=8000 + i) for i in range(4)]
        for child, code in zip(children, [0, 2, 0, 1]):
            child.wait.return_value = code
        with patch.object(four_gpu, "preflight"), \
                patch.object(four_gpu.subprocess, "Popen", side_effect=children) as popen, \
                patch.object(four_gpu.os, "killpg") as kill, contextlib.redirect_stdout(io.StringIO()):
            result = four_gpu.execute(plan, self.tasks, self.output)
        self.assertFalse(result)
        self.assertEqual(popen.call_count, 4)
        for child in children:
            child.wait.assert_called_once_with()
        kill.assert_not_called()
        status = json.loads((self.output / "execution_status.json").read_text())
        self.assertEqual(status["exit_codes"], {"0": 0, "1": 2, "2": 0, "3": 1})
        for index, invocation in enumerate(popen.call_args_list):
            self.assertTrue(invocation.kwargs["start_new_session"])
            self.assertEqual(invocation.kwargs["env"]["CUDA_VISIBLE_DEVICES"], self.gpus[index]["uuid"])
            self.assertEqual(invocation.kwargs["env"]["HF_HUB_OFFLINE"], "1")

    def test_failure_policy_is_frozen_in_campaign_plan(self):
        default = self.plan()
        self.assertEqual(default["failure_policy"], "wait_all")
        four_gpu.save_plan(self.output, default)
        fail_fast = self.plan(fail_fast=True)
        self.assertEqual(fail_fast["failure_policy"], "fail_fast")
        self.assertTrue(all("--fail-fast" in worker["command"] for worker in fail_fast["workers"]))
        self.assertTrue(all("--fail-fast" not in worker["command"] for worker in default["workers"]))
        with self.assertRaisesRegex(ValueError, "changed"):
            four_gpu.save_plan(self.output, fail_fast)

    def test_source_changed_during_preflight_prevents_all_worker_launches(self):
        plan = self.plan()
        source = Path(self.temporary.name) / "controlled.py"
        source.write_text("before")
        plan["source_sha256"] = {source.name: four_gpu.hashlib.sha256(source.read_bytes()).hexdigest()}
        with patch.object(four_gpu, "ROOT", source.parent), \
             patch.object(four_gpu, "preflight", side_effect=lambda *_: source.write_text("after")), \
             patch.object(four_gpu.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(ValueError, "sources changed"):
                four_gpu.execute(plan, self.tasks, self.output)
        launch.assert_not_called()

    def test_fail_fast_detects_later_worker_without_waiting_for_earlier_worker(self):
        plan = self.plan(fail_fast=True)
        children = [Mock(pid=8100 + i) for i in range(4)]
        for child in children:
            child.poll.return_value = None
            child.wait.return_value = -signal.SIGINT
        children[2].poll.return_value = 7
        children[2].wait.return_value = 7
        with patch.object(four_gpu, "preflight"), \
                patch.object(four_gpu.subprocess, "Popen", side_effect=children), \
                patch.object(four_gpu.os, "killpg") as kill, \
                patch.object(four_gpu.time, "sleep") as sleep, contextlib.redirect_stdout(io.StringIO()):
            result = four_gpu.execute(plan, self.tasks, self.output, fail_fast=True)
        self.assertFalse(result)
        sleep.assert_not_called()
        # A wait without timeout on worker0 would hide worker2's failure.
        children[0].wait.assert_called_once_with(timeout=10)
        self.assertEqual(kill.call_args_list,
                         [call(children[i].pid, signal.SIGINT) for i in (0, 1, 3)])
        status = json.loads((self.output / "execution_status.json").read_text())
        self.assertEqual(status["triggering_failure"], {"shard": 2, "exit_code": 7})
        self.assertEqual(status["cancelled_shards"], [0, 1, 3])
        self.assertEqual(status["not_started_shards"], [])
        self.assertEqual(status["exit_codes"], {"0": -2, "1": -2, "2": 7, "3": -2})

    def test_fail_fast_success_polls_all_workers_without_cancelling_them(self):
        plan = self.plan(fail_fast=True)
        children = [Mock(pid=8200 + i) for i in range(4)]
        for child in children:
            child.poll.side_effect = [None, 0]
            child.wait.return_value = 0
        with patch.object(four_gpu, "preflight"), \
                patch.object(four_gpu.subprocess, "Popen", side_effect=children), \
                patch.object(four_gpu.os, "killpg") as kill, \
                patch.object(four_gpu.time, "sleep") as sleep, contextlib.redirect_stdout(io.StringIO()):
            result = four_gpu.execute(plan, self.tasks, self.output, fail_fast=True)
        self.assertTrue(result)
        sleep.assert_called_once_with(0.2)
        kill.assert_not_called()
        for child in children:
            child.wait.assert_called_once_with()
        status = json.loads((self.output / "execution_status.json").read_text())
        self.assertIsNone(status["triggering_failure"])
        self.assertEqual(status["exit_codes"], {str(i): 0 for i in range(4)})

    def test_partial_launch_failure_cleans_started_workers_despite_exit_race(self):
        plan = self.plan(fail_fast=True)
        children = [Mock(pid=8300 + i) for i in range(2)]
        for child in children:
            child.poll.return_value = None
            child.wait.return_value = -signal.SIGINT
        with patch.object(four_gpu, "preflight"), \
                patch.object(four_gpu.subprocess, "Popen", side_effect=[*children, OSError("fixture launch failure")]), \
                patch.object(four_gpu.os, "killpg", side_effect=[ProcessLookupError(), None]) as kill, \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(OSError, "fixture launch failure"):
                four_gpu.execute(plan, self.tasks, self.output, fail_fast=True)
        self.assertEqual(kill.call_args_list, [call(child.pid, signal.SIGINT) for child in children])
        for child in children:
            child.wait.assert_called_once_with(timeout=10)
        status = json.loads((self.output / "execution_status.json").read_text())
        self.assertEqual(status["coordinator_error_type"], "OSError")
        self.assertEqual(status["not_started_shards"], [2, 3])
        self.assertEqual(status["exit_codes"], {"0": -2, "1": -2})

    def test_interruption_terminates_and_reaps_only_owned_process_groups(self):
        plan = self.plan()
        children = [Mock(pid=9000 + i) for i in range(4)]
        for child in children:
            child.poll.return_value = None
            child.wait.return_value = 0
        children[0].wait.side_effect = [KeyboardInterrupt(), 0]
        children[1].wait.side_effect = [subprocess.TimeoutExpired("fixture", 10), 0]
        with patch.object(four_gpu, "preflight"), \
                patch.object(four_gpu.subprocess, "Popen", side_effect=children), \
                patch.object(four_gpu.os, "killpg") as kill, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                four_gpu.execute(plan, self.tasks, self.output)
        self.assertEqual(kill.call_args_list, [call(child.pid, signal.SIGINT) for child in children]
                         + [call(children[1].pid, signal.SIGKILL)])
        self.assertEqual([child.wait.call_count for child in children], [2, 2, 1, 1])

    def test_worker_environment_uses_plan_root_and_isolates_gpu_caches(self):
        plan = self.plan()
        with patch.dict(four_gpu.os.environ, {"HYBRID_DARD_ROOT": "/wrong-root", "HF_HOME": "/wrong-cache",
                "TMPDIR": "/wrong-tmp", "CUDA_VISIBLE_DEVICES": "wrong-gpu", "OMP_NUM_THREADS": "80"}):
            environments = [four_gpu.worker_env(plan, worker) for worker in plan["workers"]]
        for index, env in enumerate(environments):
            self.assertEqual(env["HYBRID_DARD_ROOT"], str(self.root))
            self.assertEqual(env["HF_HOME"], str(self.root / "cache/huggingface"))
            self.assertEqual(env["TMPDIR"], str(self.root / "work/tmp"))
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], self.gpus[index]["uuid"])
            self.assertEqual(env["OMP_NUM_THREADS"], "1")
            self.assertEqual(env["HF_HUB_OFFLINE"], "1")
            self.assertTrue(Path(env["TMPDIR"]).is_dir())
        for key in ("HF_MODULES_CACHE", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH"):
            self.assertEqual(len({env[key] for env in environments}), 4)

    def test_progress_tails_partial_lines_and_counts_retries_as_one_case(self):
        plan = self.plan()
        monitor = four_gpu.CampaignProgress(plan, self.output, time.monotonic())
        worker = plan["workers"][0]
        run = Path(worker["output"])
        run.mkdir()
        dataset, language, task_id = worker["task_keys"][0]
        row = {"dataset": dataset, "language": language, "task_id": task_id,
               "decoder": "ar", "repeat": 0, "seed": 7, "case_completed": False,
               "agent_wall_s": 4, "offline_grading_wall_s": 1}
        path = run / "case_results.jsonl"
        first = (json.dumps(row) + "\n").encode()
        second = (json.dumps({**row, "case_completed": True}) + "\n").encode()
        path.write_bytes(first + second[:20])
        observed = monitor.snapshot({}, now_unix=1000)
        self.assertEqual(observed["completed_cases"], 0)
        self.assertEqual(monitor.tails[0]["offset"], len(first))
        with path.open("ab") as stream:
            stream.write(second[20:])
        observed = monitor.snapshot({}, now_unix=1001)
        self.assertEqual(observed["completed_cases"], 1)
        self.assertEqual(len(monitor.tails[0]["rows"]), 1)
        self.assertEqual(observed["sample_counts"][f"ar/{dataset}/{language}"], 1)
        self.assertIsNone(observed["remaining_case_work_s"])
        self.assertIsNone(observed["projected_completion_unix"])
        self.assertIsNone(observed["completed_unix"])
        # A repeated monitor tick reads no prior bytes and cannot inflate counts.
        offset = monitor.tails[0]["offset"]
        self.assertEqual(monitor.snapshot({})["completed_cases"], 1)
        self.assertEqual(monitor.tails[0]["offset"], offset)
        # Replacement must discard stale case counts, not add another log's rows.
        replacement = run / "replacement.jsonl"
        replacement.write_bytes(first)
        replacement.replace(path)
        self.assertEqual(monitor.snapshot({})["completed_cases"], 0)

    def test_progress_projection_keeps_decoder_costs_and_requires_process_success(self):
        plan = self.plan()
        monitor = four_gpu.CampaignProgress(plan, self.output, time.monotonic())
        elapsed = {"ar": 1, "standard": 10, "dard": 20}
        for worker in plan["workers"]:
            run = Path(worker["output"])
            run.mkdir()
            rows = [{"dataset": dataset, "language": language, "task_id": task_id,
                     "decoder": decoder, "repeat": 0, "seed": 7, "case_completed": True,
                     "agent_wall_s": elapsed[decoder], "offline_grading_wall_s": 2}
                    for dataset, language, task_id in worker["task_keys"]
                    for decoder in elapsed]
            (run / "case_results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        snapshot = monitor.snapshot({}, now_unix=1000)
        # Two unobserved repeats remain. Costs include sealed offline grading.
        expected = max(w["task_count"] * 2 * (3 + 12 + 22) for w in plan["workers"])
        self.assertEqual(snapshot["remaining_case_work_s"], expected)
        self.assertEqual(snapshot["projected_completion_unix"], 1000 + expected)
        self.assertFalse(snapshot["coverage_complete"])
        self.assertIsNone(snapshot["completed_unix"])
        for worker in plan["workers"]:
            path = Path(worker["output"]) / "case_results.jsonl"
            path.rename(path.with_suffix(".old"))
        self.populate_results(plan)
        snapshot = monitor.snapshot({}, now_unix=2000)
        self.assertTrue(snapshot["coverage_complete"])
        self.assertFalse(snapshot["workers_successful"])
        self.assertIsNone(snapshot["completed_unix"])
        snapshot = monitor.snapshot({str(i): 0 for i in range(4)}, now_unix=2001)
        self.assertTrue(snapshot["workers_successful"])
        self.assertEqual(snapshot["completed_unix"], 2001)
        self.assertIsNone(snapshot["projected_completion_unix"])

    def test_progress_sampling_is_throttled_and_gpu_failure_is_observational(self):
        monitor = four_gpu.CampaignProgress(self.plan(), self.output, time.monotonic())
        with patch.object(four_gpu.subprocess, "run") as run:
            monitor.tick({})
        run.assert_not_called()
        self.assertFalse((self.output / "progress.json").exists())
        monitor.next_check = 0
        with patch.object(four_gpu.subprocess, "run", side_effect=subprocess.TimeoutExpired("gpu", 5)) as run:
            monitor.tick({})
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.kwargs["timeout"], 5)
        snapshot = json.loads((self.output / "progress.json").read_text())
        self.assertEqual(snapshot["gpu_sample_error"], "TimeoutExpired")
        self.assertEqual(snapshot["completed_cases"], 0)
        with patch.object(four_gpu.subprocess, "run") as run:
            monitor.tick({}, final=True)
        run.assert_not_called()
        snapshot = json.loads((self.output / "progress.json").read_text())
        self.assertTrue(snapshot["final_observation"])
        self.assertIsNone(snapshot["completed_unix"])

    def test_analyze_loads_frozen_plan_without_gpu_probe_or_repeating_generation_options(self):
        plan = self.plan()
        four_gpu.save_plan(self.output, plan)
        with patch.object(four_gpu, "resolve_gpus") as resolve, \
                patch.object(four_gpu, "make_plan") as make_plan, \
                patch.object(four_gpu, "audit", return_value={"coverage_complete": True}) as audit, \
                patch.object(four_gpu.subprocess, "Popen") as popen, contextlib.redirect_stdout(io.StringIO()):
            code = four_gpu.main(["functions", "--output", str(self.output), "--analyze"])
        self.assertEqual(code, 0)
        resolve.assert_not_called()
        make_plan.assert_not_called()
        popen.assert_not_called()
        self.assertEqual(audit.call_args.args, (plan, self.output))

    def run_isolated_descendant_cleanup(self, mode):
        """Exercise CLI cleanup in its own process; never change unittest's subreaper state."""
        plan = self.plan(fail_fast=True)
        pid_path = self.output / "detached.json"
        detached = (
            "import os,signal,time,json,pathlib\n"
            "signal.signal(signal.SIGINT,signal.SIG_IGN)\n"
            f"pathlib.Path({str(pid_path)!r}).write_text(json.dumps({{'pid':os.getpid(),"
            "'start':pathlib.Path('/proc/'+str(os.getpid())+'/stat').read_text().rsplit(') ',1)[1].split()[19]}))\n"
            "time.sleep(30)\n")
        doomed_worker = (
            "import os,signal,subprocess,sys,pathlib,time\n"
            f"subprocess.Popen([sys.executable,'-c',{detached!r}],start_new_session=True)\n"
            f"while not pathlib.Path({str(pid_path)!r}).exists(): time.sleep(.01)\n"
            "os.kill(os.getpid(),signal.SIGKILL)\n")
        for worker in plan["workers"]:
            worker["command"] = [sys.executable, "-c", doomed_worker if worker["index"] == 0 else "pass"]
        if mode == "grade":
            plan["kind"] = "repository"
        (self.output / "campaign.json").write_text(json.dumps(plan))
        script = '''import json,os,pathlib,signal,subprocess,sys,time
from unittest.mock import patch
from scripts import four_gpu
directory=pathlib.Path(sys.argv[1]); mode=sys.argv[2]
plan=json.loads((directory/'campaign.json').read_text())
detached=plan['workers'][0]['command'][2]
unrelated=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
def interrupted_grade(*_):
    # A real grading child starts a new session, then the coordinator receives SIGTERM.
    subprocess.Popen([sys.executable,'-c',detached],start_new_session=True)
    while not (directory/'detached.json').exists(): time.sleep(.01)
    os.kill(os.getpid(),signal.SIGTERM)
try:
    with patch.object(four_gpu,'resolve_gpus',return_value=[]), patch.object(four_gpu,'make_plan',return_value=(plan,[])), patch.object(four_gpu,'preflight'), patch.object(four_gpu,'audit',return_value={'coverage_complete':True}), patch.object(four_gpu,'grade',side_effect=interrupted_grade):
        args=[plan['kind'],'--output',str(directory),'--grade'] if mode=='grade' else [plan['kind'],'--output',str(directory),'--execute','--fail-fast']
        try: code=four_gpu.main(args)
        except KeyboardInterrupt: code=130
    child=json.loads((directory/'detached.json').read_text())
    try:
        fields=pathlib.Path('/proc/'+str(child['pid'])+'/stat').read_text().rsplit(') ',1)[1].split()
        child_alive=fields[19]==child['start'] and fields[0]!='Z'
    except OSError: child_alive=False
    print(json.dumps({'code':code,'owned_child_alive':child_alive,'unrelated_alive':unrelated.poll() is None}))
finally:
    unrelated.kill(); unrelated.wait()
'''
        process = subprocess.run([sys.executable, "-c", script, str(self.output), mode],
                                 cwd=four_gpu.ROOT, text=True, capture_output=True, timeout=20)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        result = json.loads(process.stdout.splitlines()[-1])
        self.assertNotEqual(result["code"], 0)
        self.assertFalse(result["owned_child_alive"], result)
        self.assertTrue(result["unrelated_alive"], result)

    def test_standalone_execute_reaps_detached_child_after_shard_sigkill(self):
        self.run_isolated_descendant_cleanup("execute")

    def test_standalone_grade_sigterm_cleans_owned_children_only(self):
        self.run_isolated_descendant_cleanup("grade")

    def populate_results(self, plan):
        for worker in plan["workers"]:
            run = Path(worker["output"])
            run.mkdir(parents=True, exist_ok=True)
            rows = [{"dataset": key[0], "language": key[1], "task_id": key[2],
                     "decoder": decoder, "repeat": repeat, "seed": seeds[repeat],
                     "final_status": "passed", "case_completed": True, "measurement_complete": True}
                    for key, seeds in zip(worker["task_keys"], worker["task_seeds"], strict=True)
                    for decoder in plan.get("backends", ("ar", "standard", "dard"))
                    for repeat in range(plan["repeats"])]
            (run / "case_results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_audit_requires_exact_coverage_and_flags_wrong_seed(self):
        plan = self.plan()
        self.populate_results(plan)
        with patch("experiment_analysis.write_report"):
            complete = four_gpu.audit(plan, self.output)
            self.assertTrue(complete["coverage_complete"])
            path = Path(plan["workers"][0]["output"]) / "case_results.jsonl"
            original = [json.loads(line) for line in path.read_text().splitlines()]
            for problem in ("missing", "unexpected", "seed"):
                rows = copy.deepcopy(original)
                if problem == "missing":
                    rows.pop()
                elif problem == "unexpected":
                    rows.append({**rows[0], "task_id": "not-in-plan"})
                else:
                    rows[0]["seed"] = 123456
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                with self.subTest(problem=problem):
                    report = four_gpu.audit(plan, self.output)
                    self.assertFalse(report["coverage_complete"])
                    affected = report["workers"][0]
                    self.assertGreater(affected["missing"] + affected["unexpected"], 0)


if __name__ == "__main__":
    unittest.main()
