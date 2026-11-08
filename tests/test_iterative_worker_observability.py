"""CPU checks that durable diagnostics preserve worker outcomes and recovery."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from iteration_observability import verify_event_stream
from iterative_tuning import AR, candidate, file_sha, read_json, task_id, write_json
from scripts import iterative_eval_worker as evaluation_worker
from scripts import train_diffusion_lora as training_worker


def events(root):
    result = []
    for path in sorted((Path(root) / "events").glob("*.jsonl")):
        verify_event_stream(path)
        result.extend(json.loads(line) for line in path.read_text().splitlines())
    return result


class FakeBackend:
    def __init__(self, failure=None):
        self.model = torch.nn.Linear(1, 1)
        self.closed = False
        self.failure = failure
        self.generated = {"text": "def identity(x): return x\n```", "wall_s": 0.25,
            "generated_token_ids": [12, 13, 14], "terminal_token_index": 2,
            "raw_text_before_terminal": "def identity(x): return x<end><pad>",
            "text_before_special_token_cleanup": "def identity(x): return x<end>",
            "forced_fill_records": [{"position": 2, "token_id": 14, "probability": .7}],
            "diagnostics": {"stage_wall_s": {"model": .24}, "complete": True}}

    def metadata(self):
        return {"model": "fixture", "device": "cpu"}

    def generate(self, *args, **kwargs):
        return self.generated

    def generate_messages(self, *args, **kwargs):
        self.model(torch.zeros(1, 1))
        if self.failure:
            raise self.failure
        return self.generated

    def close(self):
        self.closed = True


class EvaluationWorkerLoggingTests(unittest.TestCase):
    def fixture(self, root):
        row = dict(dataset="humaneval", family="humaneval", language="py", task_id="fixture/1",
            number=1, source_id="fixture/1", prompt="def identity(x):\n    '''Return x.'''",
            required_name="identity", test_source="assert identity(1) == 1")
        tasks = root / "tasks.jsonl"
        tasks.write_text(json.dumps(row) + "\n")
        config = candidate(AR, decoder="ar", tokens=32, native_steps=4, forward_budget=32)
        plan = {"tasks_path": str(tasks), "tasks_sha256": file_sha(tasks), "split": "smoke",
            "output": str(root / "output"), "candidates": [config], "seed": 17,
            "deadline_unix": time.time()+300, "data_root": str(root), "generation_timeout": 10,
            "grading_timeout": 5}
        path = root / "plan.json"
        write_json(path, plan)
        return SimpleNamespace(plan=path, shard=0), root / "output/shard-0", row, config

    def run_fixture(self, args, backend, verdict=None):
        verdict = verdict or {"passed": True, "status": "passed", "grading_wall_s": .1,
            "execution": {"exit_code": 0, "stdout": "checked", "stderr": ""},
            "sealed_diagnostics": {"tests": [True, True]}}
        with ExitStack() as stack:
            factory = stack.enter_context(patch("models.create_backend", return_value=backend))
            stack.enter_context(patch("evaluation.preflight_local", return_value={"ready": True}))
            stack.enter_context(patch("evaluation.grade_final", return_value=verdict))
            evaluation_worker.run(args)
        return factory

    def test_generation_and_full_verdict_survive_and_resume_skips_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, output, row, config = self.fixture(Path(temporary))
            backend = FakeBackend()
            self.run_fixture(args, backend)
            result = read_json(output / "cases" / config["id"] / (task_id(row)+".json"))
            self.assertTrue(result["passed"])
            self.assertEqual(result["generation_seconds"], .25)
            self.assertEqual(result["forward_calls"], 1)
            receipt = read_json(result["generation_receipt"])
            self.assertEqual(receipt["generation"]["diagnostics"], backend.generated["diagnostics"])
            self.assertEqual(receipt["generation"]["generated_token_ids"], [12, 13, 14])
            self.assertEqual(receipt["generation"]["raw_text_before_terminal"], backend.generated["raw_text_before_terminal"])
            self.assertEqual(receipt["raw_output"], backend.generated["text"])
            self.assertEqual(file_sha(result["generation_receipt"]), result["generation_receipt_sha256"])
            first = events(output)
            verdict = next(e["details"]["verdict"] for e in first if e["event"] == "evaluation.grading_end")
            self.assertEqual(verdict["sealed_diagnostics"], {"tests": [True, True]})
            self.assertTrue(backend.closed)
            self.run_fixture(args, backend).assert_not_called()
            names = [e["event"] for e in events(output)]
            self.assertEqual(names.count("evaluation.generation_start"), 1)
            self.assertIn("evaluation.case_recovered", names)
            self.assertIn("evaluation.candidate_skipped", names)

    def test_generated_code_is_durable_even_when_grading_infrastructure_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, output, row, config = self.fixture(Path(temporary))
            backend = FakeBackend()
            with self.assertRaisesRegex(RuntimeError, "infrastructure"):
                self.run_fixture(args, backend, {"passed": False, "status": "infra_error",
                    "error": "compiler unavailable"})
            receipt, = (output / "generations" / config["id"]).glob(task_id(row)+"-*.json")
            self.assertEqual(read_json(receipt)["raw_output"], backend.generated["text"])
            self.assertFalse((output / "cases" / config["id"] / (task_id(row)+".json")).exists())
            self.assertTrue(backend.closed)
            original_sha = file_sha(receipt)
            self.run_fixture(args, FakeBackend())
            self.assertEqual(file_sha(receipt), original_sha)
            self.assertEqual(len(list((output / "generations" / config["id"]).glob(task_id(row)+"-*.json"))), 2)

    def test_known_generation_failure_keeps_partial_diagnostics_and_failure_outcome(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, output, row, config = self.fixture(Path(temporary))
            error = evaluation_worker.ForwardBudgetExceeded("fixture budget")
            error.generation_diagnostics = {"measurement_complete": False, "model_forward_passes": 1}
            self.run_fixture(args, FakeBackend(error))
            result = read_json(output / "cases" / config["id"] / (task_id(row)+".json"))
            self.assertFalse(result["passed"])
            self.assertEqual(result["status"], "ForwardBudgetExceeded")
            self.assertEqual(read_json(result["generation_receipt"])["partial_generation_diagnostics"],
                             error.generation_diagnostics)
            failure = next(e for e in events(output) if e["event"] == "evaluation.generation_failure")
            self.assertTrue(failure["details"]["handled"])
            self.assertIn("ForwardBudgetExceeded", failure["details"]["traceback"])

    def test_grading_exception_records_traceback_without_changing_exception(self):
        with tempfile.TemporaryDirectory() as temporary:
            failure = RuntimeError("grader fixture")
            def broken(*args, **kwargs):
                raise failure
            with self.assertRaises(RuntimeError) as caught:
                evaluation_worker.observed_grade(temporary, SimpleNamespace(final_evaluator="fixture"),
                    "code", broken, 4, {"task_key": "task"})
            self.assertIs(caught.exception, failure)
            self.assertIn("grader fixture", events(temporary)[-1]["details"]["traceback"])


class TrainingWorkerLoggingTests(unittest.TestCase):
    def args(self, root):
        return ["--train-jsonl", str(root / "train.jsonl"), "--output", str(root / "output"),
                "--deadline-unix", str(time.time()+300)]

    def test_all_rank_process_result_preserves_completed_and_interrupted_exit_codes(self):
        for code in (0, 75):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                with patch.object(training_worker, "DistributedContext", return_value=SimpleNamespace(rank=2)), \
                     patch.object(training_worker, "run", return_value=code), \
                     patch.object(training_worker.dist, "is_initialized", return_value=False):
                    self.assertEqual(training_worker.main(self.args(root)), code)
                end = events(root / "output")[-1]
                self.assertEqual(end["event"], "training.process_end")
                self.assertEqual(end["details"]["rank"], 2)
                self.assertEqual(end["details"]["exit_code"], code)

    def test_rank_failure_records_full_traceback_and_preserves_failure_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            failure = FloatingPointError("nonfinite fixture")
            with patch.object(training_worker, "DistributedContext", return_value=SimpleNamespace(rank=0)), \
                 patch.object(training_worker, "run", side_effect=failure), \
                 patch.object(training_worker.dist, "is_initialized", return_value=False):
                with self.assertRaises(FloatingPointError) as caught:
                    training_worker.main(self.args(root))
            self.assertIs(caught.exception, failure)
            self.assertEqual(read_json(root / "output/training_state.json")["status"], "failed")
            self.assertEqual(read_json(root / "output/failure-rank-0000.json")["rank"], 0)
            records = events(root / "output")
            record = next(e for e in records if e["event"] == "training.process_failure")
            self.assertIn("FloatingPointError", record["details"]["traceback"])
            self.assertEqual(records[-1]["event"], "training.state")

    def test_state_history_preserves_overwritten_states(self):
        with tempfile.TemporaryDirectory() as temporary:
            training_worker.record_training_state(temporary, {"status": "loading"})
            training_worker.record_training_state(temporary, {"status": "running", "global_step": 1})
            self.assertEqual(read_json(Path(temporary) / "training_state.json")["status"], "running")
            states = [e["details"]["state"] for e in events(temporary) if e["event"] == "training.state"]
            self.assertEqual(states, [{"status": "loading"}, {"status": "running", "global_step": 1}])

    def test_checkpoint_events_bind_committed_manifest_and_reused_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def save_adapter(path, **kwargs):
                (path / "adapter_model.safetensors").write_bytes(b"fixture weights")
                write_json(path / "adapter_config.json", {"fixture": True})
            model = SimpleNamespace(save_pretrained=save_adapter)
            tokenizer = SimpleNamespace(save_pretrained=lambda path: write_json(path / "tokenizer_config.json", {}))
            stateful = SimpleNamespace(state_dict=lambda: {"value": 1})
            args = SimpleNamespace(output=root, base_model="fixture", revision="pinned")
            context = SimpleNamespace(rank=0, world_size=1, device=torch.device("cpu"), barrier=lambda: None)
            counters = {"global_step": 1, "epoch": 0, "next_batch": 1}
            before = torch.get_rng_state().clone()
            first = training_worker.save_checkpoint(args, context, model, tokenizer, stateful,
                                                     stateful, counters, {"data": "fixture"})
            again = training_worker.save_checkpoint(args, context, model, tokenizer, stateful,
                                                     stateful, counters, {"data": "fixture"})
            self.assertEqual(first, again)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            ended = [e["details"] for e in events(root) if e["event"] == "training.checkpoint_end"]
            self.assertEqual([row["reused"] for row in ended], [False, True])
            self.assertEqual(ended[0]["manifest_sha256"], file_sha(first / "checkpoint_manifest.json"))


if __name__ == "__main__":
    unittest.main()
