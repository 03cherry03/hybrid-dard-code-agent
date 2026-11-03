"""Mock integration tests: tests model orchestration and JSONL/CSV persistence offline."""
import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
import sys

from datasets_code import Task
import main as bench


class FakeBackend:
    load_count = 0
    def __init__(self, **kwargs):
        FakeBackend.load_count += 1
        self.model_load_s = 0.0
        self.counter = 0
    def generate(self, language, prompt, decoder, *, seed, num_new_tokens, steps, dard_config,
                 temperature, top_p, old_code, error):
        self.counter += 1
        # First generation fails, next generation fixes it
        code = "def f(x):\n    return x + 1\n" if not error else "def f(x):\n    return x * 2\n"
        return {"text": code, "wall_s": .02, "prompt_tokens": 20, "output_tokens": 50,
                "model_forward_passes": 3, "forced_tokens": 0}


class TestEndToEnd(unittest.TestCase):
    def test_runner_repair_resume_with_mocks(self):
        task = Task("humaneval", "py", "HumanEval/0", 0, "def f(x): pass", "")
        fake_module = ModuleType("models")
        fake_module.DiffuCoderBackend = FakeBackend
        def simulate_test(_task, code, _runner, _timeout):
            passed = "x * 2" in code
            return {"passed": passed, "status": "passed" if passed else "test_failed",
                    "feedback": "assertion failed" if not passed else "", "test_wall_s": .01}
        with tempfile.TemporaryDirectory() as temp:
            args = ["--suites", "humaneval", "--limit", "1", "--runner", "docker",
                    "--repairs", "1", "--decoders", "standard,dard", "--output", str(Path(temp)/"out")]
            with patch.dict(sys.modules, {"models": fake_module}), \
                 patch.object(bench, "preflight_docker", return_value=None), \
                 patch.object(bench, "load_tasks", return_value=[task]), \
                 patch.object(bench, "test_task", side_effect=simulate_test):
                self.assertEqual(bench.main(args), 0)
                results_path = Path(temp)/"out"/"case_results.jsonl"
                rows = [json.loads(x) for x in results_path.read_text().splitlines()]
                self.assertEqual(len(rows), 2)
                self.assertTrue(all(r["first_pass"] is False and r["final_pass"] is True for r in rows))
                self.assertEqual(bench.main(args), 0)  # resume without GPU
                rows_2 = results_path.read_text().splitlines()
                self.assertEqual(len(rows_2), 2)
                summary = json.loads((Path(temp)/"out"/"summary.json").read_text())
                self.assertEqual(summary["groups"]["standard/humaneval/py"]["repair_success_given_initial_failure"],1.0)

if __name__ == "__main__":
    unittest.main()
