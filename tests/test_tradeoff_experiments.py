"""Campaign contracts: paired contexts, sealed selection and blind grading."""
import copy
import json
import time
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack

from datasets_code import Task
from iterative_tuning import digest, shard_tasks
from scripts.iterative_eval_worker import candidate_order, extract_for_grading, ForwardBudgetExceeded
from scripts.tradeoff_experiments import (CONTEXTS, choose_finalists, configured,
                                         make_grid, shared_context, smoke_candidates, selection_rows, execute,
                                         make_smoke_tasks)


ADAPTER = {"path": "/fixture/adapter", "sha256": {"adapter_config.json": "a", "adapter_model.safetensors": "b"}}


class TradeoffDesignTests(unittest.TestCase):
    def test_smoke_cannot_leave_a_gpu_empty_when_language_spans_families(self):
        rows = []
        for language in ("py", "js", "cpp", "rs"):
            for family, count in (("small", 2), ("large", 5)):
                for i in range(count):
                    rows.append({"dataset":family, "family":family, "language":language,
                                 "task_id":str(i), "source_id":family+str(i)})
        selected = make_smoke_tasks(rows)
        self.assertEqual(len(selected), 16)
        for index in range(4):
            self.assertEqual({t["language"] for t in shard_tasks(selected,index)}, {"py","js","cpp","rs"})
            self.assertEqual(len(shard_tasks(selected,index)), 4)

    def test_controller_freezes_selection_before_descriptive_holdout_and_uses_three_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            (data / "validation_tasks.jsonl").write_text('{}\n'*4)
            (root / "audit").mkdir()
            (root / "audit/grading_audit.json").write_text(json.dumps({"passed": True,
                "source_contract": {"source_sha256": {"fixture": "same"}}}))
            configs = make_grid(ADAPTER)
            plan = {"output": str(root), "data": str(data), "data_root": str(root),
                "deadline_unix": time.time()+3600, "seed": 1, "repeat_seeds": [2,3],
                "failure_limit": 3, "initial_candidates": configs,
                "smoke_candidates": smoke_candidates(configs), "source_sha256": {key:"same" for key in
                    ("grading_audit.py", "evaluation.py", "function_benchmarks.py", "scripts/iterative_eval_worker.py")}}
            (root / "iteration_plan.json").write_text(json.dumps(plan))
            calls = []
            def fake_evaluate(p, label, candidates, split="validation", **kwargs):
                calls.append((label, p["seed"], split))
                if split == "holdout":
                    self.assertTrue((root / "final_selection.json").is_file())
                result = [{"candidate":c, "cases":16 if split == "smoke" else 4,
                           "passed":2, "pass_rate":.5, "mean_generation_seconds":1.} for c in candidates]
                (root / label).mkdir(exist_ok=True)
                (root / label / "results.json").write_text(json.dumps(result))
                return result
            def fake_analyze(p, phases, name, candidate_ids=None):
                selected = [c for c in configs if candidate_ids is None or c["id"] in candidate_ids]
                n = 4*len(phases)
                return {"summaries":[{"candidate":c, "candidate_id":c["id"], "expected_cases":n,
                    "passed":n//2, "pass_rate":.5, "mean_generation_seconds":1., "valid_for_selection":True}
                    for c in selected]}
            with ExitStack() as stack:
                for name, replacement in (("verify_plan", lambda _:None), ("update", lambda *a,**k:None),
                    ("evaluate", fake_evaluate), ("analyze", fake_analyze),
                    ("review_smoke", lambda _:{"passed": True})):
                    stack.enter_context(patch("scripts.tradeoff_experiments."+name, replacement))
                execute(plan)
            final = json.loads((root / "final_selection.json").read_text())
            self.assertTrue(all(r["cases"] == 12 for r in final["all_finalists"]))
            self.assertEqual([seed for label,seed,split in calls if split == "validation"], [1,2,3])
            self.assertFalse(final["automatic_deployment_change"])
            self.assertTrue((root / "tradeoff_result.json").is_file())

    def test_selection_refuses_invalid_or_missing_artifacts(self):
        config = configured()
        summary = {"candidate": config, "candidate_id": config["id"], "expected_cases": 4,
                   "passed": 2, "pass_rate": .5, "mean_generation_seconds": 1.,
                   "valid_for_selection": True}
        self.assertEqual(selection_rows({"summaries": [summary]}, [config["id"]])[0]["cases"], 4)
        with self.assertRaises(ValueError):
            selection_rows({"summaries": [{**summary, "valid_for_selection": False}]}, [config["id"]])
        with self.assertRaises(ValueError):
            selection_rows({"summaries": []}, [config["id"]])

    def test_every_shared_context_has_both_decoders_and_paper_grid_is_complete(self):
        grid = make_grid(ADAPTER)
        self.assertEqual(len(grid), len({c["id"] for c in grid}))
        contexts = {}
        for c in grid:
            if c["decoder"] != "ar":
                contexts.setdefault(shared_context(c), set()).add(c["decoder"])
        self.assertEqual(len(contexts), 2*len(CONTEXTS))
        self.assertTrue(all(v == {"standard", "dard"} for v in contexts.values()))
        for adapter in (None, ADAPTER):
            thresholds = {(c["dard"]["tau_c"], c["dard"]["tau_u"]) for c in grid
                if c["decoder"] == "dard" and c["adapter"] == adapter and c["tokens"] == 256
                and c["prefix"] == "none" and c["dard"]["temperature"] == 0
                and c["dard"]["steps_per_block"] == 64}
            self.assertEqual(thresholds, {(c,u) for c in (.4,.5,.6) for u in (.6,.7,.8)})
        self.assertEqual(make_grid(ADAPTER), grid)

    def test_smoke_covers_completion_policies_adapter_equality_and_four_languages_manifest_contract(self):
        configs = smoke_candidates(make_grid(ADAPTER))
        self.assertEqual(len(configs), 6)
        self.assertEqual({c["decoder"] for c in configs}, {"ar", "standard", "dard"})
        self.assertEqual({c["dard"].get("completion_policy") for c in configs if c["decoder"] == "dard"},
                         {"legacy_force", "verified_only", "force_then_verify"})
        self.assertTrue(any(c["adapter"] and c["dard"]["tau_c"] == c["dard"]["tau_u"] for c in configs))

    def test_shortlist_retains_comparison_pairs_and_never_exceeds_bound(self):
        grid = make_grid(ADAPTER)
        results = []
        for index, c in enumerate(grid):
            results.append({"candidate": c, "cases": 273, "passed": index % 110,
                            "pass_rate": (index % 110)/273, "mean_generation_seconds": 1+(index % 13)})
        chosen = choose_finalists(results)
        self.assertLessEqual(len(chosen), 18)
        self.assertEqual(chosen, choose_finalists(list(reversed(results))))
        for c in chosen:
            if c["decoder"] != "ar":
                self.assertTrue(any(shared_context(c) == shared_context(other)
                    and c["decoder"] != other["decoder"] for other in chosen))

    def test_order_is_permutation_and_keeps_model_residency(self):
        configs = make_grid(ADAPTER)
        baseline = candidate_order(configs, 0, 42)
        self.assertEqual(baseline, configs)
        orders = [candidate_order(configs, i, 42, True) for i in range(4)]
        for order in orders:
            self.assertEqual({c["id"] for c in order}, {c["id"] for c in configs})
            self.assertEqual(len(order), len(configs))
            keys = [(c["model"], digest(c["adapter"])) for c in order]
            transitions = sum(a != b for a,b in zip(keys, keys[1:]))
            self.assertEqual(transitions, 2)
        self.assertNotEqual(orders[0], orders[1])

    def test_blind_extraction_preserves_code_from_unclosed_fence(self):
        task = Task(dataset="humaneval", family="humaneval", language="py", task_id="fixture/1",
            number=1, source_id="fixture/1", prompt="Write identity", required_name="identity",
            test_source="assert identity(1) == 1")
        code, receipt = extract_for_grading(task, "def identity(x):\n    return x", "\n```python\n",
                                          {"extraction_version": "v2", "grading_timeout": 5})
        self.assertEqual(code, "def identity(x):\n    return x")
        self.assertIsInstance(receipt, dict)
        self.assertTrue(ForwardBudgetExceeded.abort_dard_generation)


if __name__ == "__main__":
    unittest.main()
