from pathlib import Path
import tempfile
import unittest

from iterative_tuning import (AR, CPGRPO, adapter_identity, candidate, digest,
    freeze_json, initial_candidates, round_candidates, select_results,
    shard_tasks, summarize, task_id, validate_candidate)


class IterationContracts(unittest.TestCase):
    def tasks(self):
        return [dict(dataset=f, family=f, language=l, task_id=str(i), source_id=f+str(i))
                for f in ("humaneval", "mbpp") for l in ("py", "cpp", "rs", "js") for i in range(9)]

    def test_all_tasks_appear_in_exactly_one_balanced_shard(self):
        tasks = self.tasks()
        shards = [shard_tasks(tasks, i) for i in range(4)]
        observed = [task_id(t) for shard in shards for t in shard]
        self.assertEqual(len(observed), len(set(observed)))
        self.assertEqual(set(observed), {task_id(t) for t in tasks})
        for language in ("py", "cpp", "rs", "js"):
            sizes = [sum(t["language"] == language for t in shard) for shard in shards]
            self.assertLessEqual(max(sizes)-min(sizes), 2)

    def test_frozen_artifacts_cannot_be_rebound(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/"plan.json"
            freeze_json(p, {"train": "a", "validation": "b"})
            freeze_json(p, {"train": "a", "validation": "b"})
            with self.assertRaises(ValueError):
                freeze_json(p, {"train": "b", "validation": "a"})

    def test_incomplete_adapter_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/"adapter_config.json").write_text("{}")
            with self.assertRaises(ValueError):
                adapter_identity(directory)

    def test_candidate_search_is_bounded_and_valid(self):
        configs = initial_candidates()
        self.assertEqual(len(configs), 15)
        self.assertEqual(len({c["id"] for c in configs}), len(configs))
        for cfg in configs:
            validate_candidate(cfg)
        for index in range(1, 9):
            generated = round_candidates(configs[-1], None, index)
            self.assertLessEqual(len(generated), 7)
            for cfg in generated:
                validate_candidate(cfg)

    def test_speed_alone_does_not_replace_ar(self):
        ar = dict(cases=100, pass_rate=.8, mean_generation_seconds=2, candidate=candidate(AR, decoder="ar"))
        slow_bad = dict(cases=100, pass_rate=.2, mean_generation_seconds=1, candidate=candidate(CPGRPO))
        result = select_results([slow_bad], ar)
        self.assertEqual(result["deployment"], ar)
        self.assertFalse(result["goal_screen_passed"])
        good = dict(cases=100, pass_rate=.79, mean_generation_seconds=1.8, candidate=candidate(CPGRPO))
        result = select_results([slow_bad, good], ar)
        self.assertEqual(result["deployment"], good)

    def test_incomplete_cohort_and_infra_failures_are_not_accuracy(self):
        tasks = self.tasks()[:2]
        config = candidate(CPGRPO)
        rows = [dict(task_key=task_id(t), candidate_id=config["id"], passed=True,
                     infrastructure_error=False, generation_seconds=1, language=t["language"]) for t in tasks]
        self.assertEqual(summarize(rows, tasks, config)["pass_rate"], 1)
        with self.assertRaises(ValueError):
            summarize(rows[:1], tasks, config)
        with self.assertRaises(ValueError):
            summarize(rows+rows[:1], tasks, config)
        with self.assertRaises(ValueError):
            summarize(rows, tasks+tasks[:1], config)
        rows[0]["infrastructure_error"] = True
        with self.assertRaises(ValueError):
            summarize(rows, tasks, config)

    def test_full_candidate_identity_changes_with_decoding_or_adapter(self):
        a = candidate(CPGRPO)
        b = candidate(CPGRPO, temperature=0)
        c = candidate(CPGRPO, adapter={"path": "/adapter", "sha256": {"weights": "x"}})
        self.assertEqual(len({a["id"], b["id"], c["id"]}), 3)


if __name__ == "__main__":
    unittest.main()
