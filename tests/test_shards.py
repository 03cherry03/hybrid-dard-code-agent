"""Offline coverage and balance contracts for independent GPU task partitions."""
from collections import Counter
from types import SimpleNamespace
import unittest

from datasets_code import Task
from experiment_shards import select_shard, validate_shard
from repository_agent import RepoTask


class ShardTests(unittest.TestCase):
    def test_full_function_catalog_has_balanced_disjoint_complete_partitions(self):
        counts = [164, 500, 156, 161, 161, 208, 235, 235, 164, 378]
        tasks = [SimpleNamespace(dataset=f"suite-{suite:02d}", language="py", task_id=f"{suite}/{i}")
                 for suite, count in enumerate(counts) for i in range(count)]
        shards = [select_shard(tasks, 4, i) for i in range(4)]
        self.assertEqual([len(shard) for shard in shards], [591, 591, 590, 590])
        identities = [task.task_id for shard in shards for task in shard]
        self.assertEqual(len(identities), len(set(identities)))
        self.assertEqual(set(identities), {task.task_id for task in tasks})
        for suite in range(len(counts)):
            per_shard = [sum(task.dataset == f"suite-{suite:02d}" for task in shard) for shard in shards]
            self.assertLessEqual(max(per_shard) - min(per_shard), 1)

    def test_interleaved_input_preserves_order_and_is_deterministic(self):
        tasks = [SimpleNamespace(dataset="repo", language=language, task_id=i)
                 for i, language in enumerate(["rust", "py", "java", "py", "rust", "java", "py"])]
        for shard_index in range(3):
            first = select_shard(tasks, 3, shard_index)
            self.assertEqual(first, select_shard(iter(tasks), 3, shard_index))
            self.assertEqual([task.task_id for task in first], sorted(task.task_id for task in first))
        totals = Counter(task.language for task in tasks)
        self.assertEqual(totals, sum((Counter(task.language for task in select_shard(tasks, 3, i))
                                     for i in range(3)), Counter()))

    def test_shared_helper_accepts_function_and_repository_tasks_without_mutation(self):
        tasks = [Task("humaneval", "py", "0", 0, "prompt", "tests"),
                 RepoTask("swe", "instance", "org/repo", "a" * 40, "issue", "rust", "image")]
        original = list(tasks)
        self.assertEqual(select_shard(tasks), original)
        self.assertIs(select_shard(tasks, 2, 0)[0], tasks[0])
        self.assertIs(select_shard(tasks, 2, 1)[0], tasks[1])
        self.assertEqual(tasks, original)

    def test_invalid_or_empty_partition_never_becomes_all_tasks(self):
        for num_shards, shard_index in [(0, 0), (-1, 0), (True, 0), (2.0, 0),
                                         (2, -1), (2, 2), (2, True), (2, .5)]:
            with self.subTest(num_shards=num_shards, shard_index=shard_index), self.assertRaises(ValueError):
                validate_shard(num_shards, shard_index)
        with self.assertRaisesRegex(ValueError, "empty"):
            select_shard([])
        with self.assertRaisesRegex(ValueError, "empty"):
            select_shard([SimpleNamespace(dataset="a", language="py")], 4, 3)
        with self.assertRaisesRegex(ValueError, "dataset and language"):
            select_shard([SimpleNamespace(dataset="a")])


if __name__ == "__main__":
    unittest.main()
