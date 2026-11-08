"""Teacher labels require train identities, pinned provenance and passing code."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from datasets_code import Task
from iterative_data import build_iterative_messages, fenced_completion
from iterative_tuning import AR, candidate, digest, file_sha, task_id, write_json
from scripts.iterative_teacher import augment_training_corpus, missing_tasks


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def task(number, language="js"):
    return asdict(Task(dataset="multiple_mbpp_js", language=language, task_id=f"mbpp_{number}_f",
        number=number, prompt="function f(x) {", test_source="assert.equal(f(1), 2)",
        required_name="f", source_id=f"MBPP/{number}", family="mbpp"))


class TeacherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name); self.data = self.root / "data"; self.data.mkdir()
        self.train = [task(1), task(2)]; self.validation = [task(3)]; self.holdout = [task(4)]
        first = self.train[0]
        self.original = [{"example_id": "MBPP/1/js", "source_id": "MBPP/1", "language": "js",
            "messages": build_iterative_messages(first), "completion": fenced_completion("js", "function f(x) {return x+1;}"),
            "provenance": {"task_sha256": digest(first), "kind": "fixture_verified"}}]
        for split, rows in [("train", self.train), ("validation", self.validation), ("holdout", self.holdout)]:
            write_jsonl(self.data / (split + "_tasks.jsonl"), rows)
        write_jsonl(self.data / "training_examples.jsonl", self.original)
        self.freeze_manifest()
        self.teacher = self.root / "teacher"; self.teacher.mkdir()
        write_jsonl(self.teacher / "tasks.jsonl", [self.train[1]])
        self.config = candidate(AR, decoder="ar", temperature=.3, forward_budget=512)
        self.plan = {"split": "train_teacher", "tasks_path": str(self.teacher / "tasks.jsonl"),
            "tasks_sha256": file_sha(self.teacher / "tasks.jsonl"), "candidates": [self.config]}
        self.update_plan()
        self.path = self.teacher / "shard-0/cases" / self.config["id"] / (task_id(self.train[1]) + ".json")
        self.record = {"split": "train_teacher", "task_key": task_id(self.train[1]),
            "task_sha256": digest(self.train[1]), "source_id": "MBPP/2", "language": "js", "task_id": "mbpp_2_f",
            "candidate_id": self.config["id"], "config_sha256": digest(self.config),
            "code": "function f(x) {return x+1;}", "passed": True, "status": "passed", "infrastructure_error": False}
        self.record["code_sha256"] = digest(self.record["code"])
        write_json(self.path, self.record)
        self.output = self.data / "training_augmented.jsonl"

    def freeze_manifest(self):
        names = ["train_tasks.jsonl", "validation_tasks.jsonl", "holdout_tasks.jsonl", "training_examples.jsonl"]
        write_json(self.data / "split_manifest.json", {"schema_version": 1,
            "source_assignment": {"MBPP/1": "train", "MBPP/2": "train", "MBPP/3": "validation", "MBPP/4": "holdout"},
            "source_ids": {"train": ["MBPP/1", "MBPP/2"], "validation": ["MBPP/3"], "holdout": ["MBPP/4"]},
            "artifact_sha256": {name: file_sha(self.data / name) for name in names}})

    def update_plan(self):
        write_json(self.teacher / "evaluation_plan.json", self.plan)
        write_json(self.teacher / "shard-0/binding.json", {"plan_sha256": file_sha(self.teacher / "evaluation_plan.json"), "shard": 0})

    def test_adds_only_missing_verified_label_and_preserves_original_idempotently(self):
        self.assertEqual(missing_tasks(self.data), [self.train[1]])
        receipt = augment_training_corpus(self.data, self.teacher, self.output)
        rows = [json.loads(line) for line in self.output.read_text().splitlines()]
        self.assertEqual(rows[0], self.original[0]); self.assertEqual(len(rows), 2)
        self.assertTrue(rows[1]["provenance"]["synthetic"])
        self.assertEqual(rows[1]["messages"], build_iterative_messages(self.train[1]))
        self.assertEqual(receipt["unlabeled_count"], 0)
        self.assertEqual(augment_training_corpus(self.data, self.teacher, self.output), receipt)

    def test_failed_teacher_remains_explicitly_unlabeled(self):
        self.record.update(passed=False, status="test_failed"); write_json(self.path, self.record)
        receipt = augment_training_corpus(self.data, self.teacher, self.output)
        self.assertEqual(receipt["new_verified_synthetic_labels"], 0)
        self.assertEqual(receipt["unlabeled"][0]["reason"], "teacher_failed")

    def test_missing_case_is_reported_without_inventing_a_label(self):
        self.path.unlink()
        receipt = augment_training_corpus(self.data, self.teacher, self.output)
        self.assertEqual(receipt["unlabeled"][0]["reason"], "not_evaluated")

    def test_holdout_task_in_teacher_plan_fails_closed(self):
        write_jsonl(self.teacher / "tasks.jsonl", self.holdout)
        self.plan["tasks_sha256"] = file_sha(self.teacher / "tasks.jsonl"); self.update_plan()
        with self.assertRaisesRegex(ValueError, "missing TRAIN"):
            augment_training_corpus(self.data, self.teacher, self.output)

    def test_validation_result_cannot_masquerade_as_train_result(self):
        self.record["split"] = "validation"; write_json(self.path, self.record)
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            augment_training_corpus(self.data, self.teacher, self.output)

    def test_changed_task_digest_config_or_code_is_rejected(self):
        for field in ("task_sha256", "config_sha256", "code_sha256"):
            with self.subTest(field=field):
                row = dict(self.record); row[field] = "changed"; write_json(self.path, row)
                with self.assertRaises(ValueError):
                    augment_training_corpus(self.data, self.teacher, self.output)

    def test_source_identity_mismatch_and_infrastructure_errors_are_rejected(self):
        for patch in ({"source_id": "MBPP/3"}, {"infrastructure_error": True}, {"passed": 1}):
            with self.subTest(patch=patch):
                write_json(self.path, {**self.record, **patch})
                with self.assertRaises(ValueError):
                    augment_training_corpus(self.data, self.teacher, self.output)

    def test_changed_frozen_data_and_original_overwrite_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "overwrite"):
            augment_training_corpus(self.data, self.teacher, self.data / "training_examples.jsonl")
        with (self.data / "train_tasks.jsonl").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            missing_tasks(self.data)

    def test_unpinned_teacher_model_is_rejected(self):
        self.plan["candidates"][0]["revision"] = "0" * 40; self.update_plan()
        with self.assertRaisesRegex(ValueError, "Unpinned"):
            augment_training_corpus(self.data, self.teacher, self.output)

    def test_existing_augmented_corpus_is_not_overwritten(self):
        augment_training_corpus(self.data, self.teacher, self.output)
        before = self.output.read_bytes()
        self.record["code"] = "function f(x) {return 1+x;}"
        self.record["code_sha256"] = digest(self.record["code"])
        write_json(self.path, self.record)
        with self.assertRaisesRegex(ValueError, "Immutable augmented"):
            augment_training_corpus(self.data, self.teacher, self.output)
        self.assertEqual(self.output.read_bytes(), before)

    def test_already_labeled_train_task_cannot_replace_existing_label(self):
        write_jsonl(self.teacher / "tasks.jsonl", [self.train[0]])
        self.plan["tasks_sha256"] = file_sha(self.teacher / "tasks.jsonl"); self.update_plan()
        with self.assertRaisesRegex(ValueError, "missing TRAIN"):
            augment_training_corpus(self.data, self.teacher, self.output)


if __name__ == "__main__":
    unittest.main()
