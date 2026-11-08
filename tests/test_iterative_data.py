"""Leakage, source identity, strict evaluator, and offline preparation contracts."""
from copy import deepcopy
from dataclasses import asdict, replace
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from datasets_code import Task
import iterative_data as data


def task(source="MBPP/11", language="js", *, plus=False):
    family, number = source.split("/")
    family = family.lower()
    suite = (("humaneval" if family == "humaneval" else "mbpp") + ("_plus" if plus else "")) if language == "py" else (
        "multiple_" + ("mbpp_" if family == "mbpp" else "") + language)
    return Task(suite, language, source if language == "py" else source.replace("/", "_") + "_f",
                int(number), "Implement f.", "SEALED_TEST_SENTINEL", "f", family=family,
                source_id=source, final_evaluator="evalplus" if plus else "legacy_assert")


def teacher_pair(t):
    code = "function f(x) { return x; }\n"
    case = {"decoder": "ar", "dataset": t.dataset, "language": t.language, "task_id": t.task_id,
            "source_id": t.source_id, "repeat": 0, "seed": 42, "model": "fixture/teacher",
            "final_pass": True, "final_status": "passed", "case_completed": True,
            "failure_kind": None, "model_identity_verified": True, "measurement_complete": True,
            "attempts": 1, "task_sha256": data.digest(asdict(t)), "messages_sha256": "message-hash",
            "grading": {"final": {"passed": True, "sealed": True}}}
    run = {key: case[key] for key in ("decoder", "dataset", "language", "task_id", "repeat", "seed")}
    run.update(attempt=1, code=code, code_sha256=hashlib.sha256(code.encode()).hexdigest(),
               generation={"messages_sha256": "message-hash"})
    return case, run


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))


def cached_fixture(root):
    # Reuse complete offline cache shape, then add the canonical labels and
    # out-of-historical-test-scope rows that this workflow must include.
    from tests.test_codegen_independence import fixture_datasets
    fixture_datasets(root)
    cache = root/"datasets/function-cache"
    hp = cache/"HumanEval.jsonl.gz"
    rows = [json.loads(line) for line in gzip.decompress(hp.read_bytes()).splitlines()]
    for row in rows:
        row["canonical_solution"] = "    return x\n"
    hp.write_bytes(gzip.compress("".join(json.dumps(row)+"\n" for row in rows).encode()))
    mp = cache/"mbpp.jsonl"
    rows = [json.loads(line) for line in mp.read_text().splitlines()]
    rows.extend({"task_id": n, "text": "Return x.", "code": "def f(x):\n    return x\n",
                 "test_list": ["assert f(1) == 1"]} for n in (2, 700))
    write_rows(mp, rows)
    for language in ("rs", "js", "cpp"):
        p = cache/f"multiple_mbpp_{language}_0.json"
        doc = json.loads(p.read_text())
        extra = deepcopy(doc["rows"][0])
        extra["row_idx"] = 1
        extra["row"]["name"] = "mbpp_700_f"
        doc["rows"].append(extra); doc["num_rows_total"] = 2
        p.write_text(json.dumps(doc))


class IterativeDataTests(unittest.TestCase):
    def test_original_plus_identity_and_cross_family_mismatch(self):
        self.assertEqual(data.normalize_source_id("Mbpp/002"), "MBPP/2")
        self.assertEqual(data.source_from_identity({"task_id": "mbpp_700_f"}), "MBPP/700")
        with self.assertRaisesRegex(ValueError, "disagrees"):
            data.source_from_identity({"source_id": "MBPP/2", "task_id": "mbpp_3_f"})
        with self.assertRaisesRegex(ValueError, "family mismatch"):
            data.normalize_source_id("MBPP/2", "humaneval")

    def test_plus_priority_preserves_full_original_task_and_strict_grader(self):
        native, plus = task(language="py"), task(language="py", plus=True)
        plus = replace(plus, evaluator_metadata={"raw_sha256": "strict-original"}, test_source="unchanged")
        selected, excluded = data.deduplicate_tasks([native, plus, task(language="js")])
        chosen = next(t for t in selected if t.language == "py")
        self.assertIs(chosen, plus)
        self.assertEqual(asdict(chosen), asdict(plus))
        self.assertEqual(excluded[0]["excluded_dataset"], "mbpp")
        with self.assertRaisesRegex(ValueError, "Duplicate records"):
            data.deduplicate_tasks([plus, plus])

    def test_partition_is_source_disjoint_stratified_and_frozen_by_identity(self):
        tasks = []
        exposed = set()
        for family in ("HumanEval", "MBPP"):
            for n in range(40):
                languages = ("py", "rs", "js", "cpp") if n < 20 else ("py",)
                for language in languages:
                    tasks.append(task(f"{family}/{n}", language))
                if n < 20:
                    exposed.add(f"{family}/{n}")
        split = data.partition_sources(tasks, exposed_sources=exposed)
        self.assertEqual({k: len(v) for k,v in split["source_ids"].items()},
                         {"train": 64, "validation": 8, "holdout": 8})
        self.assertEqual(split, data.partition_sources(list(reversed(tasks)), exposed_sources=exposed))
        changed = [replace(t, prompt="different", test_source="different") for t in tasks]
        self.assertEqual(split, data.partition_sources(changed, exposed_sources=exposed))
        for source in {t.source_id for t in tasks}:
            self.assertEqual(sum(source in values for values in split["source_ids"].values()), 1)
        self.assertEqual(len(split["previously_exposed_by_split"]["holdout"]), 4)
        self.assertFalse(split["globally_unseen_holdout_claim"])
        for stratum in split["strata"]:
            self.assertEqual(stratum["split_sources"]["holdout"], 2)
            self.assertEqual(stratum["split_sources"]["validation"], 2)

    def test_holdout_prefers_unexposed_inside_language_coverage_stratum(self):
        tasks = [task(f"MBPP/{i}") for i in range(20)]
        exposed = {f"MBPP/{i}" for i in range(18)}
        split = data.partition_sources(tasks, exposed_sources=exposed)
        self.assertEqual(set(split["source_ids"]["holdout"]), {"MBPP/18", "MBPP/19"})

    def test_prompt_and_completion_have_consistent_fenced_contract(self):
        messages = data.build_iteration_messages(task())
        self.assertEqual(data.build_iterative_messages(task()), messages)
        self.assertIn("one Markdown code block tagged javascript", messages[-1]["content"])
        self.assertNotIn("or Markdown fences", messages[-1]["content"])
        self.assertEqual(data.fenced_completion("js", "function f() {}\n"),
                         "\n```javascript\nfunction f() {}\n```")
        with self.assertRaises(ValueError):
            data.fenced_completion("js", "```javascript\nfunction f() {}\n```")

    def test_teacher_requires_exact_passing_case_code_and_task_binding(self):
        t = task(); case, run = teacher_pair(t)
        self.assertEqual(data.validate_teacher_pair(t, case, run), run["code"])
        mutations = [("task_sha256", "wrong"), ("model_identity_verified", False),
                     ("final_pass", False), ("attempts", 2), ("messages_sha256", "wrong")]
        for field, value in mutations:
            altered = deepcopy(case); altered[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                data.validate_teacher_pair(t, altered, run)
        altered = deepcopy(run); altered["code"] += "// tampered"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            data.validate_teacher_pair(t, case, altered)

    def test_teacher_payloads_from_nontraining_sources_are_never_selected(self):
        train, held = task("MBPP/11"), task("MBPP/12")
        good_case, good_run = teacher_pair(train)
        held_case, held_run = teacher_pair(held)
        # Invalid held-out payload must be filtered before validation/code use.
        held_run["code"] = {"oracle": "NEVER_TRAIN_THIS"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_rows(root/"case_results.jsonl", [held_case, good_case])
            write_rows(root/"runs.jsonl", [held_run, good_run])
            labels, report = data.collect_teacher_examples([train], [root])
        self.assertEqual(set(labels), {("MBPP/11", "js")})
        self.assertNotIn("NEVER_TRAIN_THIS", json.dumps(labels[("MBPP/11", "js")]))
        self.assertEqual(report["nontraining_payloads_used"], 0)

    def test_training_validation_rejects_split_leak_and_test_payload(self):
        t = task(); example = data._example(t, "function f() {}", {"kind": "fixture"})
        data.validate_training_examples([example], [t])
        with self.assertRaisesRegex(ValueError, "outside train"):
            data.validate_training_examples([example], [task("MBPP/12")])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            data.validate_training_examples([example, example], [t])
        example["plus_input"] = [[1, 2]]
        with self.assertRaisesRegex(ValueError, "test/oracle"):
            data.validate_training_examples([example], [t])

    def test_offline_full_cached_scope_accounting_and_frozen_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); cached_fixture(root); output = root/"prepared"
            with patch("urllib.request.urlopen", side_effect=AssertionError("Network forbidden")), \
                 patch("subprocess.run", side_effect=AssertionError("Execution forbidden")):
                report = data.prepare_iterative_data(root, output, teacher_roots=[])
                self.assertTrue(report["proof"]["source_splits_disjoint"])
                self.assertFalse(report["proof"]["validation_or_holdout_payloads_in_training"])
                self.assertEqual(report["inventory"]["raw_by_dataset"]["mbpp"], 502)
                self.assertEqual(report["inventory"]["raw_by_dataset"]["multiple_mbpp_js"], 2)
                manifests = {name: [Task(**row) for row in data._read_jsonl(output/f"{name}_tasks.jsonl")]
                             for name in ("train", "validation", "holdout")}
                all_tasks = sum(manifests.values(), [])
                self.assertIn(("MBPP/700", "js"), {(t.source_id,t.language) for t in all_tasks})
                examples = data._read_jsonl(output/"training_examples.jsonl")
                data.validate_training_examples(examples, manifests["train"])
                self.assertNotIn("test_source", examples[0])
                with patch.object(data, "discover_exposure", side_effect=AssertionError("Must preserve frozen split")):
                    self.assertEqual(data.prepare_iterative_data(root, output, teacher_roots=[]), report)
                with self.assertRaisesRegex(ValueError, "frozen split"):
                    data.prepare_iterative_data(root, output, seed=7, teacher_roots=[])
                with (output/"training_examples.jsonl").open("a") as stream:
                    stream.write("\n")
                with self.assertRaisesRegex(ValueError, "artifact changed"):
                    data.prepare_iterative_data(root, output, teacher_roots=[])


if __name__ == "__main__":
    unittest.main()
