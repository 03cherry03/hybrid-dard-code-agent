"""Loader and native-runner regression tests using tiny local fixtures only."""
import ast
import gzip
import json
from pathlib import Path
import shutil
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import datasets_code as data
import evaluation as runner


def translated_row(lang="rs", family="humaneval", number=0):
    name = f"{'HumanEval' if family == 'humaneval' else 'mbpp'}_{number}_solve"
    prefixes = {
        "rs": "use std::collections::HashMap;\n/// supplied description\nfn solve(x: isize) -> isize {\n",
        "cpp": "#include<bits/stdc++.h>\nstruct Box { long value; };\n// supplied description\nlong solve(long x) {\n",
        "js": "// supplied description\nfunction solve(x) {\n",
    }
    tests = {
        "rs": "}\n\nfn main() {\n    let candidate = solve;\n    assert_eq!(candidate(1), 2);\n}\n",
        "cpp": "}\nint main() {\n    auto candidate = solve;\n    assert(candidate(1) == 2);\n}\n",
        "js": "const assert = require('node:assert');\nfunction test() {\n  let candidate = solve;\n  assert.deepEqual(candidate(1), 2);\n}\ntest();\n",
    }
    return {"name": name, "language": lang, "prompt": prefixes[lang], "tests": tests[lang]}


def page(rows, offset=0, total=None):
    return json.dumps({"rows": [{"row_idx": offset + i, "row": row, "truncated_cells": []}
                                for i, row in enumerate(rows)],
                       "num_rows_total": len(rows) if total is None else total,
                       "partial": False}).encode()


class DatasetTests(unittest.TestCase):
    def test_registry_covers_two_families_and_four_languages(self):
        expected = {"humaneval", "mbpp", "multiple_rs", "multiple_js", "multiple_cpp",
                    "multiple_mbpp_rs", "multiple_mbpp_js", "multiple_mbpp_cpp",
                    "humaneval_plus", "mbpp_plus", "bigcodebench_instruct"}
        self.assertEqual(set(data.SUITES), expected)
        self.assertEqual(set(data.LANGUAGE_NAMES), {"py", "rs", "js", "cpp"})
        for name, info in data.SUITES.items():
            if name.startswith("multiple_"):
                self.assertEqual(info["source_config"], info["family"] + "-" + info["language"])

    def test_mbpp_api_omits_reference_bodies_and_internal_helpers(self):
        reference = ("import math\n"
                     "def helper(x):\n    return 'SECRET_HELPER_BODY'\n"
                     "@decorator\n"
                     "def requested(x, /, count=2, *, scale=1):\n"
                     "    '''SECRET_DOCSTRING'''\n"
                     "    return helper(x) + 'SECRET_SOLUTION_BODY'\n")
        api, name = data._mbpp_api(reference, ["assert requested(3, scale=4) == 7"])
        self.assertEqual(name, "requested")
        self.assertIn("def requested(x, /, count=2, *, scale=1):", api)
        self.assertIn("pass", api)
        self.assertNotIn("SECRET", api)
        self.assertNotIn("helper", api)
        self.assertNotIn("decorator", api)
        self.assertNotIn("return", api)
        ast.parse(api)

    def test_mbpp_lambda_api_has_signature_only(self):
        api, name = data._mbpp_api("compute = lambda x, y=1: x * y + 987654",
                                   ["assert compute(2) == 2"])
        self.assertEqual(name, "compute")
        self.assertIn("def compute(x, y=1):", api)
        self.assertNotIn("987654", api)

    def test_mbpp_official_test_scope_and_public_api(self):
        rows = [{"task_id": i, "text": "Return the input.", "code": "def identity(value):\n    return value",
                 "test_list": ["assert identity(2) == 2"], "test_setup_code": ""}
                for i in range(1, 975)]
        payload = "\n".join(json.dumps(row) for row in rows).encode()
        with patch.object(data, "fetch_cached", return_value=payload):
            tasks = data.load_mbpp(Path("/unused"))
        self.assertEqual(len(tasks), 500)
        self.assertEqual((tasks[0].number, tasks[-1].number), (11, 510))
        self.assertEqual(tasks[0].source_id, "MBPP/11")
        self.assertIn("def identity(value):", tasks[0].prompt)
        self.assertNotIn("return value", tasks[0].prompt)

    def test_humaneval_retains_provided_helper_context_only(self):
        rows = [{"task_id": f"HumanEval/{i}", "entry_point": "f",
                 "prompt": "import math\n\ndef helper(x):\n    return x\n\ndef f(x):\n    '''Return x.'''\n",
                 "test": "def check(candidate):\n    assert candidate(1) == 1"}
                for i in range(164)]
        payload = gzip.compress("\n".join(json.dumps(row) for row in rows).encode())
        with patch.object(data, "fetch_cached", return_value=payload):
            tasks = data.load_humaneval(Path("/unused"))
        self.assertIn("def helper(x)", tasks[0].context_source)
        self.assertNotIn("def f(", tasks[0].context_source)
        self.assertTrue(tasks[0].test_source.endswith("check(f)\n"))

    def test_translated_full_function_harnesses_keep_context(self):
        bodies = {"rs": "fn solve(x: isize) -> isize { x + 1 }",
                  "cpp": "long solve(long x) { return x + 1; }",
                  "js": "function solve(x) { return x + 1; }"}
        for lang in bodies:
            with self.subTest(language=lang):
                row = translated_row(lang)
                task = data._translated_task(row, "multiple_" + lang)
                source, _ = runner._program(task, bodies[lang])
                self.assertEqual(source.count("{"), source.count("}"))
                self.assertEqual(task.required_name, "solve")
                self.assertEqual(task.source_id, "HumanEval/0")
                self.assertIn("candidate = solve", task.test_source)
                self.assertNotIn(bodies[lang].split("(")[0], task.context_source)
                if lang == "js":
                    self.assertEqual(task.test_source, row["tests"])
                else:
                    self.assertFalse(task.test_source.lstrip().startswith("}"))
                if lang == "cpp":
                    self.assertIn("struct Box { long value; };", source)
                    self.assertIn("#include<bits/stdc++.h>", source)
                if lang == "rs":
                    self.assertIn("use std::collections::HashMap;", source)

    def test_unrecognized_leading_brace_is_rejected_not_deleted(self):
        row = translated_row("rs")
        row["tests"] = "}\nfn helper() {}\n" + row["tests"][2:]
        with self.assertRaisesRegex(ValueError, "Unrecognized completion harness"):
            data._translated_task(row, "multiple_rs")

    def test_pagination_exceeds_two_pages(self):
        rows = [translated_row("cpp", "mbpp", i) for i in range(11, 216)]
        responses = [page(rows[:100], 0, 205), page(rows[100:200], 100, 205),
                     page(rows[200:], 200, 205)]
        with patch.object(data, "fetch_cached", side_effect=responses) as fetch:
            tasks = data.load_multiple(Path("/unused"), "multiple_mbpp_cpp")
        self.assertEqual(len(tasks), 205)
        params = [parse_qs(urlparse(call.args[0]).query) for call in fetch.call_args_list]
        self.assertEqual([p["offset"][0] for p in params], ["0", "100", "200"])
        self.assertTrue(all(p["config"] == ["mbpp-cpp"] for p in params))

    def test_translated_mbpp_excludes_non_test_ids(self):
        rows = [translated_row("js", "mbpp", i) for i in (1, 11, 510, 511)]
        with patch.object(data, "fetch_cached", return_value=page(rows)):
            tasks = data.load_multiple(Path("/unused"), "multiple_mbpp_js")
        self.assertEqual([task.number for task in tasks], [11, 510])

    def test_pagination_rejects_incomplete_duplicate_and_truncated_pages(self):
        one = translated_row()
        cases = [
            [page([one], total=2), page([], offset=1, total=2)],
            [page([one, one])],
            [json.dumps({"rows": [{"row": one, "row_idx": 0, "truncated_cells": ["prompt"]}],
                         "num_rows_total": 1}).encode()],
            [json.dumps({"rows": [], "num_rows_total": 0, "partial": True}).encode()],
        ]
        for responses in cases:
            with self.subTest(responses=responses), patch.object(data, "fetch_cached", side_effect=responses):
                with self.assertRaises(RuntimeError):
                    data.load_multiple(Path("/unused"), "multiple_rs")

    def test_limit_and_start_apply_independently_per_suite(self):
        tasks = [data.Task("a", "py", str(i), i, "", "") for i in range(5)]
        with patch.object(data, "load_humaneval", return_value=tasks), \
             patch.object(data, "load_multiple", return_value=tasks):
            selected = data.load_tasks(["humaneval", "multiple_rs"], Path("/unused"), 2, start=1)
        self.assertEqual([task.number for task in selected], [1, 2, 1, 2])


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.pin_patch = patch.dict(runner._EXECUTABLES,
            {"py": sys.executable, "rs": "/native/rustc", "cpp": "/native/g++", "js": "/native/node"}, clear=True)
        self.pin_patch.start()
        self.addCleanup(self.pin_patch.stop)

    @staticmethod
    def task(lang="py", context=""):
        return data.Task("fixture", lang, "0", 0, "", "", context_source=context)

    @staticmethod
    def result(returncode=0, output="", timed_out=False):
        return SimpleNamespace(returncode=returncode, output=output, timed_out=timed_out, output_truncated=False)

    def test_generate_only_never_starts_process(self):
        with patch.object(runner, "run_bounded_command") as run:
            verdict = runner.test_task(self.task(), "anything", "none")
        self.assertIsNone(verdict["passed"])
        run.assert_not_called()

    def test_all_languages_use_fresh_local_paths_and_common_time_limits(self):
        samples = {"py": "def f(x): return x", "rs": "fn f(x: isize) -> isize { x }",
                   "js": "function f(x) { return x; }", "cpp": "long f(long x) { return x; }"}
        directories = []
        def simulate(command, **kwargs):
            work = Path(kwargs["cwd"])
            directories.append(work)
            self.assertTrue(next(work.glob("solution.*")).is_file())
            self.assertEqual(kwargs["timeout"], 40)
            self.assertEqual(kwargs["cpu_seconds"], 41)
            self.assertEqual(kwargs["max_output_bytes"], 64000)
            self.assertEqual(kwargs["env"]["HOME"], str(work))
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
            self.assertIn(str(work), " ".join(command))
            return self.result()
        with patch.object(runner, "run_bounded_command", side_effect=simulate) as run:
            for lang, code in samples.items():
                self.assertTrue(runner.test_task(self.task(lang), code, "local")["passed"])
        self.assertEqual(run.call_count, 4)
        self.assertEqual(len(set(directories)), 4)
        self.assertTrue(all(not path.exists() for path in directories))

    def test_timeout_is_candidate_timeout_and_removes_workdir(self):
        paths = []
        def simulate(command, **kwargs):
            paths.append(Path(kwargs["cwd"]))
            return self.result(-9, "partial log", True)
        with patch.object(runner, "run_bounded_command", side_effect=simulate):
            verdict = runner.test_task(self.task(), "def f(): pass", "local", timeout=2)
        self.assertEqual(verdict["status"], "timeout")
        self.assertFalse(verdict["passed"])
        self.assertEqual(verdict['execution']['returncode'], -9)
        self.assertEqual(verdict['execution']['signal'], 9)
        self.assertTrue(verdict['execution']['timed_out'])
        self.assertEqual(verdict['execution']['timeout_seconds'], 2)
        self.assertTrue(all(not path.exists() for path in paths))

    def test_missing_executable_is_not_a_model_failure(self):
        with patch.object(runner, "run_bounded_command", side_effect=FileNotFoundError("missing interpreter")):
            verdict = runner.test_task(self.task(), "def f(): pass", "local")
        self.assertIsNone(verdict["passed"])
        self.assertEqual(verdict["status"], "infra_error")

    def test_compile_failure_is_distinguished_from_test_failure(self):
        with patch.object(runner, "run_bounded_command", return_value=self.result(
                65, "error: bad source\n" + runner._COMPILE_ERROR)):
            verdict = runner.test_task(self.task("cpp"), "long f() { broken; }", "local")
        self.assertFalse(verdict["passed"])
        self.assertEqual(verdict["status"], "compile_error")
        self.assertNotIn(runner._COMPILE_ERROR, verdict["feedback"])

    def test_native_python_executes_public_and_sealed_fixture(self):
        task = data.Task("fixture", "py", "0", 0, "", "assert f(2) == 3",
                         feedback_test_source="assert f(1) == 1")
        public = runner.test_task(task, "def f(x): return x", "local")
        final = runner.grade_final(task, "def f(x): return x", "local")
        self.assertTrue(public["passed"])
        self.assertFalse(final["passed"])
        self.assertEqual(final["feedback"], "")
        self.assertIn("AssertionError", final["sealed_diagnostics"])

    @unittest.skipUnless(shutil.which("g++"), "native C++ compiler is unavailable")
    def test_native_cpp_compiles_runs_and_reports_invalid_source(self):
        task = data.Task("fixture", "cpp", "0", 0, "", "int main() {return f(2) == 2 ? 0 : 1;}")
        with patch.dict(runner._EXECUTABLES, {"cpp": shutil.which("g++")}):
            passing = runner.grade_final(task, "int f(int x) {return x;}", "local")
            failing = runner.grade_final(task, "int f(int x) {return x + 1;}", "local")
            invalid = runner.test_task(task, "int f(int x) {broken;}", "local")
        self.assertTrue(passing["passed"], passing)
        self.assertFalse(failing["passed"])
        self.assertEqual(invalid["status"], "compile_error")

    def test_preflight_requires_dedicated_python_and_records_package_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "runtime missing"):
                runner.preflight_local({"py"}, data_root=root)
            python = root / "envs/function/bin/python"
            python.parent.mkdir(parents=True)
            python.symlink_to(sys.executable)
            probe = {"executable": str(python), "python": "3.fixture", "packages": [["fixture", "1"]]}
            old_root = runner._LOCAL_DATA_ROOT
            self.addCleanup(setattr, runner, "_LOCAL_DATA_ROOT", old_root)
            with patch.object(runner, "run_bounded_command", return_value=self.result(output=json.dumps(probe))):
                identities = runner.preflight_local({"py"}, data_root=root)
            self.assertEqual(identities["py"]["executable"], str(python))
            self.assertEqual(len(identities["py"]["packages_sha256"]), 64)
            self.assertEqual(runner.local_executable("py"), str(python))

    def test_public_context_and_future_imports_are_preserved(self):
        context = "from hashlib import md5\n\ndef helper(x): return x\n"
        source, _ = runner._program(self.task(context=context),
                                    "from __future__ import annotations\n\ndef f(x): return helper(x)\n")
        self.assertTrue(source.startswith("from __future__ import annotations"))
        self.assertIn("from hashlib import md5", source)
        self.assertIn("def helper", source)
        ast.parse(source)

    def test_future_import_keeps_same_line_statements_and_utf8_columns(self):
        headers = [
            "from __future__ import annotations; sentinel = 123",
            "'''한글 docstring'''; from __future__ import annotations; sentinel = 123",
            "from __future__ import (\n    annotations\n); sentinel = 123",
            "from __future__ import annotations; from __future__ import division; sentinel = 123",
        ]
        for header in headers:
            with self.subTest(header=header):
                code = header + "\ndef f(x: MissingType): return sentinel\n"
                future, body = runner._split_future_imports(code)
                namespace = {}
                exec("\n".join([*future, body]), namespace)
                self.assertEqual(namespace["f"](0), 123)
                self.assertEqual(namespace["f"].__annotations__["x"], "MissingType")
                if "한글" in header:
                    self.assertIn("한글 docstring", body)

    def test_duplicate_rust_imports_are_not_added(self):
        statement = "use std::collections::HashMap;"
        source, _ = runner._program(self.task("rs", context=statement), statement + "\nfn f() {}")
        self.assertEqual(source.count(statement), 1)

    def test_javascript_and_cpp_fences_are_removed(self):
        fence = chr(96) * 3
        for language in ("javascript", "cpp", "c++"):
            self.assertEqual(runner.strip_code(fence + language + "\nsource\n" + fence), "source\n")


if __name__ == "__main__":
    unittest.main()
