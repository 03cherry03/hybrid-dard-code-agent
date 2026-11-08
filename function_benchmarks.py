"""Offline official-dataset adapters and sealed local grading.

No package import in this module downloads data or executes generated code.
EvalPlus keeps its official differential oracle (including MBPP special cases);
BigCodeBench keeps its official unittest runner and dependency environment. The small
worker invokes the pinned official checker APIs or CLI, never a rewritten
approximation of the tests.
"""
from __future__ import annotations

import ast
import copy
import gzip
import hashlib
import json
import os
import re
import tempfile
import time
from functools import lru_cache
from pathlib import Path

from datasets_code import SUITES, Task, _python_context, public_feedback_tests
from local_execution import run_bounded_command

DEFAULT_DATA_ROOT = Path("/data/hybrid-dard")
OFFICIAL_SUITES = {
    "humaneval_plus": {"revision": "v0.1.10", "evaluator": "evalplus", "dataset": "humaneval",
                       "version": "0.3.1"},
    "mbpp_plus": {"revision": "v0.2.0", "evaluator": "evalplus", "dataset": "mbpp",
                  "version": "0.3.1"},
    "bigcodebench_instruct": {"revision": "v0.1.4", "evaluator": "bigcodebench", "dataset": "instruct",
                              "version": "0.2.2.dev2"},
}


def manifest_path(data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    return Path(data_root) / "datasets" / "functions" / "manifest.json"


def _digest(value) -> str:
    if not isinstance(value, bytes):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(value).hexdigest()


@lru_cache(maxsize=6)
def _read_verified_records(path_string: str, expected_sha: str, size: int, mtime: int) -> dict:
    path = Path(path_string)
    payload = path.read_bytes()
    if _digest(payload) != expected_sha:
        raise ValueError(f"Prepared dataset checksum mismatch: {path}")
    if path.suffix == ".gz":
        payload = gzip.decompress(payload)
    records = {}
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"Missing task_id in {path}:{line_number}")
        if task_id in records:
            raise ValueError(f"Duplicate task_id {task_id} in {path}")
        records[task_id] = row
    if not records:
        raise ValueError(f"Empty prepared dataset: {path}")
    return records


def _records(path: Path, expected_sha: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise ValueError("A full lowercase SHA256 is required for prepared datasets")
    info = path.stat()
    return _read_verified_records(str(path.resolve()), expected_sha, info.st_size, info.st_mtime_ns)


def write_official_manifest(data_root: Path, raw_paths: dict[str, Path], *,
                            revisions: dict | None = None,
                            evaluator_versions: dict | None = None) -> Path:
    """Register already prepared original JSONL files; never fetch or execute them."""
    target = manifest_path(data_root)
    suites = {}
    for suite, raw_path in raw_paths.items():
        if suite not in OFFICIAL_SUITES:
            raise ValueError(f"Unknown official function suite: {suite}")
        spec = OFFICIAL_SUITES[suite]
        raw_path = Path(raw_path).resolve()
        sha = _digest(raw_path.read_bytes())
        rows = _records(raw_path, sha)
        suites[suite] = {
            "path": os.path.relpath(raw_path, target.parent.resolve()), "sha256": sha,
            "revision": (revisions or {}).get(suite, spec["revision"]), "num_tasks": len(rows),
            "evaluator_version": (evaluator_versions or {}).get(suite, spec["version"]),
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"schema_version": 1, "suites": suites}, indent=2) + "\n", encoding="utf-8")
    return target


def _mbpp_prompt(row: dict) -> str:
    prompt = row["prompt"]
    # Some releases expose the prose and examples but put the API in the
    # canonical solution. Extract only the tested entry point's signature.
    try:
        supplied = ast.parse(prompt)
    except SyntaxError:
        supplied = ast.Module(body=[], type_ignores=[])
    if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
           and node.name == row["entry_point"] for node in supplied.body):
        return prompt
    try:
        reference = ast.parse(row["canonical_solution"])
    except SyntaxError:
        reference = ast.parse(prompt + row["canonical_solution"])
    target = next((node for node in reference.body
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and node.name == row["entry_point"]), None)
    if target is None:
        raise ValueError(f"Cannot derive public API for {row['task_id']}")
    stub = copy.deepcopy(target)
    stub.body, stub.decorator_list = [ast.Pass()], []
    return prompt + "\n\nRequired Python API (implement the complete function):\n" + ast.unparse(stub)


def load_official_tasks(suite: str, data_root: Path = DEFAULT_DATA_ROOT) -> list[Task]:
    if suite not in OFFICIAL_SUITES:
        raise ValueError(f"Unknown official suite: {suite}")
    path = manifest_path(data_root)
    if not path.is_file():
        raise FileNotFoundError(f"Official datasets are not prepared: {path}. Run prepare_data.py explicitly.")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema_version") != 1 or suite not in doc.get("suites", {}):
        raise ValueError(f"Unsupported manifest schema or missing {suite}: {path}")
    entry, spec = doc["suites"][suite], OFFICIAL_SUITES[suite]
    raw_path = (path.parent / entry["path"]).resolve()
    rows = _records(raw_path, entry["sha256"])
    if entry.get("num_tasks", len(rows)) != len(rows):
        raise ValueError(f"Prepared task count mismatch for {suite}")
    version = entry.get("evaluator_version", spec["version"])
    if version != spec["version"]:
        raise ValueError(f"Unsupported {spec['evaluator']} evaluator version {version}; expected {spec['version']}")
    revision = entry.get("revision")
    if not revision:
        raise ValueError("Prepared datasets must record their source revision")
    tasks = []
    for task_id, row in rows.items():
        required = ("entry_point", "canonical_solution", "instruct_prompt", "complete_prompt", "code_prompt", "test") \
            if spec["evaluator"] == "bigcodebench" else \
            ("entry_point", "prompt", "canonical_solution", "base_input", "plus_input", "atol")
        if any(key not in row for key in required):
            raise ValueError(f"Incomplete official record: {suite}/{task_id}")
        number = int(task_id.rsplit("/", 1)[-1])
        context = ""
        if suite == "bigcodebench_instruct":
            prompt = row["instruct_prompt"]
        elif suite == "mbpp_plus":
            prompt = _mbpp_prompt(row)
        else:
            prompt = row["prompt"]
            context = _python_context(prompt, row["entry_point"])
            if context:
                prompt = ("The imports/helpers before the requested function are supplied by the runner. "
                          "Return the complete requested function without repeating those helpers.\n\n" + prompt)
        source_id = f"{'HumanEval' if suite == 'humaneval_plus' else 'MBPP' if suite == 'mbpp_plus' else 'BigCodeBench'}/{number}"
        metadata = {"raw_path": str(raw_path), "raw_sha256": entry["sha256"],
                    "record_sha256": _digest(row), "revision": revision,
                    "evaluator_version": version,
                    "official_dataset": spec["dataset"], "dataset_task_count": len(rows)}
        tasks.append(Task(suite, "py", task_id, number, prompt, "", row["entry_point"],
                          family=SUITES[suite]["family"], source_config=SUITES[suite]["source_config"],
                          source_url=SUITES[suite]["source_url"], source_id=source_id,
                          context_source=context, final_evaluator=spec["evaluator"], evaluator_metadata=metadata,
                          feedback_test_source=public_feedback_tests(
                              row["instruct_prompt"] if suite == "bigcodebench_instruct" else row["prompt"],
                              row["entry_point"])))
    return sorted(tasks, key=lambda task: task.number)


def _official_record(task: Task) -> dict:
    meta = task.evaluator_metadata
    row = _records(Path(meta["raw_path"]), meta["raw_sha256"])[task.task_id]
    if _digest(row) != meta["record_sha256"]:
        raise ValueError("Official task record differs from the frozen experiment")
    return row


def _official_solution(task: Task, code: str) -> str:
    # Supplied context is public. Do not attach any test, contract or oracle.
    from evaluation import _split_future_imports
    try:
        futures, body = _split_future_imports(code)
        context_futures, context = _split_future_imports(task.context_source)
    except SyntaxError:
        return task.context_source + "\n" + code
    return "\n".join([*dict.fromkeys(context_futures + futures), context, body]) + "\n"


def grade_official(task: Task, code: str, *, timeout: int = 300) -> dict:
    """Run one frozen submission in the pinned local evaluator environment.

    The outer timeout includes reference/oracle preparation, so expiry is an
    infrastructure result, never a fabricated candidate timeout.
    """
    from evaluation import official_python, local_environment
    if timeout <= 0 or task.final_evaluator not in ("evalplus", "bigcodebench"):
        raise ValueError("Invalid official grading request")
    started = time.perf_counter()
    result = {"passed": None, "status": "infra_error", "feedback": "", "sealed": True, "stage": "final"}
    try:
        row = _official_record(task)
        meta = task.evaluator_metadata
        executable = official_python(task.final_evaluator)
        with tempfile.TemporaryDirectory(prefix="dardgrade_") as tmp:
            work = Path(tmp).resolve()
            scratch = work / "scratch"
            scratch.mkdir()
            payload = {"task_id": task.task_id, "evaluator": task.final_evaluator,
                       "dataset": meta["official_dataset"], "version": meta["evaluator_version"],
                       "solution": _official_solution(task, code)}
            request = work / "request.json"
            request.write_text(json.dumps(payload), encoding="utf-8")
            (work / "raw.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            command = [executable, "-I", "-B", str(Path(__file__).with_name("official_function_worker.py")),
                       "--work", str(work), "--scratch", str(scratch), "--request", str(request)]
            run = run_bounded_command(command, cwd=scratch, env=local_environment(scratch, executable),
                                      timeout=timeout, max_output_bytes=64000, cpu_seconds=timeout + 1)
            result["execution"] = {
                "returncode": run.returncode, "signal": -run.returncode if run.returncode < 0 else None,
                "timed_out": bool(run.timed_out), "output_truncated": bool(run.output_truncated),
                "subprocess_wall_s": getattr(run, "wall_s", None), "timeout_seconds": timeout,
            }
            if run.timed_out:
                result["sealed_diagnostics"] = f"Whole official grader exceeded {timeout} s"
            elif run.returncode or run.output_truncated:
                result["sealed_diagnostics"] = f"Official worker exited {run.returncode}: " + run.output[-5000:]
            else:
                answer = json.loads(run.output)
                if answer.get("task_id") != task.task_id or type(answer.get("passed")) not in (bool, type(None)):
                    raise ValueError("Malformed official evaluator response")
                expected_statuses = {True: {"passed"}, False: {"test_failed", "timeout"}, None: {"infra_error"}}
                if answer.get("status") not in expected_statuses[answer["passed"]]:
                    raise ValueError("Inconsistent official evaluator verdict")
                result.update(answer)
                result["feedback"] = ""
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        result.update(passed=None, status="infra_error", sealed_diagnostics=str(exc))
    result.update(test_wall_s=time.perf_counter() - started, phase="grading", evaluator=task.final_evaluator,
                  sealed=True, stage="final", feedback="")
    return result
