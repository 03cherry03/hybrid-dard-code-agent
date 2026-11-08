"""Local process bridge to pinned official benchmark evaluators.

Importing this file is inert. It never imports a model or evaluation package on
import. Pure command/result helpers are covered by offline tests.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import subprocess
import sys

VERSIONS = {"evalplus": "0.3.1", "bigcodebench": "0.2.2.dev2"}


def official_command(evaluator: str, dataset: str, samples: str, python: str = "python3") -> list[str]:
    if evaluator == "evalplus" and dataset in ("humaneval", "mbpp"):
        return [python, "-m", "evalplus.evaluate", "--dataset", dataset,
                "--samples", samples, "--parallel", "1"]
    if evaluator == "bigcodebench" and dataset == "instruct":
        return [python, "-m", "bigcodebench.evaluate", "--split", "instruct", "--subset", "full",
                "--execution", "local", "--samples", samples, "--parallel", "1", "--pass_k", "1",
                "--calibrated", "False"]
    raise ValueError(f"Unsupported official evaluator/dataset: {evaluator}/{dataset}")


def parse_official_result(document: dict, evaluator: str, task_id: str, *, groundtruth: dict | None = None) -> dict:
    evaluation = document["eval"]
    if set(evaluation) != {task_id} or len(evaluation[task_id]) != 1:
        raise ValueError("Official grading must contain exactly one submitted task/solution")
    record = evaluation[task_id][0]
    if record.get("task_id") != task_id:
        raise ValueError("Official result task identity mismatch")
    if evaluator == "evalplus":
        statuses = [record["base_status"], record["plus_status"]]
    elif evaluator == "bigcodebench":
        if (not groundtruth or groundtruth.get("gt_pass_rate") != 1
                or groundtruth.get("failed_tasks") != []):
            return {"task_id": task_id, "passed": None, "status": "infra_error",
                    "sealed_diagnostics": "BigCodeBench reference solution failed in this environment"}
        statuses = [record["status"]]
    else:
        raise ValueError(f"Unsupported evaluator: {evaluator}")
    if any(value not in ("pass", "fail", "timeout") for value in statuses):
        raise ValueError("Official evaluator returned missing/unknown status")
    passed = all(value == "pass" for value in statuses)
    result = {"task_id": task_id, "passed": passed,
              "status": "passed" if passed else "timeout" if "timeout" in statuses else "test_failed",
              "official_statuses": statuses, "feedback": ""}
    if evaluator == "evalplus":
        result.update(base_pass=statuses[0] == "pass", plus_pass=statuses[1] == "pass")
    return result


def _run_official(command: list[str], *, env: dict, cwd: Path) -> tuple[int, str]:
    # Inherit the worker process group so the supervising timeout also kills
    # official evaluators and all of their candidate subprocesses. Drain output
    # continuously but retain only its bounded tail.
    output = bytearray()
    with subprocess.Popen(command, env=env, cwd=cwd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT) as process:
        while chunk := process.stdout.read(8192):
            output.extend(chunk)
            if len(output) > 64000:
                del output[:-64000]
        return process.wait(), output.decode("utf-8", errors="replace")


def _run_evalplus_api(request: dict, *, env: dict) -> dict:
    """Use the unchanged official oracle and checker without the CLI monitor.

    EvalPlus 0.3.1's CLI starts a non-daemon progress thread which sleeps for
    twenty seconds even when a single submission has already finished. Each
    request runs in a fresh worker, with its override set before package import.
    """
    override = {"humaneval": "HUMANEVAL_OVERRIDE_PATH", "mbpp": "MBPP_OVERRIDE_PATH"}[request["dataset"]]
    previous = os.environ.get(override)
    os.environ[override] = env[override]
    try:
        # Ground-truth preparation prints progress. Keep all official output
        # outside the worker's sole JSON response and the repair feedback field.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from evalplus.data import (get_human_eval_plus, get_human_eval_plus_hash,
                                       get_mbpp_plus, get_mbpp_plus_hash)
            from evalplus.evaluate import get_groundtruth, check_correctness
            from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS

            if request["dataset"] == "humaneval":
                problems, dataset_hash = get_human_eval_plus(), get_human_eval_plus_hash()
                special_outputs = []
            else:
                # The official loader restores tuples/sets in serialized MBPP
                # inputs before either the reference or candidate is executed.
                problems, dataset_hash = get_mbpp_plus(), get_mbpp_plus_hash()
                special_outputs = MBPP_OUTPUT_NOT_NONE_TASKS
            task_id = request["task_id"]
            if set(problems) != {task_id}:
                raise ValueError("Official override must contain exactly the submitted task")
            expected = get_groundtruth(problems, dataset_hash, special_outputs)
            checked = check_correctness(request["dataset"], 0, problems[task_id], request["solution"],
                                        expected[task_id], base_only=False, fast_check=True)
            if checked.get("task_id") != task_id:
                raise ValueError("Official checker returned a different task")
            document = {"eval": {task_id: [{"task_id": task_id,
                         "base_status": checked["base"][0], "plus_status": checked["plus"][0]}]}}
            result = parse_official_result(document, "evalplus", task_id)
            result["evaluation_api"] = "evalplus.evaluate.check_correctness"
            return result
    finally:
        if previous is None:
            os.environ.pop(override, None)
        else:
            os.environ[override] = previous


def run_request(request: dict, work: Path, scratch: Path) -> dict:
    work, scratch = Path(work).resolve(), Path(scratch).resolve()
    if not work.is_dir() or not scratch.is_dir():
        raise ValueError("Official worker requires existing work and scratch directories")
    evaluator = request["evaluator"]
    expected_version = VERSIONS[evaluator]
    actual_version = importlib.metadata.version(evaluator)
    if request["version"] != expected_version or actual_version != expected_version:
        raise RuntimeError(f"Evaluator version mismatch: expected {expected_version}, installed {actual_version}")
    dataset = request["dataset"]
    override = {("evalplus", "humaneval"): "HUMANEVAL_OVERRIDE_PATH",
                ("evalplus", "mbpp"): "MBPP_OVERRIDE_PATH",
                ("bigcodebench", "instruct"): "BIGCODEBENCH_OVERRIDE_PATH"}[(evaluator, dataset)]
    env = dict(os.environ)
    env[override] = str(work / "raw.jsonl")
    env["HF_HUB_OFFLINE"], env["HF_DATASETS_OFFLINE"] = "1", "1"
    sample_path = scratch / "samples.jsonl"
    sample_path.write_text(json.dumps({"task_id": request["task_id"], "solution": request["solution"]}) + "\n",
                           encoding="utf-8")
    result_path = scratch / "samples_eval_results.json"
    if result_path.exists():
        raise RuntimeError("Refusing to reuse an old official grade")
    if evaluator == "evalplus":
        result = _run_evalplus_api(request, env=env)
        result["evaluator_version"] = actual_version
        return result
    command = official_command(evaluator, dataset, str(sample_path), sys.executable)
    returncode, output = _run_official(command, env=env, cwd=scratch)
    if returncode:
        raise RuntimeError(f"Official evaluator exited {returncode}: " + output[-4000:])
    document = json.loads(result_path.read_text(encoding="utf-8"))
    groundtruth = None
    if evaluator == "bigcodebench":
        groundtruth = json.loads((scratch / "samples_pass_at_k.json").read_text(encoding="utf-8"))
    result = parse_official_result(document, evaluator, request["task_id"], groundtruth=groundtruth)
    result["evaluator_version"] = actual_version
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args(argv)
    request = {}
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        result = run_request(request, args.work, args.scratch)
    except Exception as exc:
        result = {"task_id": request.get("task_id"), "passed": None, "status": "infra_error",
                  "sealed_diagnostics": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
