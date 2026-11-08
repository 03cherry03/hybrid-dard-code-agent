"""Native SWE evaluation worker. Only the frozen grading stage reads sealed tests.

The upstream parser and resolution rules are reused at the pinned harness revision;
this is a local environment adaptation, not an official environment reproduction.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib
import importlib.metadata
import importlib.machinery
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import sys
import time
import types

from local_execution import clone_environment, native_environment, run_bounded_command

NATIVE_ADAPTER_VERSION = "native_swe_v1"


def load_official_api():
    """Load unchanged pinned leaf modules without the optional harness initializer.

    Upstream's harness package initializer imports unrelated execution services.
    A namespace package with the installed package path avoids those side
    effects; parser, constants, types and grading implementations are unchanged.
    """
    from local_repository_preparation import HARNESS_COMMIT, HARNESS_VERSION
    distribution = importlib.metadata.distribution("swebench")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    identity = {"version": distribution.version, "commit": direct.get("vcs_info", {}).get("commit_id")}
    if identity != {"version": HARNESS_VERSION, "commit": HARNESS_COMMIT}:
        raise ValueError("Native grading requires the pinned official SWE parser revision")
    import swebench
    path = Path(distribution.locate_file("swebench/harness")).resolve()
    if not (path / "grading.py").is_file():
        raise ValueError("Pinned official grading module is unavailable")
    if "swebench.harness" not in sys.modules:
        package = types.ModuleType("swebench.harness")
        package.__path__ = [str(path)]
        package.__package__ = "swebench.harness"
        package.__spec__ = importlib.machinery.ModuleSpec("swebench.harness", loader=None, is_package=True)
        package.__spec__.submodule_search_locations = package.__path__
        sys.modules["swebench.harness"] = package
        setattr(swebench, "harness", package)
    grading = importlib.import_module("swebench.harness.grading")
    spec = importlib.import_module("swebench.types").TestSpec
    return {"identity": identity, "grading": grading, "TestSpec": spec}


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def test_ids(value):
    value = json.loads(value) if isinstance(value, str) else value
    if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
        raise ValueError("Expected a list of official test IDs")
    return value


def adapt_eval_script(script, workspace, prefix, runtime):
    """Rewrite infrastructure paths outside heredocs, keeping test patch bytes intact."""
    paths = [("/opt/miniconda3/envs/testbed", str(prefix)),
             ("/testbed", str(workspace)), ("/root", str(runtime / "home")),
             ("/tmp", str(runtime / "tmp"))]
    # Existing upstream scripts mix quoted and unquoted paths. Reject a path
    # needing shell escaping instead of silently changing embedded test data.
    for _, value in paths:
        if not re.fullmatch(r"[A-Za-z0-9_./-]+", value):
            raise ValueError("Native SWE paths must not contain spaces or shell metacharacters")
    output, heredoc, changes = [], None, []
    start_count = end_count = 0
    for line in script.splitlines(keepends=True):
        if heredoc is not None:
            output.append(line)
            if line.strip() == heredoc:
                heredoc = None
            continue
        original = line
        stripped = line.strip()
        if stripped in ("source /opt/miniconda3/bin/activate", "conda activate testbed"):
            line = ": # Native prefix is already active\n"
        elif "/etc/locale.gen" in line or stripped.startswith("locale-gen"):
            line = "locale -a | tr '[:upper:]' '[:lower:]' | grep -qx 'en_us.utf8'\n"
        elif stripped.startswith("set -") and "pipefail" in line:
            line = "set -euo pipefail\n"
        elif ">>>>> Start Test Output" in line:
            start_count += 1
            line = "set +e\n{ set +x; } 2>/dev/null\necho '>>>>> Start Test Output'\n"
        elif ">>>>> End Test Output" in line:
            end_count += 1
            line = ("SWEBENCH_NATIVE_TEST_EXIT_CODE=$?\n"
                    "echo '>>>>> End Test Output'\n"
                    "echo \"\u003e\u003e\u003e\u003e\u003e Test Exit Code: $SWEBENCH_NATIVE_TEST_EXIT_CODE\"\n"
                    "set -e\n")
        else:
            # One substitution pass: a replacement under /tmp must not itself
            # be rewritten when mapping the original script's /tmp paths.
            mapping = dict(paths)
            pattern = "(?:" + "|".join(re.escape(key) for key in mapping) + r")(?=/|[^A-Za-z0-9_.-]|$)"
            line = re.sub(pattern, lambda match: mapping[match.group(0)], line)
            # The native recipe supplies Maven rather than its optional daemon.
            # The goals/options and the official log parser remain unchanged.
            line = re.sub(r"(?<![A-Za-z0-9_/])mvnd(?=\s|$)", "mvn", line)
            line = line.replace("git config --global", "git config --local")
        if line != original:
            changes.append({"line": len(output) + 1, "original_sha256": text_hash(original),
                            "adapted_sha256": text_hash(line)})
        output.append(line)
        match = re.search(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?", original)
        if match:
            heredoc = match.group(1)
    if heredoc is not None or start_count != 1 or end_count != 1:
        raise ValueError("Unrecognized sealed evaluation script structure")
    return "".join(output), changes


def parse_official_report(row, log_path, patch):
    api = load_official_api()
    grading = api["grading"]
    spec = api["TestSpec"](instance_id=row["instance_id"], image="", eval_script_list=[],
                    repo=row["repo"], version=row["version"],
                    FAIL_TO_PASS=test_ids(row["FAIL_TO_PASS"]),
                    PASS_TO_PASS=test_ids(row["PASS_TO_PASS"]),
                    log_parser=row["log_parser"], eval_type=row["eval_type"])
    statuses, valid = grading.get_logs_eval(spec, str(log_path))
    report = grading.get_eval_report(spec, {"instance_id": row["instance_id"], "model_patch": patch},
                             str(log_path), include_tests_status=True)[row["instance_id"]]
    observed = {case: statuses.get(grading._resolve_case(case, statuses))
                for case in spec.FAIL_TO_PASS + spec.PASS_TO_PASS}
    if report.get("infra_failure"):
        return {"resolved": None, "status": "native_infrastructure_error", "official_report": report,
                "observed_expected_tests": observed, "parser_output_valid": valid}
    return {"resolved": report["resolved"], "status": "native_official_parser_report",
            "official_report": report, "observed_expected_tests": observed, "parser_output_valid": valid}


def run_case(request, output, *, command_runner=run_bounded_command,
             environment_cloner=clone_environment, parser=parse_official_report):
    """Evaluate one immutable submission in a new checkout and cloned prefix."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    row, prepared = request["task"], request["prepared"]
    patch = request.get("patch", "")
    timeout = float(request.get("timeout", 1800))
    setup_timeout = float(request.get("setup_timeout", 1800))
    started = time.monotonic()
    workspace, runtime, prefix = output / "workspace", output / "runtime", output / "environment"
    report = {"instance_id": row["instance_id"], "backend": "native",
              "native_adapter_version": NATIVE_ADAPTER_VERSION,
              "native_adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "environment_id": prepared["environment_id"],
              "base_commit": row["base_commit"], "patch_sha256": text_hash(patch),
              "original_eval_script_sha256": text_hash(row["eval_script"])}
    if workspace.exists() or prefix.exists():
        raise ValueError("Native grading work directory already exists; use a new attempt directory")
    try:
        from local_repository_preparation import verify_prepared_environment
        if request.get("verify_prepared", True):
            verify_prepared_environment(prepared)
        shutil.copytree(prepared["source_path"], workspace, symlinks=True)
        environment_cloner(Path(prepared["environment_path"]), prefix, timeout=setup_timeout)
        env = native_environment(prefix, workspace, runtime)
        remaining = lambda: max(.1, setup_timeout - (time.monotonic() - started))
        result = command_runner(["git", "rev-parse", "HEAD"], cwd=workspace, env=env,
                                timeout=remaining(), max_output_bytes=2048)
        if result.returncode or result.output.strip() != row["base_commit"]:
            raise ValueError("Prepared source does not match the sealed original base commit")
        if patch:
            result = command_runner(["git", "-c", "core.hooksPath=/dev/null", "apply", "--whitespace=nowarn", "-"],
                                    cwd=workspace, env=env, input_text=patch,
                                    timeout=remaining(), max_output_bytes=200000)
            (output / "patch_apply.log").write_text(result.output)
            if result.timed_out:
                raise TimeoutError("Patch application exceeded the grading deadline")
            if result.returncode:
                report.update(resolved=False, status="patch_apply_failed")
                return report
        script, adaptations = adapt_eval_script(row["eval_script"], workspace, prefix, runtime)
        path = output / "evaluation.private.sh"
        path.write_text(script); path.chmod(0o600)
        report.update(adapted_eval_script_sha256=text_hash(script), adaptations=adaptations)
        result = command_runner(["bash", str(path)], cwd=workspace, env=env,
                                timeout=timeout, max_output_bytes=20 * 1024 * 1024)
        log = output / "test_output.txt"; log.write_text(result.output)
        report.update(returncode=result.returncode, timed_out=result.timed_out,
                      output_truncated=result.truncated)
        if result.timed_out:
            # The same environment passed original/reference controls. A
            # candidate that makes its tests hang is an unresolved candidate.
            report.update(resolved=False, status="candidate_test_timeout")
        elif result.truncated:
            report.update(resolved=None, status="native_log_truncated")
        else:
            report.update(parser(row, log, patch))
        if request.get("verify_prepared", True):
            verify_prepared_environment(prepared)
    except Exception as exc:
        report.update(resolved=None, status="native_infrastructure_error",
                      error=type(exc).__name__ + ": " + str(exc))
    finally:
        report["wall_s"] = round(time.monotonic() - started, 6)
        # Keep logs, frozen requests, and hashes, not a mutable dependency copy
        # for every candidate. Preparation is the reproducible starting point.
        for path in (workspace, prefix, runtime):
            if path.exists():
                try:
                    shutil.rmtree(path)
                except OSError as exc:
                    report.update(resolved=None, status="native_cleanup_error",
                                  error=type(exc).__name__ + ": " + str(exc))
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_case(json.loads(args.request.read_text()), args.output)
    print(json.dumps({k: result.get(k) for k in ("instance_id", "status", "resolved", "wall_s")}))
    return 0 if isinstance(result.get("resolved"), bool) else 1


if __name__ == "__main__":
    raise SystemExit(main())
