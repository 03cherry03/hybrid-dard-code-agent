"""Safe-by-default evaluation for generated functions. Docker only, never native."""
from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from datasets_code import Task

IMAGES = {"py": "python:3.11-slim", "rs": "rust:1.85-slim"}


def strip_code(text: str) -> str:
    text = text.strip()
    fenced = re.search(r"```(?:python|rust|py|rs)?\s*\n(.*?)```", text, re.I | re.S)
    if fenced:
        text = fenced.group(1).strip()
    return text + "\n"


def preflight_docker(languages: set[str], *, pull_missing: bool = False) -> None:
    if not shutil.which("docker"):
        raise RuntimeError("Docker CLI is missing. Install/start Docker or use --runner none (generate only).")
    r = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                       capture_output=True, text=True, timeout=20)
    if r.returncode:
        raise RuntimeError("Docker engine isn't reachable. Start docker service and check `docker info`.")
    for lang in sorted(languages):
        image = IMAGES[lang]
        r = subprocess.run(["docker", "image", "inspect", image],
                           capture_output=True, text=True, timeout=15)
        if r.returncode:
            if not pull_missing:
                raise RuntimeError(f"Docker image {image} absent. Run `docker pull {image}` or --pull-images.")
            print(f"[runner] pulling {image}", flush=True)
            subprocess.run(["docker", "pull", image], check=True, timeout=600)


def _result(ok, status, feedback, start, phase=None) -> dict[str, Any]:
    return {"passed": ok, "status": status, "feedback": feedback[:5000],
            "test_wall_s": round(time.perf_counter() - start, 3), "phase": phase}


def _syntax_check(task: Task, code: str) -> dict[str, Any] | None:
    t = time.perf_counter()
    if not code.strip():
        return _result(False, "empty_code", "Model returned no source code", t, "static")
    if task.language == "py":
        try:
            ast.parse(code)
        except SyntaxError as e:
            return _result(False, "syntax_error", f"SyntaxError: {e}", t, "static")
    elif re.search(r"(?m)^\s*fn\s+main\s*\(", code):
        return _result(False, "output_format_error", "Rust benchmark harness has its own fn main(); do not output main()", t, "static")
    return None


def test_task(task: Task, code: str, runner: str, timeout: int = 40) -> dict[str, Any]:
    """Benchmark tests only in a restricted Docker container. No native fallback."""
    t = time.perf_counter()
    if runner == "none":
        return _result(None, "not_tested", "--runner none", t)
    assert runner == "docker", runner
    pre = _syntax_check(task, code)
    if pre is not None:
        return pre
    with tempfile.TemporaryDirectory(prefix="dardbench_") as tmp:
        path = Path(tmp).resolve()
        path.chmod(0o755)
        if task.language == "py":
            # HumanEval is normally a *completion* benchmark. We ask complete
            # functions, hence not an official pass@1 comparable protocol.
            source = "from typing import *\nimport math\n" + code + "\n" + task.test_source + "\n"
            (path / "solution.py").write_text(source, encoding="utf-8")
            command = ["python", "-I", "-B", "/work/solution.py"]
        else:
            source = code + "\n" + task.test_source
            (path / "solution.rs").write_text(source, encoding="utf-8")
            command = ["sh", "-c", "rustc --edition=2021 /work/solution.rs -Awarnings -o /tmp/solution && /tmp/solution"]
        cmd = ["docker", "run", "--rm", "--network", "none", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--pids-limit", "64",
               "--memory", "2g", "--cpus", "2", "--read-only", "--user", "65534:65534",
               "--tmpfs", "/tmp:rw,exec,nosuid,size=256m", "--workdir", "/tmp",
               "--mount", f"type=bind,source={path},target=/work,readonly",
               IMAGES[task.language], *command]
        try:
            run = subprocess.run(cmd, text=True, errors="replace", capture_output=True, timeout=timeout)
            text = (run.stdout + "\n" + run.stderr).strip()
            if run.returncode == 0:
                return _result(True, "passed", text, t, "test")
            if "link.exe" in text or "cannot find -l" in text and "libc" in text:
                return _result(None, "infra_error", text, t, "infra")
            if task.language == "rs" and re.search(r"error(?:\[E\d+\])?:", text):
                return _result(False, "compile_error", text, t, "compile")
            return _result(False, "test_failed", text, t, "test")
        except subprocess.TimeoutExpired:
            return _result(False, "timeout", f"Docker test exceeded {timeout} s", t, "test")
        except OSError as exc:
            return _result(None, "infra_error", f"Docker subprocess error: {exc}", t, "infra")


def category(status: str, feedback: str) -> str:
    if status in ("passed", "not_tested", "infra_error", "syntax_error", "output_format_error", "timeout", "empty_code"):
        return status
    m = feedback.lower()
    if "assertionerror" in m or "assertion `left == right`" in m or "panicked at" in m:
        return "assertion_failure"
    if "nameerror" in m or "e0425" in m or "cannot find value" in m:
        return "identifier_error"
    if "typeerror" in m or "e0308" in m or "mismatched types" in m:
        return "type_error"
    if status == "compile_error":
        return "compile_error"
    return "runtime_or_logic_error"
