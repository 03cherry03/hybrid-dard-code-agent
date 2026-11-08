"""Local evaluation of complete functions with separate public and sealed tests."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shlex
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from datasets_code import Task
from local_execution import environment_fingerprint, run_bounded_command
from storage import data_root as resolve_data_root

LANGUAGES = {"py", "rs", "js", "cpp"}
_EXECUTABLES: dict[str, str] = {}
_OFFICIAL_PYTHONS: dict[str, str] = {}
_LOCAL_DATA_ROOT: Path | None = None
_COMPILE_ERROR = "__HYBRID_DARD_COMPILE_ERROR__"


def _root() -> Path:
    return _LOCAL_DATA_ROOT or resolve_data_root()


def _find_tool(language: str, root: Path) -> str:
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported test language: {language}")
    bindir = root / "envs" / "function" / "bin"
    names = {"py": ("python",), "rs": ("rustc",), "js": ("node",),
             "cpp": ("g++", "x86_64-conda-linux-gnu-g++", "x86_64-conda-linux-gnu-c++", "c++")}[language]
    for name in names:
        candidate = bindir / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.absolute())
    if language != "py":
        for name in names:
            candidate = shutil.which(name)
            if candidate:
                return candidate
    raise RuntimeError(f"Local {language} runtime missing under {bindir}; run scripts/setup_env.sh function")


def local_executable(language: str) -> str:
    return _EXECUTABLES.get(language) or _find_tool(language, _root())


def official_python(evaluator: str) -> str:
    if evaluator not in ("evalplus", "bigcodebench"):
        raise ValueError(f"Unsupported evaluator: {evaluator}")
    candidate = _OFFICIAL_PYTHONS.get(evaluator) or str(_root() / "envs" / evaluator / "bin" / "python")
    if not Path(candidate).is_file() or not os.access(candidate, os.X_OK):
        raise RuntimeError(f"Local {evaluator} interpreter missing: {candidate}; run scripts/setup_env.sh evaluators")
    return candidate


def local_environment(work: Path, executable: str) -> dict[str, str]:
    """Use private caches and deterministic thread limits for every backend."""
    env = {key: os.environ[key] for key in ("LANG", "LC_ALL", "TZ", "SYSTEMROOT") if key in os.environ}
    env.update(PATH=str(Path(executable).parent) + os.pathsep + os.environ.get("PATH", os.defpath),
               HOME=str(work), TMPDIR=str(work), XDG_CACHE_HOME=str(work / ".cache"),
               XDG_CONFIG_HOME=str(work / ".config"), MPLCONFIGDIR=str(work / "matplotlib"),
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", CUDA_VISIBLE_DEVICES="",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", PYTHONHASHSEED="0", PYTHONNOUSERSITE="1",
               NLTK_DATA=str(_root() / "cache/nltk_data"))
    # A system rustc may be a rustup proxy; keep its installed toolchain lookup
    # while each candidate still has a fresh HOME and working directory.
    rustup = Path(os.environ.get("RUSTUP_HOME", str(Path.home() / ".rustup")))
    if rustup.is_dir():
        env["RUSTUP_HOME"] = str(rustup)
    return env


def _python_identity(executable: str, evaluator: str | None = None, version: str | None = None) -> dict:
    probe = ("import sys,json,importlib.metadata as m; "
             "print(json.dumps({'executable':sys.executable,'python':sys.version,"
             "'packages':sorted((d.metadata['Name'],d.version) for d in m.distributions())}))")
    run = run_bounded_command([executable, "-I", "-c", probe], timeout=30, max_output_bytes=2000000)
    if run.returncode or run.timed_out or run.output_truncated:
        raise RuntimeError(f"Cannot inspect local interpreter {executable}: {run.output[-2000:]}")
    identity = json.loads(run.output)
    packages = identity.pop("packages")
    identity["packages_sha256"] = hashlib.sha256(json.dumps(packages, sort_keys=True).encode()).hexdigest()
    if evaluator:
        installed = dict((name.lower().replace('_', '-'), value) for name, value in packages).get(evaluator)
        if installed != version:
            raise RuntimeError(f"{evaluator} version mismatch in {executable}: expected {version}, installed {installed}")
        identity.update(evaluator=evaluator, evaluator_version=installed)
    return identity


def preflight_local(languages: set[str], *, tasks=(), data_root: Path | None = None) -> dict[str, dict]:
    """Validate native executables and freeze the installed evaluator environments."""
    global _LOCAL_DATA_ROOT
    root = resolve_data_root(data_root)
    tools = {language: _find_tool(language, root) for language in sorted(languages)}
    identities = {}
    function_prefix = root / "envs/function"
    function_fingerprint = environment_fingerprint(function_prefix) if function_prefix.is_dir() else None
    for language, executable in tools.items():
        if language == "py":
            identities[language] = _python_identity(executable)
        else:
            run = run_bounded_command([executable, "--version"], timeout=20, max_output_bytes=20000)
            if run.returncode or run.timed_out or run.output_truncated:
                raise RuntimeError(f"Cannot inspect {executable}: {run.output}")
            identities[language] = {"executable": executable, "version": run.output.strip(),
                                    "executable_sha256": hashlib.sha256(Path(executable).read_bytes()).hexdigest()}
        identities[language]["prefix_fingerprint"] = function_fingerprint
    versions = {}
    for task in tasks:
        if task.final_evaluator != "legacy_assert":
            version = task.evaluator_metadata["evaluator_version"]
            if task.final_evaluator in versions and versions[task.final_evaluator] != version:
                raise ValueError("Conflicting official evaluator versions")
            versions[task.final_evaluator] = version
    official = {}
    for evaluator, version in sorted(versions.items()):
        executable = str(root / "envs" / evaluator / "bin" / "python")
        if not Path(executable).is_file():
            raise RuntimeError(f"Local evaluator missing: {executable}; run scripts/setup_env.sh evaluators")
        identities["official:" + evaluator] = _python_identity(executable, evaluator, version)
        identities["official:" + evaluator]["prefix_fingerprint"] = environment_fingerprint(root / "envs" / evaluator)
        official[evaluator] = executable
    _LOCAL_DATA_ROOT = root
    _EXECUTABLES.clear()
    _EXECUTABLES.update(tools)
    _OFFICIAL_PYTHONS.clear()
    _OFFICIAL_PYTHONS.update(official)
    return identities


def strip_code(text: str) -> str:
    text = text.strip()
    fenced = re.search(r"```(?:[A-Za-z][\w.+#-]*)?[ \t]*\r?\n(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    return text + "\n"


def _result(ok, status, feedback, start, phase=None) -> dict[str, Any]:
    return {"passed": ok, "status": status, "feedback": feedback[:5000],
            "test_wall_s": round(time.perf_counter() - start, 3), "phase": phase}


def _syntax_check(task: Task, code: str) -> dict[str, Any] | None:
    start = time.perf_counter()
    if not code.strip():
        return _result(False, "empty_code", "Model returned no source code", start, "static")
    if task.language == "py":
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return _result(False, "syntax_error", f"SyntaxError: {exc}", start, "static")
    elif task.language == "rs" and re.search(r"(?m)^\s*(?:pub\s+)?fn\s+main\s*\(", code):
        return _result(False, "output_format_error",
                       "Rust benchmark harness has its own fn main(); return only the requested functions",
                       start, "static")
    elif task.language == "cpp" and re.search(r"(?m)^\s*(?:int|auto)\s+main\s*\(", code):
        return _result(False, "output_format_error",
                       "C++ benchmark harness has its own main(); return only the requested functions",
                       start, "static")
    return None


def _split_future_imports(source: str) -> tuple[list[str], str]:
    """Move Python future imports ahead of runner-provided imports."""
    # AST columns count UTF-8 bytes, not characters. Delete only the import's
    # span: removing its whole line also deletes legal semicolon-separated
    # imports, assignments or the module docstring beside it.
    encoded = source.encode("utf-8")
    offsets = [0]
    for line in encoded.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    future = []
    spans = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            future.append(ast.get_source_segment(source, node) or "")
            start = offsets[node.lineno - 1] + node.col_offset
            end = offsets[node.end_lineno - 1] + node.end_col_offset
            # Remove a following separator and spaces so a retained statement
            # at the start of a line does not acquire unexpected indentation.
            remainder = encoded[end:]
            separator = re.match(rb"[ \t]*;[ \t]*", remainder)
            if separator:
                end += separator.end()
            spans.append((start, end))
    for start, end in reversed(spans):
        encoded = encoded[:start] + encoded[end:]
    return future, encoded.decode("utf-8")


def _program(task: Task, code: str, *, stage: str = "final", work: Path | None = None,
             executable: str | None = None) -> tuple[str, list[str]]:
    """The feedback assembly has no access to final test_source."""
    if stage not in ("feedback", "final"):
        raise ValueError(f"Unknown evaluation stage: {stage}")
    work = Path(work or Path.cwd()).resolve()
    executable = executable or {"py": "python", "rs": "rustc", "js": "node", "cpp": "g++"}[task.language]
    filename = work / f"solution.{task.language}"
    tests = task.feedback_test_source if stage == "feedback" else task.test_source
    compile_only = stage == "feedback" and not tests.strip()
    if task.language == "py":
        future_code, code = _split_future_imports(code)
        future_context, context = _split_future_imports(task.context_source)
        future = "\n".join(dict.fromkeys(future_context + future_code))
        source = "\n".join([future, "from typing import *", "import math", context, code, tests])
        if compile_only:
            return source + "\n", [executable, "-I", "-B", "-c",
                                    "import ast,sys; ast.parse(open(sys.argv[1], encoding='utf-8').read())", str(filename)]
        return source + "\n", [executable, "-I", "-B", str(filename)]

    context = task.context_source
    if task.language == "rs":
        # Repeating a Rust use declaration is a compiler error. Preserve public
        # helpers, but avoid identical imports emitted by the model.
        code_lines = {line.strip() for line in code.splitlines()}
        context = "\n".join(line for line in context.splitlines()
                            if not (line.strip().startswith("use ") and line.strip() in code_lines))
    source = "\n".join([context, code, tests]) + "\n"
    if task.language == "rs":
        compile_args = [executable, "--edition=2021", str(filename), "-Awarnings"]
        if compile_only:
            compile_args.extend(["--crate-type", "lib", "--emit=metadata", "-o", str(work / "solution.rmeta")])
        else:
            compile_args.extend(["-o", str(work / "solution")])
    elif task.language == "cpp":
        compile_args = [executable, "-std=c++17"]
        compile_args.extend(["-fsyntax-only", str(filename)] if compile_only else
                            ["-O2", "-pipe", str(filename), "-o", str(work / "solution")])
    elif task.language == "js":
        return source, [executable, "--max-old-space-size=1536",
                        "--check" if compile_only else "--unhandled-rejections=strict", str(filename)]
    else:
        raise ValueError(f"Unsupported test language: {task.language}")
    command = (shlex.join(compile_args) + "; compile_status=$?; "
               "if [ \"$compile_status\" -ne 0 ]; then "
               "case \"$compile_status\" in 126|127) exit \"$compile_status\";; esac; "
               "printf '%s\\n' '" + _COMPILE_ERROR + "' >&2; exit 65; fi; "
               + ("exit 0" if compile_only else "exec " + shlex.quote(str(work / "solution"))))
    return source, ["/bin/sh", "-c", command]


def _test_task(task: Task, code: str, runner: str, timeout: int, *, stage: str) -> dict[str, Any]:
    """Execute each attempt in a fresh local work directory with bounded output/time."""
    started = time.perf_counter()
    if runner == "none":
        return _result(None, "not_tested", "--runner none", started)
    if runner != "local":
        raise ValueError(f"Unsupported runner: {runner}")
    if timeout <= 0:
        raise ValueError("test timeout must be positive")
    if task.language not in LANGUAGES:
        raise ValueError(f"Unsupported test language: {task.language}")
    pre = _syntax_check(task, code)
    if pre is not None:
        return pre
    execution_details = {}
    try:
        executable = (official_python(task.final_evaluator)
                      if stage == "feedback" and task.final_evaluator != "legacy_assert"
                      else local_executable(task.language))
        with tempfile.TemporaryDirectory(prefix="dardbench_") as tmp:
            work = Path(tmp).resolve()
            source, command = _program(task, code, stage=stage, work=work, executable=executable)
            (work / f"solution.{task.language}").write_text(source, encoding="utf-8")
            run = run_bounded_command(command, cwd=work, env=local_environment(work, executable),
                                      timeout=timeout, max_output_bytes=64000, cpu_seconds=timeout + 1)
            execution_details = {"returncode": run.returncode,
                                 "signal": -run.returncode if run.returncode < 0 else None,
                                 "timed_out": bool(run.timed_out),
                                 "output_truncated": bool(getattr(run, "output_truncated", False)),
                                 "subprocess_wall_s": getattr(run, "wall_s", None), "timeout_seconds": timeout}
            feedback = run.output.strip()
            if run.timed_out:
                feedback = f"Local test exceeded {timeout} s\n" + feedback
                verdict = (False, "timeout", "test")
            elif run.returncode == 0:
                verdict = (True, "passed", "test")
            elif run.returncode in (126, 127):
                verdict = (None, "infra_error", "infra")
            elif _COMPILE_ERROR in feedback:
                feedback = feedback.replace(_COMPILE_ERROR, "").strip()
                verdict = (False, "compile_error", "compile")
            elif task.language == "js" and "SyntaxError:" in feedback:
                verdict = (False, "syntax_error", "static")
            else:
                verdict = (False, "test_failed", "test")
    except (OSError, RuntimeError) as exc:
        execution_details["error_type"] = type(exc).__name__
        feedback, verdict = f"Local subprocess error: {exc}", (None, "infra_error", "infra")
    passed, status, phase = verdict
    result = _result(passed, status, feedback, started, phase)
    result["execution"] = execution_details
    return result


def test_task(task: Task, code: str, runner: str, timeout: int = 40, *,
              stage: str = "feedback") -> dict[str, Any]:
    """Public feedback is the default. Final grades must never drive repair."""
    if stage not in ("feedback", "final"):
        raise ValueError(f"Unknown evaluation stage: {stage}")
    if stage == "final" and task.final_evaluator != "legacy_assert" and runner != "none":
        if runner != "local":
            raise ValueError("Official grading requires the local runner")
        from function_benchmarks import grade_official
        verdict = grade_official(task, code, timeout=timeout)
    else:
        verdict = _test_task(task, code, runner, timeout, stage=stage)
    verdict["stage"] = stage
    verdict["sealed"] = stage == "final"
    if stage == "final":
        # Logs may contain assertions or expected outputs. Keep them out of the
        # only field that generation/repair consumes, even on infrastructure errors.
        if verdict.get("feedback"):
            verdict["sealed_diagnostics"] = verdict["feedback"]
        verdict["feedback"] = ""
        verdict["grading_wall_s"] = verdict["test_wall_s"]
    else:
        verdict["feedback_scope"] = "public_tests" if task.feedback_test_source.strip() else "syntax_compile_only"
        if verdict["passed"] is True:
            verdict["status"] = "feedback_passed"
    return verdict


def grade_final(task: Task, code: str, runner: str, timeout: int = 300) -> dict[str, Any]:
    """One sealed submission, called only after the source/repair decision freezes."""
    return test_task(task, code, runner, timeout, stage="final")


def category(status: str, feedback: str) -> str:
    if status in ("passed", "feedback_passed", "not_tested", "infra_error", "syntax_error",
                  "output_format_error", "timeout", "empty_code"):
        return status
    message = feedback.lower()
    if ("assertionerror" in message or "panicked at" in message
            or ("assertion" in message and ("failed" in message or "left == right" in message))):
        return "assertion_failure"
    if ("nameerror" in message or "referenceerror" in message or "e0425" in message
            or "cannot find value" in message or "was not declared in this scope" in message):
        return "identifier_error"
    if ("typeerror" in message or "e0308" in message or "mismatched types" in message
            or "invalid conversion" in message):
        return "type_error"
    if status == "compile_error":
        return "compile_error"
    return "runtime_or_logic_error"
