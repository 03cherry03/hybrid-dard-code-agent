"""Decoder-blind extraction and a sealed, offline grading fairness audit.

V2 selects the first Markdown fenced block (including an unclosed block), or
the entire response when there is no fence. It only selects a source substring:
it never repairs code, adds imports, chooses a passing block, or consults decoder
diagnostics. Offsets refer to ``prefix + raw_output``, in Unicode characters.
Runtime audit fixtures are synthetic; no benchmark oracle is loaded or copied.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Callable, Mapping

from datasets_code import Task
import evaluation

EXTRACTION_VERSION = "v2"
LANGUAGE_TAGS = {"py": "python", "js": "javascript", "cpp": "cpp", "rs": "rust"}
_ALIASES = {"py": {"python", "py", "python3"}, "js": {"javascript", "js"},
            "cpp": {"cpp", "c++", "cc", "cxx"}, "rs": {"rust", "rs"}}
_OPEN_FENCE = re.compile(r"(?m)^[ \t]*(?P<marker>`{3,}|~{3,})(?P<info>[^\r\n]*)(?:\r?\n|$)")
_LEGACY_FENCE = re.compile(r"```(?:[A-Za-z][\w.+#-]*)?[ \t]*\r?\n(.*?)```", re.S)
_ALGORITHMIC = {"empty_code", "output_format_error", "syntax_error", "compile_error",
                "test_failed", "timeout"}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _text_sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass(frozen=True)
class Extraction:
    version: str
    code: str = field(repr=False)
    start: int
    end: int
    restored_text_sha256: str
    raw_output_sha256: str
    prefix_sha256: str
    prefix_characters: int
    raw_characters: int
    warnings: tuple[str, ...] = ()
    fence_tag: str | None = None

    def receipt(self) -> dict:
        """No generated source, task prompt, tests, or expected outputs."""
        return {"version": self.version, "source_span": {"start": self.start, "end": self.end,
                "coordinate_system": "prefix_plus_raw_output_unicode_characters"},
                "source_is_exact_substring": self.version == "v2",
                "code_sha256": _text_sha(self.code), "code_characters": len(self.code),
                "restored_text_sha256": self.restored_text_sha256,
                "raw_output_sha256": self.raw_output_sha256, "prefix_sha256": self.prefix_sha256,
                "prefix_characters": self.prefix_characters, "raw_characters": self.raw_characters,
                "raw_empty": self.raw_characters == 0, "extracted_empty": not self.code.strip(),
                "fence_tag": self.fence_tag, "warnings": list(self.warnings)}


def extract_code(raw_output: str, *, prefix: str = "", language: str,
                 version: str = EXTRACTION_VERSION) -> Extraction:
    """Apply a fixed, decoder-independent presentation rule, never code repair.

    First block wins regardless of its language tag or correctness. A closing
    fence has the same marker, at least as long as the opener. In addition to
    a standalone closing line, a terminal inline marker at EOF is accepted
    only after an opening fence was found. Empty blocks stay empty. Missing
    closing fences end at EOF. Bare source is never stripped of backticks.
    A legacy receipt's span locates its unmodified payload, while legacy's
    historical whitespace stripping/trailing newline are retained in ``code``.
    """
    if not isinstance(raw_output, str) or not isinstance(prefix, str):
        raise TypeError("raw_output and prefix must be strings")
    if language not in LANGUAGE_TAGS or version not in {"legacy", "v2"}:
        raise ValueError("Unsupported language or extraction version")
    restored = prefix + raw_output
    warnings: list[str] = []
    tag = None
    if version == "legacy":
        code = evaluation.strip_code(restored)
        stripped = restored.strip()
        leading = len(restored) - len(restored.lstrip())
        match = _LEGACY_FENCE.search(stripped)
        original_payload = match.group(1) if match else stripped
        payload = original_payload.strip()
        start = leading + (match.start(1) if match else 0) + len(original_payload) - len(original_payload.lstrip())
        end = start + len(payload)
        warnings.append("legacy_whitespace_and_trailing_newline_normalization")
        if "```" in restored and not match:
            warnings.append("unmatched_fence_preserved")
    else:
        openings = (m for m in _OPEN_FENCE.finditer(restored)
                    if not (m.group("marker").startswith("`") and "`" in m.group("info")))
        opening = next(openings, None)
        if opening is None:
            start, end = 0, len(restored)
        else:
            marker = opening.group("marker")
            info = opening.group("info").strip()
            tag = info.split()[0].lower() if info else ""
            if tag and tag not in _ALIASES[language]:
                warnings.append("fence_language_mismatch")
            start = opening.end()
            closing_pattern = re.compile(r"(?m)^[ \t]*" + re.escape(marker[0]) +
                                         "{" + str(len(marker)) + r",}[ \t]*(?:\r?\n|$)")
            closing = closing_pattern.search(restored, start)
            if closing is None:
                terminal_marker = re.compile(re.escape(marker[0]) + "{" + str(len(marker)) + r",}[ \t\r\n]*\Z")
                closing = terminal_marker.search(restored, start)
                if closing is not None:
                    warnings.append("terminal_inline_closing_fence")
            end = closing.start() if closing else len(restored)
            if _OPEN_FENCE.search(restored[start:end]):
                warnings.append("fence_marker_inside_selected_body")
            if closing is None:
                warnings.append("unclosed_fence")
            else:
                if _OPEN_FENCE.search(restored, closing.end()):
                    warnings.append("multiple_fenced_blocks_first_selected")
                if restored[closing.end():].strip():
                    warnings.append("text_after_selected_block_discarded")
            if restored[:opening.start()].strip():
                warnings.append("text_before_selected_block_discarded")
        code = restored[start:end]
    if not raw_output.strip():
        warnings.append("empty_raw_output")
    if not code.strip():
        warnings.append("empty_extracted_code")
    return Extraction(version, code, start, end, _text_sha(restored), _text_sha(raw_output),
                      _text_sha(prefix), len(prefix), len(raw_output), tuple(warnings), tag)


@dataclass(frozen=True)
class BlindGradeRequest:
    task: Task = field(repr=False)
    extraction: Extraction
    timeout: float
    runner: str = "local"

    @property
    def code(self) -> str:
        return self.extraction.code

    def receipt(self) -> dict:
        identity = {"task_sha256": _digest(asdict(self.task)), "code_sha256": _text_sha(self.code),
                    "timeout": self.timeout, "runner": self.runner}
        return {**identity, "request_sha256": _digest(identity), "language": self.task.language,
                "evaluator": self.task.final_evaluator, "extraction": self.extraction.receipt()}


def prepare_grade_request(task: Task, raw_output: str, *, prefix: str = "", timeout: float = 300,
                          extraction_version: str = EXTRACTION_VERSION,
                          decoder_metadata: Mapping | None = None) -> BlindGradeRequest:
    """Metadata is deliberately neither read nor forwarded to the evaluator.

    Incomplete generation, verified-token counts and forced filling are separate
    generation-quality metrics. They cannot change a visible source's grade.
    """
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a finite positive number")
    extraction = extract_code(raw_output, prefix=prefix, language=task.language, version=extraction_version)
    return BlindGradeRequest(task, extraction, timeout)


def execute_grade_request(request: BlindGradeRequest, *, grader: Callable | None = None) -> dict:
    """Return the original sealed verdict; callers must keep diagnostics sealed."""
    return (grader or evaluation.grade_final)(request.task, request.code, request.runner,
                                              timeout=request.timeout)


def classify_result(record: Mapping) -> str:
    """Do not silently count infrastructure or unattributed deadlines as wrong.

    The old worker clips its generation timeout to the whole-run deadline but
    records both as GenerationDeadlineExceeded. Without a scope receipt these
    cannot retrospectively be treated as known candidate budget failures.
    """
    status, passed = record.get("status"), record.get("passed")
    if record.get("infrastructure_error") or status == "infra_error":
        return "infrastructure_failure"
    if status == "not_tested":
        return "unassessed"
    if status == "passed" and passed is True:
        return "success"
    if passed is False and status in _ALGORITHMIC:
        return "algorithmic_failure"
    if passed is False and status == "ForwardBudgetExceeded":
        return "budget_failure"
    if passed is False and status == "GenerationDeadlineExceeded":
        return "deadline_failure_unattributed"
    return "invalid_or_unknown_result"


def compare_extractions(raw_output: str, *, prefix: str = "", language: str) -> dict:
    legacy = extract_code(raw_output, prefix=prefix, language=language, version="legacy")
    revised = extract_code(raw_output, prefix=prefix, language=language, version="v2")
    return {"code_equal": legacy.code == revised.code,
            "code_equal_ignoring_outer_whitespace": legacy.code.strip() == revised.code.strip(),
            "legacy": legacy.receipt(), "v2": revised.receipt()}


def audit_case_artifacts(case_roots=(), *, candidate_ids=None) -> dict:
    """Read completed validation/smoke cases, without regrading or oracle access."""
    counts: Counter = Counter()
    per_candidate = defaultdict(Counter)
    changed = []
    errors = []
    seen = set()
    selected_candidates = set(candidate_ids) if candidate_ids is not None else None
    for root in case_roots:
        root = Path(root)
        if not root.is_dir():
            errors.append({"path": str(root), "error": "case_root_missing"})
            continue
        for path in sorted(root.glob("**/cases/*/*.json")):
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            try:
                payload = path.read_bytes()
                row = json.loads(payload)
                if row.get("split") not in {"validation", "smoke"}:
                    counts["non_validation_or_smoke_skipped"] += 1
                    continue
                if selected_candidates is not None and row.get("candidate_id") not in selected_candidates:
                    counts["other_candidates_skipped"] += 1
                    continue
                outcome = classify_result(row)
                candidate = row["candidate_id"]
                counts["cases"] += 1
                counts[outcome] += 1
                per_candidate[candidate][outcome] += 1
                # A failed generation has no submission: its saved empty code
                # is not evidence that the shared extraction rule differed.
                if row["status"] in {"ForwardBudgetExceeded", "GenerationDeadlineExceeded"}:
                    counts["generation_failures_without_submission"] += 1
                    continue
                comparison = compare_extractions(row["raw_output"], prefix=row["prefix"], language=row["language"])
                old_code = evaluation.strip_code(row["prefix"] + row["raw_output"])
                legacy_matches = old_code == row["code"]
                saved_hash_matches = row.get("code_sha256") == _digest(row["code"])
                if not legacy_matches or not saved_hash_matches:
                    errors.append({"path": str(path), "error": "saved_legacy_extraction_or_hash_mismatch",
                                   "legacy_matches": legacy_matches, "saved_hash_matches": saved_hash_matches})
                if not comparison["code_equal"]:
                    counts["extraction_byte_changes"] += 1
                    if not comparison["code_equal_ignoring_outer_whitespace"]:
                        counts["extraction_payload_changes"] += 1
                        per_candidate[candidate]["extraction_payload_changes"] += 1
                    else:
                        counts["extraction_outer_whitespace_only_changes"] += 1
                        per_candidate[candidate]["extraction_outer_whitespace_only_changes"] += 1
                    changed.append({"path": str(path), "case_sha256": hashlib.sha256(payload).hexdigest(),
                                    "candidate_id": candidate, "language": row["language"],
                                    "task_key": row.get("task_key"), "status": row["status"],
                                    "comparison": comparison})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append({"path": str(path), "error": "malformed_case", "error_type": type(exc).__name__})
    return {"status": "passed" if not errors else "failed", "counts": dict(counts),
            "candidate_filter": sorted(selected_candidates) if selected_candidates is not None else None,
            "by_candidate": {key: dict(value) for key, value in sorted(per_candidate.items())},
            "changed_cases": changed, "errors": errors,
            "scope": "completed validation/smoke cases only; aborted infrastructure attempts may have no case"}


def _worker_source_checks(parsed: ast.Module) -> dict:
    # Compare the executable helper's full AST, not a matching fragment anywhere
    # in a file. A semantic change requires a new review and audit receipt.
    expected_helper = ast.parse('''
def extract_for_grading(task, raw_output, prefix, plan):
    if plan.get("extraction_version") == "v2":
        from grading_audit import prepare_grade_request
        request = prepare_grade_request(task, raw_output, prefix=prefix,
            timeout=plan["grading_timeout"], extraction_version="v2")
        return request.code, request.receipt()
    from evaluation import strip_code
    return strip_code(prefix + raw_output), {"version": "legacy_strip_code"}
''').body[0]
    functions = {node.name: node for node in parsed.body if isinstance(node, ast.FunctionDef)}
    helper = functions.get("extract_for_grading")
    exact_helper = helper is not None and ast.dump(helper) == ast.dump(expected_helper)
    calls = [node for node in ast.walk(parsed) if isinstance(node, ast.Call)]
    assignments = [node for node in ast.walk(functions["run"])
                   if isinstance(node, ast.Assign) and any(
                       isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store) and target.id == "code"
                       for left in node.targets for target in ast.walk(left))]
    expected_assignment = ast.parse('code, extraction = extract_for_grading(task, text, prefix, plan) if generation else ("", None)').body[0]
    common_assignment = len(assignments) == 1 and ast.dump(assignments[0]) == ast.dump(expected_assignment)
    grade_calls = [node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "grading_function"]
    grading = len(grade_calls) == 1 and ast.dump(grade_calls[0]) == ast.dump(
        ast.parse('grading_function(task, code, "local", timeout=timeout)', mode="eval").body)
    observed = functions.get("observed_grade")
    grading = grading and observed is not None and not any(
        isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == "code"
        for node in ast.walk(observed))
    submissions = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "submit"]
    common_submit = len(submissions) == 1 and ast.dump(submissions[0]) == ast.dump(
        ast.parse('pool.submit(grade, task, code, context)', mode="eval").body)
    return {"worker_exact_versioned_extraction_helper": exact_helper,
            "worker_common_code_assignment": common_assignment,
            "worker_submits_unmodified_common_code": common_submit,
            "worker_blind_grading_call": grading}


def _source_contract() -> dict:
    import function_benchmarks
    root = Path(__file__).resolve().parent
    worker = root / "scripts/iterative_eval_worker.py"
    parsed = ast.parse(worker.read_text())
    signatures = {"grade_final": list(inspect.signature(evaluation.grade_final).parameters),
                  "grade_official": list(inspect.signature(function_benchmarks.grade_official).parameters)}
    grading_functions = [evaluation.grade_final, evaluation.test_task, evaluation._test_task,
                         evaluation._program, function_benchmarks.grade_official,
                         function_benchmarks._official_solution]
    no_reextraction = all(not any(isinstance(node, ast.Call) and
        ((isinstance(node.func, ast.Name) and node.func.id == "strip_code") or
         (isinstance(node.func, ast.Attribute) and node.func.attr == "strip_code"))
        for node in ast.walk(ast.parse(inspect.getsource(function)))) for function in grading_functions)
    checks = {**_worker_source_checks(parsed), "grader_does_not_reextract_code": no_reextraction,
              "grade_final_arguments": signatures["grade_final"] == ["task", "code", "runner", "timeout"],
              "grade_official_arguments": signatures["grade_official"] == ["task", "code", "timeout"]}
    files = [Path(__file__), Path(evaluation.__file__), Path(function_benchmarks.__file__), worker]
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
            "signatures": signatures, "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                                          for path in files},
            "scope": "full extraction-helper AST, code assignment and grader call boundaries; not a formal proof of all execution paths"}


def _synthetic_cases(language: str):
    code = {"py": "def identity(x):\n    return x\n", "js": "function identity(x) { return x; }\n",
            "cpp": "long identity(long x) { return x; }\n", "rs": "fn identity(x: i64) -> i64 { x }\n"}[language]
    wrong = code.replace("return x", "return 0") if language != "rs" else code.replace("{ x }", "{ 0 }")
    tests = {"py": "assert identity(7) == 7\nassert identity(-3) == -3",
             "js": "if (identity(7) !== 7 || identity(-3) !== -3) throw Error('synthetic identity mismatch');",
             "cpp": "int main() { return identity(7) == 7 && identity(-3) == -3 ? 0 : 1; }",
             "rs": "fn main() { assert_eq!(identity(7), 7); assert_eq!(identity(-3), -3); }"}[language]
    task = Task("grading_audit_synthetic", language, "synthetic_identity", 0,
                "Return the supplied integer.", tests, "identity")
    fence = "\n```" + LANGUAGE_TAGS[language] + "\n"
    return task, [("plain_valid", code, "", True), ("fenced_valid", fence + code + "```\n", "", True),
                  ("prefilled_valid", code + "```\n", fence, True),
                  ("unclosed_prefill_valid", code, fence, True),
                  ("wrong_body", wrong + "```\n", fence, False),
                  ("empty_fenced", "```\n", fence, False),
                  ("prose_only", "Here is the requested function.", "", False)]


def audit_grading(*, data_root: Path | str = "/data/hybrid-dard", case_roots=(), candidate_ids=None,
                  languages=("py", "js", "cpp", "rs"), runtime_checks: bool = True,
                  timeout: float = 40) -> dict:
    """Run synthetic grade metamorphisms and optional historical extraction audit."""
    languages = tuple(sorted(set(languages)))
    if not languages or any(language not in LANGUAGE_TAGS for language in languages):
        raise ValueError("Expected supported audit languages")
    started = time.time()
    report = {"schema_version": 1, "audit": "decoder_blind_grading", "extraction_version": EXTRACTION_VERSION,
              "started_unix": started, "source_contract": _source_contract(),
              "historical_cases": audit_case_artifacts(case_roots, candidate_ids=candidate_ids), "runtime_checks": [],
              "data_policy": "synthetic runtime fixtures only; no benchmark code, prompt, tests or diagnostics copied",
              "limitations": ["Synthetic checks do not establish benchmark test completeness or absence of model contamination.",
                              "Official evaluator environments/oracles are not exercised by synthetic legacy_assert fixtures.",
                              "Equal extraction and grading do not make unequal generation budgets a fair decoder comparison.",
                              "Outer whitespace changes may affect indentation-sensitive code; byte changes also require sensitivity checks.",
                              "Generation incompleteness/forced filling are separate metrics, never grading overrides."]}
    if runtime_checks:
        try:
            report["environment"] = evaluation.preflight_local(set(languages), data_root=Path(data_root))
        except (OSError, RuntimeError, ValueError) as exc:
            report["runtime_environment_error"] = {"error_type": type(exc).__name__}
        else:
            for language in languages:
                task, fixtures = _synthetic_cases(language)
                for name, raw, prefix, expected in fixtures:
                    requests = [prepare_grade_request(task, raw, prefix=prefix, timeout=timeout, decoder_metadata=metadata)
                                for metadata in ({"decoder": "standard", "incomplete": False, "forced_tokens": 0},
                                                 {"decoder": "dard", "incomplete": True, "forced_tokens": 999,
                                                  "verified_success": False})]
                    verdicts = [execute_grade_request(request) for request in requests]
                    identical_requests = requests[0].receipt() == requests[1].receipt()
                    outcomes = [{"passed": verdict.get("passed"), "status": verdict.get("status"),
                                 "classification": classify_result(verdict)} for verdict in verdicts]
                    safe = all(verdict.get("sealed") is True and verdict.get("feedback") == "" for verdict in verdicts)
                    passed = identical_requests and outcomes[0] == outcomes[1] and safe and all(
                        verdict.get("passed") is expected for verdict in verdicts)
                    report["runtime_checks"].append({"fixture": name, "language": language,
                        "status": "passed" if passed else "failed", "expected_passed": expected,
                        "identical_requests": identical_requests, "sealed_feedback_empty": safe,
                        "request": requests[0].receipt(), "outcomes": outcomes})
    checks_pass = (report["source_contract"]["status"] == "passed" and
                   report["historical_cases"]["status"] == "passed" and
                   "runtime_environment_error" not in report and
                   all(row["status"] == "passed" for row in report["runtime_checks"]))
    report["status"] = ("passed" if runtime_checks else "incomplete") if checks_pass else "failed"
    report["passed"] = report["status"] == "passed"
    report["completed_unix"] = time.time()
    return report


def write_audit_report(path: Path | str, **kwargs) -> dict:
    report = audit_grading(**kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/data/hybrid-dard"))
    parser.add_argument("--case-root", action="append", type=Path, default=[])
    parser.add_argument("--candidate-id", action="append", default=None)
    parser.add_argument("--languages", nargs="+", choices=sorted(LANGUAGE_TAGS), default=sorted(LANGUAGE_TAGS))
    parser.add_argument("--skip-runtime-checks", action="store_true")
    parser.add_argument("--timeout", type=float, default=40)
    args = parser.parse_args(argv)
    report = write_audit_report(args.output, data_root=args.data_root, case_roots=args.case_root,
                                candidate_ids=args.candidate_id,
                                languages=args.languages, runtime_checks=not args.skip_runtime_checks,
                                timeout=args.timeout)
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "synthetic_checks": len(report["runtime_checks"])}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
