"""Decoder metadata must not affect source selection, requests or verdicts."""
from dataclasses import asdict
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from datasets_code import Task
import evaluation
import grading_audit as audit


def fixture(language="py"):
    return Task("audit_fixture", language, "fixture/0", 0, "PUBLIC_PROMPT_SENTINEL",
                "assert identity(7) == 7 # SEALED_ORACLE_SENTINEL", "identity")


class ExtractionTests(unittest.TestCase):
    def test_prefix_restored_once_and_exact_original_span(self):
        raw = "def identity(x):\n    return x\n```\nExplanation outside the fence."
        prefix = "\n```python\n"
        extracted = audit.extract_code(raw, prefix=prefix, language="py")
        self.assertEqual(extracted.code, "def identity(x):\n    return x\n")
        self.assertEqual((prefix + raw)[extracted.start:extracted.end], extracted.code)
        self.assertIn("text_after_selected_block_discarded", extracted.warnings)
        self.assertEqual(extracted.start, len(prefix))

    def test_closed_and_unclosed_fences_have_identical_code(self):
        code = "def identity(x):\n    return x\n"
        opened = audit.extract_code("```python\n" + code, language="py")
        closed = audit.extract_code("```python\n" + code + "```", language="py")
        self.assertEqual(opened.code, closed.code)
        self.assertIn("unclosed_fence", opened.warnings)
        self.assertNotIn("unclosed_fence", closed.warnings)
        self.assertNotEqual(evaluation.strip_code("```python\n" + code), opened.code)

    def test_first_block_selected_even_when_wrong_language_or_empty(self):
        text = "```js\n```\n```python\ndef identity(x): return x\n```"
        extracted = audit.extract_code(text, language="py")
        self.assertEqual(extracted.code, "")
        self.assertIn("fence_language_mismatch", extracted.warnings)
        self.assertIn("multiple_fenced_blocks_first_selected", extracted.warnings)
        self.assertIn("empty_extracted_code", extracted.warnings)

    def test_unfenced_whitespace_and_broken_code_are_not_repaired(self):
        for source in ("  def f(:\n\treturn missing_name", "Here is the requested function.", "\n"):
            self.assertEqual(audit.extract_code(source, language="py").code, source)

    def test_closing_marker_must_match_length_and_occupy_own_line(self):
        text = "````python\nx = '```'\n```\n~~~~\n````\n"
        extracted = audit.extract_code(text, language="py")
        self.assertEqual(extracted.code, "x = '```'\n```\n~~~~\n")

    def test_tilde_crlf_and_unicode_offsets(self):
        text = "설명\r\n~~~python\r\nx = '한글'\r\n~~~\r\n"
        extracted = audit.extract_code(text, language="py")
        self.assertEqual(extracted.code, "x = '한글'\r\n")
        self.assertEqual(text[extracted.start:extracted.end], extracted.code)

    def test_empty_raw_with_prefill_stays_empty(self):
        extracted = audit.extract_code("", prefix="\n```cpp\n", language="cpp")
        self.assertEqual(extracted.code, "")
        self.assertIn("empty_raw_output", extracted.warnings)
        self.assertIn("empty_extracted_code", extracted.warnings)

    def test_terminal_inline_fence_does_not_regress_complete_python(self):
        for code in ("def identity(x):\n    return x", "def identity(x):\n    return (x + 0)"):
            text = code + "```"
            revised = audit.extract_code(text, prefix="```python\n", language="py")
            self.assertEqual(revised.code, code)
            self.assertIn("terminal_inline_closing_fence", revised.warnings)
            self.assertEqual(revised.code.strip(), evaluation.strip_code("```python\n" + text).strip())

    def test_javascript_template_literals_and_bare_backticks_preserved(self):
        bare = "const text = `value ${1}`;\nconst escaped = `\\`\\`\\``;"
        self.assertEqual(audit.extract_code(bare, language="js").code, bare)
        self.assertEqual(audit.extract_code(bare + "```", language="js").code, bare + "```")
        fenced = audit.extract_code(bare + "```", prefix="```javascript\n", language="js")
        self.assertEqual(fenced.code, bare)
        # A delimiter inside a valid string is not a terminal closing marker.
        inner = 'const text = "```";\n'
        self.assertEqual(audit.extract_code(inner, prefix="```javascript\n", language="js").code, inner)

    def test_inline_closing_never_repairs_leading_indentation(self):
        code = "    def identity(x):\n        return x"
        self.assertEqual(audit.extract_code(code + "```", prefix="```python\n", language="py").code, code)


class BlindRequestTests(unittest.TestCase):
    def test_metadata_not_read_and_never_forwarded_to_grader(self):
        class PoisonMetadata(dict):
            def get(self, *args):
                raise AssertionError("Decoder metadata was inspected")
            def __iter__(self):
                raise AssertionError("Decoder metadata was inspected")
        task = fixture()
        first = audit.prepare_grade_request(task, "def identity(x): return x", timeout=5,
                                             decoder_metadata={"decoder": "standard", "incomplete": False})
        second = audit.prepare_grade_request(task, "def identity(x): return x", timeout=5,
                                              decoder_metadata=PoisonMetadata(decoder="dard", incomplete=True))
        calls = []
        def grader(*args, **kwargs):
            calls.append((args, kwargs))
            return {"passed": True, "status": "passed", "feedback": "", "sealed": True}
        self.assertEqual(audit.execute_grade_request(first, grader=grader),
                         audit.execute_grade_request(second, grader=grader))
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(calls[0], ((task, first.code, "local"), {"timeout": 5}))
        self.assertEqual(first.receipt(), second.receipt())
        encoded = json.dumps(first.receipt())
        self.assertNotIn("PUBLIC_PROMPT_SENTINEL", encoded)
        self.assertNotIn("SEALED_ORACLE_SENTINEL", encoded)
        self.assertNotIn("def identity", encoded)

    def test_request_hash_binds_code_task_and_timeout(self):
        task = fixture()
        original = audit.prepare_grade_request(task, "def f(): pass", timeout=5).receipt()["request_sha256"]
        changed_code = audit.prepare_grade_request(task, "def f(): return 1", timeout=5)
        changed_time = audit.prepare_grade_request(task, "def f(): pass", timeout=6)
        changed_task = audit.prepare_grade_request(Task(**{**asdict(task), "test_source": "assert False"}),
                                                   "def f(): pass", timeout=5)
        for request in (changed_code, changed_time, changed_task):
            self.assertNotEqual(original, request.receipt()["request_sha256"])

    def test_invalid_timeout_rejected_before_execution(self):
        for timeout in (True, 0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                audit.prepare_grade_request(fixture(), "", timeout=timeout)

    def test_status_classification_does_not_conflate_infrastructure_and_candidates(self):
        cases = [(None, "infra_error", "infrastructure_failure"),
                 (None, "not_tested", "unassessed"), (True, "passed", "success"),
                 (False, "compile_error", "algorithmic_failure"),
                 (False, "timeout", "algorithmic_failure"),
                 (False, "ForwardBudgetExceeded", "budget_failure"),
                 (False, "GenerationDeadlineExceeded", "deadline_failure_unattributed"),
                 (True, "test_failed", "invalid_or_unknown_result"),
                 (1, "passed", "invalid_or_unknown_result")]
        for passed, status, expected in cases:
            with self.subTest(status=status, passed=passed):
                self.assertEqual(audit.classify_result({"passed": passed, "status": status}), expected)
        self.assertEqual(audit.classify_result({"passed": False, "status": "test_failed",
                                               "infrastructure_error": True}), "infrastructure_failure")

    def test_real_python_grading_same_code_even_when_dard_incomplete(self):
        task = fixture()
        for code, expected in [("def identity(x): return x", True),
                               ("def identity(x): return 0", False), ("", False)]:
            outcomes = []
            with patch.object(evaluation, "local_executable", return_value=sys.executable):
                for decoder in ("standard", "dard"):
                    request = audit.prepare_grade_request(task, code, timeout=5,
                        prefix="```python\n", decoder_metadata={"decoder": decoder, "incomplete": decoder == "dard"})
                    verdict = audit.execute_grade_request(request)
                    self.assertIs(verdict["passed"], expected)
                    self.assertEqual(verdict["feedback"], "")
                    self.assertTrue(verdict["sealed"])
                    outcomes.append((verdict["passed"], verdict["status"]))
            self.assertEqual(*outcomes)


class ArtifactAuditTests(unittest.TestCase):
    def test_case_reconstruction_hashes_and_private_report_without_source_or_tests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "shard-0/cases/standard/case.json"
            path.parent.mkdir(parents=True)
            raw = "def identity(x): return x # RAW_SENTINEL"
            prefix = "```python\n"
            code = evaluation.strip_code(prefix + raw)
            row = dict(candidate_id="standard", split="validation", status="syntax_error", passed=False,
                       language="py", raw_output=raw, prefix=prefix, code=code, code_sha256=audit._digest(code))
            path.write_text(json.dumps(row))
            report = audit.audit_case_artifacts([root, root])
            self.assertEqual(report["counts"]["cases"], 1)
            self.assertEqual(report["counts"]["extraction_payload_changes"], 1)
            self.assertEqual(report["status"], "passed")
            self.assertNotIn("RAW_SENTINEL", json.dumps(report))
            row["code_sha256"] = "tampered"
            path.write_text(json.dumps(row))
            self.assertEqual(audit.audit_case_artifacts([root])["status"], "failed")

    def test_holdout_skipped_and_failed_generation_not_reextracted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "cases/candidate"
            folder.mkdir(parents=True)
            (folder / "holdout.json").write_text(json.dumps({"split": "holdout"}))
            (folder / "failed.json").write_text(json.dumps({"split": "validation", "candidate_id": "candidate",
                "passed": False, "status": "ForwardBudgetExceeded"}))
            report = audit.audit_case_artifacts([root])
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["counts"]["non_validation_or_smoke_skipped"], 1)
            self.assertEqual(report["counts"]["generation_failures_without_submission"], 1)

    def test_atomic_report_is_private_and_skipped_runtime_never_claims_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit/report.json"
            report = audit.write_audit_report(path, runtime_checks=False)
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(path.read_text()), report)

    def test_checked_source_call_boundaries_are_blind(self):
        self.assertEqual(audit._source_contract()["status"], "passed")

    def test_source_contract_rejects_decoder_metadata_and_rewritten_grade_input(self):
        worker = Path(audit.__file__).parent / "scripts/iterative_eval_worker.py"
        original = worker.read_text()
        parsed = ast.parse(original)
        helper = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "extract_for_grading")
        call = next(node for node in ast.walk(helper) if isinstance(node, ast.Call) and
                    isinstance(node.func, ast.Name) and node.func.id == "prepare_grade_request")
        call.keywords.append(ast.keyword(arg="decoder_metadata", value=ast.Name(id="plan", ctx=ast.Load())))
        self.assertFalse(audit._worker_source_checks(parsed)["worker_exact_versioned_extraction_helper"])
        parsed = ast.parse(original)
        observed = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "observed_grade")
        observed.body.insert(0, ast.parse('code = "replaced source"').body[0])
        self.assertFalse(audit._worker_source_checks(parsed)["worker_blind_grading_call"])

    def test_live_worker_helper_uses_common_v2_substring_with_metadata_ignored(self):
        from scripts.iterative_eval_worker import extract_for_grading
        raw = "def identity(x): return x\n"
        prefix = "```python\n"
        outputs = [extract_for_grading(fixture(), raw, prefix, {"extraction_version": "v2", "grading_timeout": 5,
                   "decoder": decoder, "incomplete": decoder == "dard"}) for decoder in ("standard", "dard")]
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0][0], raw)


if __name__ == "__main__":
    unittest.main()
