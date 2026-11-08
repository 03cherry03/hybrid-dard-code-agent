"""Full-function pilot tasks from HumanEval, MBPP and official MultiPL-E translations.

All backends receive the same task prompt and harness. Translated MBPP tasks are
restricted to the native MBPP test IDs (11..510); translation coverage varies by
language, so source_id is retained for comparisons on intersecting problem sets.
"""
from __future__ import annotations

import ast
import copy
import doctest
import gzip
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

HUMANEVAL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
MULTIPLE_HF = "https://datasets-server.huggingface.co/rows"
MULTIPLE_SOURCE = "https://huggingface.co/datasets/nuprl/MultiPL-E"
MBPP = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
LANGUAGE_NAMES = {"py": "Python", "rs": "Rust", "js": "JavaScript", "cpp": "C++"}

# Configurations published at the official nuprl/MultiPL-E repository.
SUITES = {
    "humaneval": {"language": "py", "family": "humaneval", "source_config": "test", "source_url": HUMANEVAL},
    "mbpp": {"language": "py", "family": "mbpp", "source_config": "test", "source_url": MBPP},
    **{f"multiple_{lang}": {"language": lang, "family": "humaneval",
                           "source_config": f"humaneval-{lang}", "source_url": MULTIPLE_SOURCE}
       for lang in ("rs", "js", "cpp")},
    **{f"multiple_mbpp_{lang}": {"language": lang, "family": "mbpp",
                                "source_config": f"mbpp-{lang}", "source_url": MULTIPLE_SOURCE}
       for lang in ("rs", "js", "cpp")},
    "humaneval_plus": {"language": "py", "family": "humaneval", "source_config": "HumanEvalPlus-v0.1.10", "source_url": "https://github.com/evalplus/humanevalplus_release"},
    "mbpp_plus": {"language": "py", "family": "mbpp", "source_config": "MbppPlus-v0.2.0", "source_url": "https://github.com/evalplus/mbppplus_release"},
    "bigcodebench_instruct": {"language": "py", "family": "bigcodebench", "source_config": "instruct-v0.1.4", "source_url": "https://huggingface.co/datasets/bigcode/bigcodebench"},
}

# Active comparison protocol. Library-use tasks remain readable for historical
# artifacts, but are not part of code-generation experiments or their setup.
CODE_GENERATION_SUITES = (
    "humaneval", "mbpp", "multiple_rs", "multiple_js", "multiple_cpp",
    "multiple_mbpp_rs", "multiple_mbpp_js", "multiple_mbpp_cpp",
    "humaneval_plus", "mbpp_plus",
)


@dataclass(frozen=True)
class Task:
    dataset: str
    language: str
    task_id: str
    number: int
    prompt: str
    test_source: str
    required_name: str | None = None
    family: str = ""
    source_config: str = ""
    source_url: str = ""
    source_id: str = ""
    context_source: str = ""
    # Only this explicitly public material may provide repair feedback.
    feedback_test_source: str = ""
    final_evaluator: str = "legacy_assert"
    evaluator_metadata: dict = field(default_factory=dict)


def fetch_cached(url: str, path: Path, *, allow_download: bool = False) -> bytes:
    if path.exists() and path.stat().st_size > 0:
        return path.read_bytes()
    if not allow_download:
        raise FileNotFoundError(f"Dataset is not prepared: {path}. Run the explicit data preparation command first.")
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "local-dard-bench/2.0"})
    with urllib.request.urlopen(req, timeout=150) as response:
        data = response.read()
    if not data:
        raise RuntimeError(f"Downloaded empty data from {url}")
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)
    return data


def _entries(contents: bytes):
    return [json.loads(line) for line in contents.decode("utf-8").splitlines() if line.strip()]


def _python_context(prompt: str, entry_point: str) -> str:
    """Keep imports/helpers supplied by HumanEval, excluding the target."""
    tree = ast.parse(prompt)
    target = next((node for node in tree.body
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and node.name == entry_point), None)
    if target is None:
        raise ValueError(f"HumanEval prompt has no entry point {entry_point!r}")
    first_line = min([target.lineno] + [d.lineno for d in target.decorator_list])
    return "".join(prompt.splitlines(keepends=True)[:first_line - 1]).rstrip()


def public_feedback_tests(prompt: str, entry_point: str | None = None) -> str:
    """Extract only examples already visible in a Python task's public prompt.

    No reference implementation, base tests, expanded tests or expected-output
    oracle is accepted as input. Ambiguous prose/flattened examples are skipped.
    """
    documents = []
    try:
        tree = ast.parse(prompt)
        module_doc = ast.get_docstring(tree)
        if module_doc:
            documents.append(module_doc)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (entry_point is None or node.name == entry_point):
                doc = ast.get_docstring(node)
                if doc:
                    documents.append(doc)
    except SyntaxError:
        pass
    # Plain instructions can contain fenced assertions. Docstrings are preferred
    # so closing source delimiters never become doctest expected output.
    if not documents:
        documents = [prompt]
    assertions = []
    doctest_documents = []
    for document in documents:
        try:
            examples = doctest.DocTestParser().get_examples(document)
        except ValueError:
            examples = []
        if examples:
            # All sample source must be Python, and literal Markdown delimiters
            # must never be interpreted as expected program output.
            valid = True
            for example in examples:
                try:
                    parsed_example = ast.parse(example.source)
                except SyntaxError:
                    valid = False
                    continue
                if "```" in example.want:
                    valid = False
                # Some public prompts show a comparison after >>> but omit
                # its True/False output. Doctest would demand silence and
                # reject every implementation. Keep such incomplete examples
                # out of repair feedback rather than inventing an outcome.
                if (not example.want and len(parsed_example.body) == 1
                        and isinstance(parsed_example.body[0], ast.Expr)
                        and isinstance(parsed_example.body[0].value, ast.Compare)):
                    valid = False
            if valid:
                doctest_documents.append(document)
        lines = document.splitlines()
        index = 0
        while index < len(lines):
            if not re.match(r"^\s*assert\s+", lines[index]):
                index += 1
                continue
            statement = lines[index].strip()
            consumed = 1
            while consumed <= 20:
                try:
                    parsed = ast.parse(statement)
                    if len(parsed.body) == 1 and isinstance(parsed.body[0], ast.Assert):
                        assertions.append(ast.unparse(parsed.body[0]))
                    break
                except SyntaxError:
                    if index + consumed >= len(lines):
                        break
                    statement += "\n" + lines[index + consumed].strip()
                    consumed += 1
            index += consumed
    parts = list(dict.fromkeys(assertions))
    if doctest_documents:
        parts.extend([
            "import doctest as _public_doctest",
            "_public_runner = _public_doctest.DocTestRunner()",
            f"for _public_document in {list(dict.fromkeys(doctest_documents))!r}:",
            "    _public_test = _public_doctest.DocTestParser().get_doctest(_public_document, globals(), 'public_prompt_examples', '<public-prompt>', 0)",
            "    _public_runner.run(_public_test)",
            "assert _public_runner.failures == 0, 'Public prompt example failed'",
        ])
    return "\n".join(parts) + ("\n" if parts else "")


def load_humaneval(cache_dir: Path, *, allow_download: bool = False) -> list[Task]:
    rows = _entries(gzip.decompress(fetch_cached(HUMANEVAL, cache_dir / "HumanEval.jsonl.gz", allow_download=allow_download)))
    tasks = []
    for row in rows:
        number = int(row["task_id"].split("/")[-1])
        tests = row["test"] + "\ncheck(" + row["entry_point"] + ")\n"
        context = _python_context(row["prompt"], row["entry_point"])
        prompt = row["prompt"]
        if context:
            prompt = ("The imports/helpers before the requested function are supplied by the runner. "
                      "Return the complete requested function without repeating those helpers.\n\n" + prompt)
        tasks.append(Task("humaneval", "py", row["task_id"], number, prompt, tests,
                          row["entry_point"], family="humaneval", source_config="test",
                          source_url=HUMANEVAL, source_id=f"HumanEval/{number}", context_source=context,
                          feedback_test_source=public_feedback_tests(row["prompt"], row["entry_point"])))
    if len(tasks) != 164 or len({task.task_id for task in tasks}) != 164:
        raise RuntimeError(f"Unexpected HumanEval size/duplicates: {len(tasks)} (expected 164 unique tasks)")
    return sorted(tasks, key=lambda t: t.number)


def _mbpp_api(reference: str, tests: list[str]) -> tuple[str, str | None]:
    """Extract tested function signatures, never reference implementation bodies.

    Helpers not called by public tests need not be exposed. Annotations, defaults
    and parameter names are API; reference decorators/docstrings are omitted.
    """
    tree = ast.parse(reference)
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for node in tree.body:
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Lambda)
                and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            functions.append(ast.fix_missing_locations(ast.FunctionDef(
                name=node.targets[0].id, args=copy.deepcopy(node.value.args),
                body=[ast.Pass()], decorator_list=[], returns=None,
                type_comment=None, type_params=[])))
    called = {node.func.id for test in tests for node in ast.walk(ast.parse(test))
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    public = [node for node in functions if node.name in called] or functions
    if not public:
        raise ValueError("MBPP reference has no callable function API")
    declarations = []
    for node in public:
        stub = copy.deepcopy(node)
        stub.decorator_list = []
        stub.body = [ast.Pass()]
        declarations.append(ast.unparse(stub))
    return "\n\n".join(declarations), public[0].name if len(public) == 1 else None


def load_mbpp(cache_dir: Path, *, allow_download: bool = False) -> list[Task]:
    rows = _entries(fetch_cached(MBPP, cache_dir / "mbpp.jsonl", allow_download=allow_download))
    tasks = []
    for row in rows:
        number = int(row["task_id"])
        if number not in range(11, 511):
            continue
        public_tests = list(row["test_list"])
        api, required = _mbpp_api(row["code"], public_tests)
        prompt = row["text"] + "\n\nRequired Python API (replace pass with your implementation):\n" + api
        tests = "\n".join([(row.get("test_setup_code") or "")] + public_tests) + "\n"
        tasks.append(Task("mbpp", "py", f"MBPP/{number}", number, prompt, tests, required,
                          family="mbpp", source_config="test", source_url=MBPP, source_id=f"MBPP/{number}",
                          feedback_test_source=public_feedback_tests(row["text"], required)))
    if len(tasks) != 500 or len({task.task_id for task in tasks}) != 500:
        raise RuntimeError(f"Unexpected MBPP test split size/duplicates: {len(tasks)} (expected 500 unique tasks)")
    return sorted(tasks, key=lambda t: t.number)


def _multiple_rows(cache_dir: Path, suite: str, config: str, *, allow_download: bool = False) -> list[dict]:
    """Read every advertised row, including datasets larger than two pages."""
    gathered: list[dict] = []
    offset = 0
    expected_total: int | None = None
    while True:
        params = urllib.parse.urlencode(dict(dataset="nuprl/MultiPL-E", config=config,
                                             split="test", offset=offset, length=100))
        doc = json.loads(fetch_cached(MULTIPLE_HF + "?" + params, cache_dir / f"{suite}_{offset}.json", allow_download=allow_download))
        if doc.get("partial") is True:
            raise RuntimeError(f"MultiPL-E {config}: dataset server returned partial data")
        if "num_rows_total" in doc:
            page_total = int(doc["num_rows_total"])
            if expected_total is not None and page_total != expected_total:
                raise RuntimeError(f"MultiPL-E {config}: row count changed during pagination")
            expected_total = page_total
        entries = doc.get("rows")
        if not isinstance(entries, list):
            raise RuntimeError(f"MultiPL-E {config}: missing rows in dataset-server response")
        for index, entry in enumerate(entries):
            if "row_idx" in entry and int(entry["row_idx"]) != offset + index:
                raise RuntimeError(f"MultiPL-E {config}: non-contiguous page at offset {offset}")
            if entry.get("truncated_cells"):
                raise RuntimeError(f"MultiPL-E {config}: truncated task fields")
            gathered.append(entry["row"])
        offset += len(entries)
        if expected_total is not None:
            if offset > expected_total or (not entries and offset < expected_total):
                raise RuntimeError(f"MultiPL-E {config}: incomplete/inconsistent pagination ({offset}/{expected_total})")
            if offset == expected_total:
                break
        elif len(entries) < 100:
            break
        if not entries:
            raise RuntimeError(f"MultiPL-E {config}: empty page before completion")
    if not gathered:
        raise RuntimeError(f"MultiPL-E {config}: no tasks returned")
    names = [row["name"] for row in gathered]
    if len(set(names)) != len(names):
        raise RuntimeError(f"MultiPL-E {config}: duplicate task names across pages")
    return gathered


def _translated_task(row: dict, suite: str) -> Task | None:
    info = SUITES[suite]
    language, family = info["language"], info["family"]
    name = str(row["name"])
    match = re.search(r"(?i)(HumanEval|mbpp)_(\d+)(?:_|$)", name)
    if not match or match.group(1).lower() != family:
        raise ValueError(f"Unexpected task name for {info['source_config']}: {name!r}")
    number = int(match.group(2))
    if family == "mbpp" and number not in range(11, 511):
        return None
    prompt, tests = row["prompt"], row["tests"]
    candidate = re.search(r"\b(?:let|auto)\s+candidate\s*=\s*([A-Za-z_]\w*)\s*;", tests)
    if not candidate:
        raise ValueError(f"Cannot identify the tested function in {name}")
    required = candidate.group(1)
    escaped = re.escape(required)
    patterns = {
        "rs": rf"(?m)^[ \t]*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+{escaped}\s*\(",
        "js": rf"(?m)^[ \t]*(?:async\s+)?function\s+{escaped}\s*\(",
        "cpp": rf"(?m)^(?![ \t]*(?://|#))[^\n;{{}}]*\b{escaped}\s*\(",
    }
    declarations = list(re.finditer(patterns[language], prompt))
    if not declarations:
        raise ValueError(f"Cannot locate the target function declaration in {name}")
    context = prompt[:declarations[-1].start()].rstrip()
    # Rust/C++ translators put the completion's closing brace in their tests.
    # JavaScript tests already follow a full function body.
    if language in ("rs", "cpp"):
        main_pattern = r"fn\s+main\s*\(" if language == "rs" else r"int\s+main\s*\("
        tests = re.sub(r"^\s*\}\s*(?=" + main_pattern + ")", "\n", tests, count=1)
        if re.match(r"^\s*\}", tests):
            raise ValueError(f"Unrecognized completion harness in {name}")
    elif re.match(r"^\s*\}", tests):
        raise ValueError(f"Unexpected leading closing brace in JavaScript harness {name}")
    prompt = ("Implement the complete function declared below, including its closing brace. "
              "The context before its declaration is supplied by the runner; do not repeat that context.\n\n" + prompt)
    source_id = f"{'HumanEval' if family == 'humaneval' else 'MBPP'}/{number}"
    return Task(suite, language, name, number, prompt, tests, required, family=family,
                source_config=info["source_config"], source_url=info["source_url"],
                source_id=source_id, context_source=context)


def load_multiple(cache_dir: Path, suite: str, *, allow_download: bool = False) -> list[Task]:
    if suite not in SUITES or not suite.startswith("multiple_"):
        raise ValueError(f"Not a supported MultiPL-E suite: {suite}")
    rows = _multiple_rows(cache_dir, suite, SUITES[suite]["source_config"], allow_download=allow_download)
    tasks = [task for row in rows if (task := _translated_task(row, suite)) is not None]
    if not tasks:
        raise RuntimeError(f"No tasks remain in the selected {suite} test scope")
    return sorted(tasks, key=lambda t: t.number)


def load_rust(cache_dir: Path) -> list[Task]:
    """Compatibility alias for the original HumanEval-Rust loader."""
    return load_multiple(cache_dir, "multiple_rs")


def load_tasks(suites: list[str], cache_dir: Path, limit: int, start: int = 0, *,
               data_root: Path = Path("/data/hybrid-dard"), allow_download: bool = False) -> list[Task]:
    if limit < 0 or start < 0:
        raise ValueError("limit and start must be nonnegative")
    tasks = []
    for suite in suites:
        if suite not in SUITES:
            raise ValueError(f"Unknown suite {suite}; available: {', '.join(SUITES)}")
        if suite == "humaneval":
            full = load_humaneval(cache_dir, allow_download=allow_download)
        elif suite == "mbpp":
            full = load_mbpp(cache_dir, allow_download=allow_download)
        elif suite in ("humaneval_plus", "mbpp_plus", "bigcodebench_instruct"):
            from function_benchmarks import load_official_tasks
            full = load_official_tasks(suite, data_root)
        else:
            full = load_multiple(cache_dir, suite, allow_download=allow_download)
        selected = full[start: (start + limit) if limit > 0 else None]
        tasks.extend(selected)
        print(f"[data] {suite}: {len(selected)}/{len(full)} selected", flush=True)
    return tasks
