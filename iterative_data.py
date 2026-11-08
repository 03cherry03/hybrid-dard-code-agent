"""Frozen source-disjoint code-generation data and train-only reference labels.

This module reads cached datasets and existing receipts only. It does not fetch
datasets, run model inference, execute reference code, or weaken any evaluator.
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
from dataclasses import asdict
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable, Sequence

from datasets_code import (CODE_GENERATION_SUITES, LANGUAGE_NAMES, MBPP, SUITES,
                           Task, _mbpp_api, _multiple_rows, _translated_task, load_humaneval,
                           public_feedback_tests)
from function_benchmarks import _official_record, load_official_tasks

LANGUAGE_TAGS = {"py": "python", "rs": "rust", "js": "javascript", "cpp": "cpp"}
SPLIT_VERSION = "source_family_language_coverage_v1"
PROMPT_VERSION = "iterative_complete_function_fenced_v1"
MAX_HISTORY_FILES = 4096
MAX_HISTORY_BYTES = 2 * 1024**3


def digest(value) -> str:
    payload = value if isinstance(value, bytes) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def normalize_source_id(value: str, family: str | None = None) -> str:
    """Unify original/Plus identities without renumbering their source tasks."""
    if not isinstance(value, str):
        raise ValueError("A canonical source identity must be a string")
    match = re.fullmatch(r"(?i)(HumanEval|MBPP)/(\d+)", value)
    if not match:
        raise ValueError(f"Invalid source identity: {value!r}")
    observed_family = match[1].lower()
    if family and family != observed_family:
        raise ValueError(f"Source identity/family mismatch: {value!r}/{family}")
    return f"{'HumanEval' if observed_family == 'humaneval' else 'MBPP'}/{int(match[2])}"


def source_from_identity(row: dict) -> str | None:
    """Read public identity fields only; payload/code never establishes identity."""
    task_id = row.get("task_id")
    if task_id is None and isinstance(row.get("task_key"), (list, tuple)):
        task_id = row["task_key"][-1]
    derived = None
    if isinstance(task_id, str):
        match = re.fullmatch(r"(?i)(HumanEval|MBPP)/(\d+)", task_id)
        translated = re.fullmatch(r"(?i)(HumanEval|MBPP)_(\d+)(?:_.*)?", task_id)
        match = match or translated
        if match:
            derived = normalize_source_id(f"{match[1]}/{match[2]}")
    declared = row.get("source_id")
    if declared is not None:
        declared = normalize_source_id(declared)
        if derived is not None and declared != derived:
            raise ValueError("Declared source_id disagrees with task_id")
    return declared or derived


def build_iteration_messages(task: Task | dict) -> list[dict[str, str]]:
    """One shared prompt for training and every iterative evaluation backend."""
    language = task.language if isinstance(task, Task) else task["language"]
    prompt = task.prompt if isinstance(task, Task) else task["prompt"]
    if language not in LANGUAGE_NAMES or not isinstance(prompt, str):
        raise ValueError("Unsupported language or task prompt")
    instruction = (
        f"Implement the requested {LANGUAGE_NAMES[language]} function exactly as specified. "
        "Return the complete source for the requested function(s), including necessary imports "
        "and helper definitions. Preserve the required names and signatures. "
        "Do not include tests, a main entry point, or explanations. "
        f"Return the source in one Markdown code block tagged {LANGUAGE_TAGS[language]}.\n\n"
        + prompt)
    return [{"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": instruction}]


# Both spellings are public for the preparation and evaluation entry points.
build_iterative_messages = build_iteration_messages


def fenced_completion(language: str, code: str) -> str:
    if language not in LANGUAGE_TAGS or not isinstance(code, str) or not code.strip():
        raise ValueError("A nonempty source implementation is required")
    if "```" in code:
        raise ValueError("Reference source contains an ambiguous Markdown fence")
    return f"\n```{LANGUAGE_TAGS[language]}\n{code.strip()}\n```"


def _read_jsonl(path: Path) -> list[dict]:
    payload = path.read_bytes()
    if path.suffix == ".gz":
        payload = gzip.decompress(payload)
    rows = []
    for number, line in enumerate(payload.decode().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Expected an object in {path}:{number}")
        rows.append(row)
    return rows


def _all_translated_task(row: dict, suite: str) -> Task:
    """Extend the existing exact adapter to MBPP IDs outside its test-only filter."""
    old = _translated_task(row, suite)
    if old is not None:
        return old
    language = SUITES[suite]["language"]
    name = str(row["name"])
    match = re.fullmatch(r"(?i)mbpp_(\d+)(?:_.*)?", name)
    if not match:
        raise ValueError("Unexpected additional MultiPL-E source identity")
    number = int(match[1])
    prompt, tests = row["prompt"], row["tests"]
    candidate = re.search(r"\b(?:let|auto)\s+candidate\s*=\s*([A-Za-z_]\w*)\s*;", tests)
    if not candidate:
        raise ValueError(f"Cannot identify the tested function in {name}")
    required = candidate[1]
    escaped = re.escape(required)
    patterns = {
        "rs": rf"(?m)^[ \t]*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+{escaped}\s*\(",
        "js": rf"(?m)^[ \t]*(?:async\s+)?function\s+{escaped}\s*\(",
        "cpp": rf"(?m)^(?![ \t]*(?://|#))[^\n;{{}}]*\b{escaped}\s*\(",
    }
    declarations = list(re.finditer(patterns[language], prompt))
    if not declarations:
        raise ValueError(f"Cannot locate translated target declaration: {name}")
    context = prompt[:declarations[-1].start()].rstrip()
    if language in ("rs", "cpp"):
        main_pattern = r"fn\s+main\s*\(" if language == "rs" else r"int\s+main\s*\("
        tests = re.sub(r"^\s*\}\s*(?=" + main_pattern + ")", "\n", tests, count=1)
    if re.match(r"^\s*\}", tests):
        raise ValueError(f"Unrecognized translated completion harness: {name}")
    prompt = ("Implement the complete function declared below, including its closing brace. "
              "The context before its declaration is supplied by the runner; do not repeat that context.\n\n" + prompt)
    return Task(suite, language, name, number, prompt, tests, required,
                family="mbpp", source_config=SUITES[suite]["source_config"],
                source_url=SUITES[suite]["source_url"], source_id=f"MBPP/{number}", context_source=context)


def load_cached_inventory(data_root: Path) -> tuple[list[Task], dict]:
    """Use every cached MBPP original, and all prepared code-generation suites."""
    data_root = Path(data_root)
    cache = data_root / "datasets/function-cache"
    tasks = load_humaneval(cache, allow_download=False)
    raw_mbpp = _read_jsonl(cache / "mbpp.jsonl")
    seen = set()
    for row in raw_mbpp:
        number = int(row["task_id"])
        if number in seen:
            raise ValueError("Duplicate original MBPP source identity")
        seen.add(number)
        tests = list(row["test_list"])
        api, required = _mbpp_api(row["code"], tests)
        prompt = row["text"] + "\n\nRequired Python API (replace pass with your implementation):\n" + api
        test_source = "\n".join([(row.get("test_setup_code") or "")] + tests) + "\n"
        tasks.append(Task("mbpp", "py", f"MBPP/{number}", number, prompt, test_source,
                          required, family="mbpp", source_config="all_original_cached",
                          source_url=MBPP, source_id=f"MBPP/{number}",
                          feedback_test_source=public_feedback_tests(row["text"], required)))
    for suite in CODE_GENERATION_SUITES:
        if suite.startswith("multiple_"):
            rows = _multiple_rows(cache, suite, SUITES[suite]["source_config"], allow_download=False)
            tasks.extend(_all_translated_task(row, suite) for row in rows)
        elif suite.endswith("_plus"):
            tasks.extend(load_official_tasks(suite, data_root))
    inputs = {cache/"mbpp.jsonl", cache/"HumanEval.jsonl.gz", data_root/"datasets/functions/manifest.json"}
    inputs.update(cache.glob("multiple_*.json"))
    inputs.update(Path(t.evaluator_metadata["raw_path"]) for t in tasks if t.dataset.endswith("_plus"))
    inventory = {"raw_task_count": len(tasks), "raw_by_dataset": dict(Counter(t.dataset for t in tasks)),
                 "native_mbpp_scope": "All cached original rows, including IDs outside the historical 11..510 test subset.",
                 "multiple_scope": "Every advertised cached MultiPL-E row, including MBPP IDs outside 11..510.",
                 "input_sha256": {str(path.resolve()): digest(path.read_bytes()) for path in sorted(inputs)},
                 "network_access": False, "reference_execution": False}
    return tasks, inventory


def deduplicate_tasks(tasks: Iterable[Task]) -> tuple[list[Task], list[dict]]:
    """Keep one unchanged Task per source/language, with Plus as the strict grader."""
    groups = defaultdict(list)
    for task in tasks:
        canonical = normalize_source_id(task.source_id, task.family)
        if task.source_id != canonical or task.language not in LANGUAGE_TAGS:
            raise ValueError("Task identities must already use the canonical source/language")
        if task.dataset not in CODE_GENERATION_SUITES:
            raise ValueError("Non-code-generation suite in iterative inventory")
        groups[(canonical, task.language)].append(task)
    selected, excluded = [], []
    for key in sorted(groups):
        choices = sorted(groups[key], key=lambda t: (not t.dataset.endswith("_plus"), t.dataset, t.task_id))
        if len({(t.dataset, t.task_id) for t in choices}) != len(choices):
            raise ValueError(f"Duplicate records within a dataset: {key}")
        if len(choices) > 1 and (key[1] != "py" or not choices[0].dataset.endswith("_plus")):
            raise ValueError(f"Ambiguous duplicate source/language: {key}")
        selected.append(choices[0])
        for other in choices[1:]:
            excluded.append({"source_id": key[0], "language": key[1],
                             "kept_dataset": choices[0].dataset, "excluded_dataset": other.dataset,
                             "excluded_task_id": other.task_id, "reason": "Prefer unchanged official Plus evaluator"})
    return selected, excluded


def discover_exposure(results_root: Path, *, exclude: Path | None = None) -> tuple[set[str], dict]:
    """Conservatively recognize attempted/evaluated identities in local receipts."""
    root = Path(results_root)
    names = {"case_results.jsonl", "runs.jsonl", "results.jsonl", "errors.jsonl"}
    paths = sorted(p for p in root.rglob("*.jsonl") if p.name in names
                   and not (exclude and p.is_relative_to(Path(exclude)))) if root.exists() else []
    if len(paths) > MAX_HISTORY_FILES or sum(p.stat().st_size for p in paths) > MAX_HISTORY_BYTES:
        raise ValueError("Historical exposure scan exceeds the declared bound")
    exposed, files = set(), []
    for path in paths:
        before = len(exposed)
        count = 0
        for row in _read_jsonl(path):
            source = source_from_identity(row)
            if source:
                exposed.add(source)
                count += 1
        files.append({"path": str(path.resolve()), "sha256": digest(path.read_bytes()),
                      "identity_rows": count, "new_source_count": len(exposed)-before})
    return exposed, {"scope": "Recognized local historical generation/evaluation JSONL receipts only; not pretraining or external exposure.",
                     "scanned_files": files, "exposed_source_ids": sorted(exposed),
                     "global_unseen_claim": False}


def _allocation(groups: dict, target: int) -> dict:
    total = sum(len(values) for values in groups.values())
    if not total:
        return {key: 0 for key in groups}
    desired = {key: target * len(values) / total for key, values in groups.items()}
    counts = {key: math.floor(value) for key, value in desired.items()}
    for key in sorted(groups, key=lambda k: (-(desired[k]-counts[k]), k))[:target-sum(counts.values())]:
        counts[key] += 1
    return counts


def partition_sources(tasks: Sequence[Task], *, seed: int = 20261105,
                      exposed_sources: Iterable[str] = ()) -> dict:
    """Family 80/10/10, retaining language-availability strata and source groups."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    exposed = {normalize_source_id(source) for source in exposed_sources}
    source_tasks = defaultdict(list)
    for task in tasks:
        source_tasks[normalize_source_id(task.source_id, task.family)].append(task)
    assigned = {}
    strata_report = []
    for family in sorted({task.family for task in tasks}):
        sources = {source: values for source, values in source_tasks.items() if values[0].family == family}
        groups = defaultdict(list)
        for source, values in sources.items():
            groups[tuple(sorted({task.language for task in values}))].append(source)
        # Exact family totals are floor(10%) each; all remaining sources train.
        hold_count = val_count = len(sources)//10
        hold_quotas = _allocation(groups, hold_count)
        rank = lambda source: (digest({"seed": seed, "source_id": source}), source)
        remains = {}
        for coverage, members in sorted(groups.items()):
            ordered = sorted(members, key=lambda source: (source in exposed, *rank(source)))
            held = ordered[:hold_quotas[coverage]]
            assigned.update({source: "holdout" for source in held})
            remains[coverage] = [source for source in members if source not in set(held)]
        val_quotas = _allocation(remains, val_count)
        for coverage, members in sorted(remains.items()):
            ordered = sorted(members, key=rank)
            val = ordered[:val_quotas[coverage]]
            assigned.update({source: "validation" for source in val})
            assigned.update({source: "train" for source in ordered[val_quotas[coverage]:]})
            original = groups[coverage]
            strata_report.append({"family": family, "languages": list(coverage),
                                  "sources": len(original),
                                  "split_sources": dict(Counter(assigned[source] for source in original)),
                                  "holdout_previously_exposed": sum(assigned[source] == "holdout" and source in exposed for source in original)})
    split_sources = {name: sorted(source for source, split in assigned.items() if split == name)
                     for name in ("train", "validation", "holdout")}
    if len(assigned) != len(source_tasks) or sum(map(len, split_sources.values())) != len(source_tasks):
        raise AssertionError("Source partition does not account for every identity exactly once")
    return {"schema_version": 1, "algorithm": SPLIT_VERSION, "seed": seed,
            "ratios": {"train": .8, "validation": .1, "holdout": .1},
            "rounding": "floor(10% of each family) for validation and holdout; remaining sources train",
            "holdout_priority": "Prefer locally unexposed sources within each family/language-availability stratum; retain translated-language coverage",
            "source_assignment": dict(sorted(assigned.items())), "source_ids": split_sources,
            "strata": strata_report,
            "previously_exposed_by_split": {name: sorted(set(values) & exposed) for name, values in split_sources.items()},
            "globally_unseen_holdout_claim": False}


def canonical_reference(task: Task, data_root: Path) -> tuple[str, dict]:
    """Call only for training tasks; no tests/contracts are returned as labels."""
    if task.language != "py":
        raise ValueError("Only Python has cached canonical source labels")
    if task.dataset.endswith("_plus"):
        row = _official_record(task)
        code = row["canonical_solution"]
        source_path = Path(task.evaluator_metadata["raw_path"])
    else:
        source_path = Path(data_root) / "datasets/function-cache" / ("mbpp.jsonl" if task.family == "mbpp" else "HumanEval.jsonl.gz")
        stat = source_path.stat()
        matches = [r for r in _canonical_rows(str(source_path), stat.st_size, stat.st_mtime_ns)
                   if normalize_source_id(f"MBPP/{r['task_id']}" if task.family == "mbpp" else r["task_id"]) == task.source_id]
        if len(matches) != 1:
            raise ValueError("Canonical reference source identity is missing/ambiguous")
        row = matches[0]
        code = row["code"] if task.family == "mbpp" else row["canonical_solution"]
    if task.family == "humaneval":
        # HumanEval canonical labels are bodies. Keep the complete target and
        # any canonical additions, omitting the public context supplied by Task.
        full = row["prompt"] + code
        tree = ast.parse(full)
        target = next((node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and node.name == task.required_name), None)
        if target is None:
            raise ValueError("Canonical HumanEval target is missing")
        first = min([target.lineno] + [d.lineno for d in target.decorator_list])
        code = "".join(full.splitlines(keepends=True)[first-1:])
    tree = ast.parse(code)
    names = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    names.update(target.id for node in tree.body if isinstance(node, ast.Assign)
                 and isinstance(node.value, ast.Lambda) for target in node.targets if isinstance(target, ast.Name))
    if task.required_name and task.required_name not in names:
        raise ValueError("Canonical label does not define the required public API")
    return code, {"kind": "canonical_reference", "dataset": task.dataset,
                  "raw_path": str(source_path.resolve()), "record_sha256": digest(row),
                  "reference_field": "code" if task.dataset == "mbpp" else "canonical_solution",
                  "reference_execution": False}


@lru_cache(maxsize=4)
def _canonical_rows(path: str, size: int, mtime_ns: int) -> list[dict]:
    return _read_jsonl(Path(path))


def _example(task: Task, code: str, provenance: dict) -> dict:
    return {"example_id": f"{task.source_id}/{task.language}", "source_id": task.source_id,
            "language": task.language, "messages": build_iteration_messages(task),
            "completion": fenced_completion(task.language, code),
            "provenance": {**provenance, "task_sha256": digest(asdict(task)),
                           "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                           "prompt_version": PROMPT_VERSION}}


def _case_key(row: dict) -> tuple:
    return tuple(row.get(key) for key in ("dataset", "language", "task_id", "repeat", "seed"))


def validate_teacher_pair(task: Task, case: dict, run: dict) -> str:
    """Bind a passing AR verdict, original Task, and exact graded implementation."""
    if source_from_identity(case) != task.source_id or source_from_identity(run) != task.source_id:
        raise ValueError("Teacher source identity mismatch")
    if (case.get("decoder") != "ar" or run.get("decoder") != "ar"
            or case.get("final_pass") is not True or case.get("final_status") != "passed"
            or case.get("case_completed") is not True or case.get("failure_kind")
            or case.get("model_identity_verified") is not True
            or case.get("measurement_complete") is not True):
        raise ValueError("Teacher is not a verified completed passing AR case")
    if case.get("task_sha256") != digest(asdict(task)):
        raise ValueError("Teacher task hash differs from the frozen Task")
    if (_case_key(case) != _case_key(run) or case.get("language") != task.language
            or case.get("dataset") != task.dataset or case.get("task_id") != task.task_id
            or run.get("attempt") != case.get("attempts")):
        raise ValueError("Teacher case/run identity or final attempt mismatch")
    grade = case.get("grading", {}).get("final", {})
    if grade.get("passed") is not True or grade.get("sealed") is not True:
        raise ValueError("Teacher lacks a sealed passing final verdict")
    generation = run.get("generation", {})
    if not case.get("messages_sha256") or generation.get("messages_sha256") != case["messages_sha256"]:
        raise ValueError("Teacher generation/message binding mismatch")
    code = run.get("code")
    if (not isinstance(code, str) or not code.strip()
            or run.get("code_sha256") != hashlib.sha256(code.encode()).hexdigest()):
        raise ValueError("Teacher implementation hash mismatch")
    return code


def collect_teacher_examples(train_tasks: Sequence[Task], teacher_roots: Sequence[Path]) -> tuple[dict, dict]:
    """Never access a code payload for a validation/holdout source identity."""
    task_lookup = {(task.source_id, task.language): task for task in train_tasks if task.language != "py"}
    train_sources = {task.source_id for task in train_tasks}
    accepted, rejected = {}, Counter()
    inspected = []
    case_paths = sorted({p.resolve() for root in teacher_roots for p in Path(root).rglob("case_results.jsonl")})
    if len(case_paths) > MAX_HISTORY_FILES:
        raise ValueError("Too many teacher result files")
    for case_path in case_paths:
        wanted = {}
        for case in _read_jsonl(case_path):
            source = source_from_identity(case)
            if source not in train_sources or (source, case.get("language")) not in task_lookup:
                continue
            if case.get("decoder") != "ar" or case.get("final_pass") is not True:
                continue
            wanted[_case_key(case)] = case
        if not wanted:
            continue
        run_path = case_path.with_name("runs.jsonl")
        if not run_path.is_file():
            rejected["missing_runs_file"] += len(wanted)
            continue
        inspected.append({"case_path": str(case_path), "case_sha256": digest(case_path.read_bytes()),
                          "runs_path": str(run_path), "runs_sha256": digest(run_path.read_bytes())})
        for run in _read_jsonl(run_path):
            source = source_from_identity(run)
            if source not in train_sources:  # filter before any code/generation field access
                continue
            case = wanted.get(_case_key(run))
            if case is None or run.get("decoder") != "ar" or run.get("attempt") != case.get("attempts"):
                continue
            task = task_lookup[(source, case["language"])]
            try:
                code = validate_teacher_pair(task, case, run)
                item = _example(task, code, {"kind": "verified_ar_teacher", "synthetic": True,
                    "model": case.get("model"), "case_path": str(case_path), "run_path": str(run_path),
                    "case_record_sha256": digest(case), "run_record_sha256": digest(run),
                    "verified_on": "Original unchanged Task sealed final evaluator",
                    "teacher_prompt": "Historical prompt; generated source is reformatted using the shared iterative prompt"})
            except ValueError as exc:
                rejected[str(exc)] += 1
                continue
            label_key = (source, task.language)
            if label_key in accepted:
                rejected["duplicate_verified_label"] += 1
            else:
                accepted[label_key] = item
    return accepted, {"inspected_files": inspected, "rejected": dict(rejected),
                       "accepted_labels": len(accepted), "nontraining_payloads_used": 0}


def validate_training_examples(examples: Sequence[dict], train_tasks: Sequence[Task]) -> None:
    lookup = {(task.source_id, task.language): task for task in train_tasks}
    seen = set()
    for example in examples:
        key = (normalize_source_id(example["source_id"]), example["language"])
        if key not in lookup or key in seen:
            raise ValueError("Training label outside train partition or duplicate source/language")
        seen.add(key)
        if example["messages"] != build_iteration_messages(lookup[key]):
            raise ValueError("Training message differs from the shared prompt")
        if example.get("provenance", {}).get("task_sha256") != digest(asdict(lookup[key])):
            raise ValueError("Training label is not bound to the frozen Task")
        if not example.get("completion", "").startswith(f"\n```{LANGUAGE_TAGS[key[1]]}\n"):
            raise ValueError("Training completion lacks the declared native fenced format")
        if set(example) - {"example_id", "source_id", "language", "messages", "completion", "provenance"}:
            raise ValueError("Unexpected training field; test/oracle payloads are forbidden")


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def prepare_iterative_data(data_root: Path, output: Path, seed: int = 20261105, *,
                           results_root: Path | None = None,
                           teacher_roots: Sequence[Path] | None = None) -> dict:
    data_root, output = Path(data_root).resolve(), Path(output).resolve()
    tasks, inventory = load_cached_inventory(data_root)
    tasks, excluded = deduplicate_tasks(tasks)
    task_digest = digest([asdict(task) for task in tasks])
    frozen = output / "split_manifest.json"
    if frozen.exists():
        manifest = json.loads(frozen.read_text())
        if (manifest.get("seed") != seed or manifest.get("task_inventory_sha256") != task_digest
                or manifest.get("input_sha256") != inventory["input_sha256"]):
            raise ValueError("Existing frozen split differs from requested seed or current datasets")
        for name, expected in manifest["artifact_sha256"].items():
            if digest((output/name).read_bytes()) != expected:
                raise ValueError(f"Frozen iterative artifact changed: {name}")
        return json.loads((output/"data_accounting.json").read_text())
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to mix a new split into an existing nonempty output")
    exposed, exposure = discover_exposure(results_root or data_root/"results", exclude=output)
    manifest = partition_sources(tasks, seed=seed, exposed_sources=exposed)
    split_tasks = {name: [task for task in tasks if manifest["source_assignment"][task.source_id] == name]
                   for name in ("train", "validation", "holdout")}
    source_sets = {name: {task.source_id for task in values} for name, values in split_tasks.items()}
    if any(source_sets[a] & source_sets[b] for a, b in (("train", "validation"), ("train", "holdout"), ("validation", "holdout"))):
        raise AssertionError("Cross-language source leakage between splits")
    if teacher_roots is None:
        teacher_roots = [data_root/"results/code-generation-dard-core-20261105",
                         data_root/"results/code-generation-optimized-20261107"]
    teacher, teacher_report = collect_teacher_examples(split_tasks["train"], teacher_roots)
    examples, missing, reference_rejected = [], [], []
    for task in split_tasks["train"]:
        item = teacher.get((task.source_id, task.language))
        if task.language == "py":
            try:
                code, provenance = canonical_reference(task, data_root)
                item = _example(task, code, provenance)
            except (ValueError, SyntaxError) as exc:
                reference_rejected.append({"source_id": task.source_id, "language": task.language,
                                           "reason": str(exc)})
        if item is None:
            missing.append({"source_id": task.source_id, "language": task.language,
                            "dataset": task.dataset, "task_id": task.task_id})
        else:
            examples.append(item)
    validate_training_examples(examples, split_tasks["train"])
    if len(examples) + len(missing) != len(split_tasks["train"]):
        raise AssertionError("Training label coverage does not reconcile")
    accounting = {"schema_version": 1, "seed": seed, "split_algorithm": SPLIT_VERSION,
        "prompt_version": PROMPT_VERSION, "inventory": inventory,
        "deduplicated_tasks": len(tasks), "canonical_sources": len(manifest["source_assignment"]),
        "excluded_duplicate_count": len(excluded), "excluded_duplicates": excluded,
        "splits": {name: {"sources": len(source_sets[name]), "tasks": len(values),
             "by_language": dict(Counter(task.language for task in values)),
             "by_family": dict(Counter(task.family for task in values)),
             "previously_exposed_sources": len(source_sets[name] & exposed),
             "locally_unexposed_sources": len(source_sets[name] - exposed)} for name, values in split_tasks.items()},
        "training_examples": len(examples), "training_examples_by_language": dict(Counter(x["language"] for x in examples)),
        "training_examples_by_provenance": dict(Counter(x["provenance"]["kind"] for x in examples)),
        "missing_training_labels": missing, "rejected_canonical_labels": reference_rejected,
        "teacher": teacher_report, "exposure": exposure,
        "proof": {"raw_tasks_equal_kept_plus_excluded": inventory["raw_task_count"] == len(tasks)+len(excluded),
                  "all_kept_tasks_assigned_once": sum(map(len, split_tasks.values())) == len(tasks),
                  "source_splits_disjoint": True, "training_labels_train_only": True,
                  "train_labels_plus_missing_equal_train_tasks": True,
                  "validation_or_holdout_payloads_in_training": False,
                  "original_task_records_and_evaluator_tests_preserved": True},
        "limitations": ["Local history coverage is bounded to known evaluation receipt types; globally unseen data is not claimed.",
                        "Previously exposed holdout source IDs remain labeled and cannot support a clean held-out generalization claim.",
                        "Teacher labels are synthetic passing implementations, not official references.",
                        "Some training language variants lack labels; generate additional verified teachers on TRAIN identities only."]}
    output.mkdir(parents=True, exist_ok=True)
    names = []
    for name, values in split_tasks.items():
        filename = f"{name}_tasks.jsonl"
        _write_jsonl(output/filename, (asdict(task) for task in values)); names.append(filename)
    _write_jsonl(output/"training_examples.jsonl", examples); names.append("training_examples.jsonl")
    _write_json(output/"data_accounting.json", accounting); names.append("data_accounting.json")
    manifest.update(task_inventory_sha256=task_digest,
                    input_sha256=inventory["input_sha256"],
                    artifact_sha256={name: digest((output/name).read_bytes()) for name in names},
                    data_root=str(data_root), prompt_version=PROMPT_VERSION)
    _write_json(frozen, manifest)
    return accounting
