"""Official HumanEval, MultiPL-E Rust and MBPP test-set loaders. No external deps."""
from __future__ import annotations

import gzip
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HUMANEVAL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
MULTIPLE_HF = "https://datasets-server.huggingface.co/rows"
MBPP = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"


@dataclass(frozen=True)
class Task:
    dataset: str
    language: str
    task_id: str
    number: int
    prompt: str
    test_source: str
    required_name: str | None = None


def fetch_cached(url: str, path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path.read_bytes()
    req = urllib.request.Request(url, headers={"User-Agent": "local-dard-bench/1.0"})
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


def load_humaneval(cache_dir: Path) -> list[Task]:
    contents = fetch_cached(HUMANEVAL, cache_dir / "HumanEval.jsonl.gz")
    rows = _entries(gzip.decompress(contents))
    tasks = []
    for row in rows:
        number = int(row["task_id"].split("/")[-1])
        tests = row["test"] + "\ncheck(" + row["entry_point"] + ")\n"
        tasks.append(Task("humaneval", "py", row["task_id"], number,
                          row["prompt"], tests, row["entry_point"]))
    if len(tasks) != 164:
        raise RuntimeError(f"Unexpected HumanEval size: {len(tasks)} (expected 164)")
    return sorted(tasks, key=lambda t: t.number)


def load_rust(cache_dir: Path) -> list[Task]:
    gathered = []
    for offset in range(0, 200, 100):
        params = urllib.parse.urlencode(dict(dataset="nuprl/MultiPL-E", config="humaneval-rs",
                                             split="test", offset=offset, length=100))
        doc = json.loads(fetch_cached(MULTIPLE_HF + "?" + params,
                                      cache_dir / f"multiple_rs_{offset}.json"))
        rows = [entry["row"] for entry in doc.get("rows", [])]
        gathered.extend(rows)
        if len(rows) < 100:
            break
    tasks = []
    for row in gathered:
        name = str(row.get("name", ""))
        m = re.search(r"HumanEval_(\d+)_", name)
        if not m:
            continue
        # In the original MultiPL-E task representation, `tests` begins with
        # the closing `}` of the function the model is supposed to complete.
        # Here we deliberately request a *whole function*, so remove that
        # synthetic closing brace, while retaining the original tests.
        tests = re.sub(r"^\s*\}\s*(?=fn\s+main\s*\()", "\n", row["tests"], count=1)
        tasks.append(Task("multiple_rs", "rs", name, int(m.group(1)), row["prompt"], tests))
    if len(tasks) < 100:
        raise RuntimeError(f"Rust dataset incomplete: received {len(tasks)} tasks")
    return sorted(tasks, key=lambda t: t.number)


def load_mbpp(cache_dir: Path) -> list[Task]:
    rows = _entries(fetch_cached(MBPP, cache_dir / "mbpp.jsonl"))
    tasks = []
    # Official MBPP paper test split: IDs 11-510. Never use validation/train
    # IDs for the reported test score.
    for row in rows:
        number = int(row["task_id"])
        if number not in range(11, 511):
            continue
        tests = "\n".join([(row.get("test_setup_code") or "")] + list(row["test_list"])) + "\n"
        tasks.append(Task("mbpp", "py", f"MBPP/{number}", number,
                          row["text"], tests))
    if len(tasks) != 500:
        raise RuntimeError(f"Unexpected MBPP test split: {len(tasks)} (expected 500)")
    return sorted(tasks, key=lambda t: t.number)


def load_tasks(suites: list[str], cache_dir: Path, limit: int, start: int = 0) -> list[Task]:
    loaders = {"humaneval": load_humaneval, "multiple_rs": load_rust, "mbpp": load_mbpp}
    tasks = []
    for suite in suites:
        if suite not in loaders:
            raise ValueError(f"Unknown suite {suite}; available: {', '.join(loaders)}")
        full = loaders[suite](cache_dir)
        selected = full[start: (start + limit) if limit > 0 else None]
        tasks.extend(selected)
        print(f"[data] {suite}: {len(selected)}/{len(full)} selected", flush=True)
    return tasks
