#!/usr/bin/env python3
"""Download public benchmark DATA only, or verify an existing immutable snapshot.

No package installs, model downloads or target repository checkout.
HF viewer pages must expose the exact selected x-revision, contiguous indices,
complete cells and a stable total. Every downloaded file is hashed in catalog.json.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import re
from pathlib import Path
import time
import urllib.parse
import urllib.request
import urllib.error

from storage import data_root, initialize

SWE_VERSION = "5.0.1"
SWE_COMMIT = "87ab1f6ced28f75ba73ca899dc759b019310944a"
HF_REVISIONS = {
    "SWE-bench/SWE-bench_Verified": "78f471bf655a3137b2e8a75af1501690ec009ec3",
    "SWE-bench/SWE-bench_Multilingual": "846e647b9f33c0b51b739d005d13d85493c9af09",
}
PLUS = {
    "humaneval_plus": ("HumanEvalPlus", "v0.1.10", 164),
    "mbpp_plus": ("MbppPlus", "v0.2.0", 378),
}
LEGACY_SUITES = ("humaneval", "mbpp", "multiple_rs", "multiple_js", "multiple_cpp",
                 "multiple_mbpp_rs", "multiple_mbpp_js", "multiple_mbpp_cpp")
FUNCTION_SUITES = (*LEGACY_SUITES, *PLUS)
REPOSITORY_SUITES = ("swebench_verified", "swebench_multilingual")
SCOPES = ("functions", "repositories", "all")


# The official viewer currently serves mbpp-js from an older processing revision.
# Pin and disclose that actual revision instead of labelling it with HEAD's SHA.
MULTIPLE_VIEWER_REVISIONS = {"mbpp-js": "a2f35c3abf9ca0142057303c26fdbee963e9a503"}
# Official multilingual appendix; JS/TS uses the repository's primary source language.
# https://www.swebench.com/multilingual.html#appendix-a-dataset
REPO_LANGUAGES = {
    **dict.fromkeys("redis/redis jqlang/jq micropython/micropython valkey-io/valkey".split(), "c"),
    **dict.fromkeys("nlohmann/json fmtlib/fmt".split(), "cpp"),
    **dict.fromkeys("caddyserver/caddy hashicorp/terraform prometheus/prometheus gohugoio/hugo gin-gonic/gin".split(), "go"),
    **dict.fromkeys("google/gson apache/druid projectlombok/lombok apache/lucene reactivex/rxjava javaparser/javaparser".split(), "java"),
    **dict.fromkeys("babel/babel vuejs/core facebook/docusaurus immutable-js/immutable-js".split(), "ts"),
    **dict.fromkeys("mrdoob/three.js preactjs/preact axios/axios".split(), "js"),
    **dict.fromkeys("phpoffice/phpspreadsheet laravel/framework php-cs-fixer/php-cs-fixer briannesbitt/carbon".split(), "php"),
    **dict.fromkeys("jekyll/jekyll fluent/fluentd fastlane/fastlane jordansissel/fpm faker-ruby/faker rubocop/rubocop".split(), "rb"),
    **dict.fromkeys("tokio-rs/tokio uutils/coreutils nushell/nushell tokio-rs/axum burntsushi/ripgrep sharkdp/bat astral-sh/ruff".split(), "rs"),
}


def sha256(contents):
    return hashlib.sha256(contents).hexdigest()


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_bytes(content)
    temp.replace(path)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def jsonl(rows):
    return ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode()


def request(url):
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "hybrid-dard-benchmark-preparation/1"})
            with urllib.request.urlopen(req, timeout=60) as response:
                contents = response.read()
                if not contents:
                    raise ValueError(f"Empty response: {url}")
                return contents, dict((k.lower(), v) for k, v in response.headers.items())
        except (OSError, ValueError) as exc:
            if attempt == 7:
                raise
            delay = min(60, 5 * (2 ** attempt))
            if isinstance(exc, urllib.error.HTTPError):
                if exc.code not in (429, 500, 502, 503, 504):
                    raise
                value = exc.headers.get("Retry-After", "")
                if value.isdigit():
                    delay = min(60, max(delay, int(value)))
            print(f"Retrying download in {delay}s: {url} ({exc})", flush=True)
            time.sleep(delay)


class Snapshot:
    def __init__(self, root):
        self.root = initialize(root)
        self.catalog_path = self.root / "datasets/catalog.json"
        self.catalog = json.loads(self.catalog_path.read_text()) if self.catalog_path.exists() else {
            "schema_version": 1, "files": {}, "datasets": {}, "created_at": datetime.now(timezone.utc).isoformat()}

    def save(self):
        write(self.catalog_path, json_bytes(self.catalog))

    def store(self, path, contents, **provenance):
        relative = str(path.relative_to(self.root))
        prior = self.catalog["files"].get(relative)
        digest = sha256(contents)
        if prior and prior["sha256"] != digest:
            raise ValueError(f"Snapshot changed: {relative}; use a new data root")
        write(path, contents)
        self.catalog["files"][relative] = {"sha256": digest, "bytes": len(contents), **provenance}

    def fetch(self, url, path, revision=None):
        relative = str(path.relative_to(self.root))
        previous = self.catalog["files"].get(relative)
        if previous and path.is_file():
            contents = path.read_bytes()
            if sha256(contents) != previous["sha256"]:
                raise ValueError(f"Corrupt snapshot: {path}")
            if previous.get("url") != url:
                raise ValueError(f"Source changed for {path}")
            if revision and previous.get("headers", {}).get("x-revision") != revision:
                raise ValueError(f"Cached viewer revision mismatch: {path}")
            return contents, previous.get("headers", {})
        contents, headers = request(url)
        if revision and headers.get("x-revision") != revision:
            raise ValueError(f"HF viewer revision mismatch: {headers.get('x-revision')} != {revision}; {url}")
        self.store(path, contents, url=url, headers={k: v for k, v in headers.items() if k in ("x-revision", "etag")})
        self.save()
        return contents, headers


def validate_page(doc, offset, total=None):
    if doc.get("partial"):
        raise ValueError("Partial dataset page")
    count = doc["num_rows_total"]
    if total is not None and total != count:
        raise ValueError("Dataset row count changed mid-download")
    items = doc["rows"]
    if not items and offset < count:
        raise ValueError("Premature end of dataset")
    for index, item in enumerate(items, offset):
        if item.get("row_idx") != index or item.get("truncated_cells"):
            raise ValueError("Truncated or out-of-order dataset page")
    return [item["row"] for item in items], count


def hf_rows(snapshot, dataset, config, split, cache_prefix, revision, page_size=100):
    rows, total = [], None
    while total is None or len(rows) < total:
        offset = len(rows)
        query = urllib.parse.urlencode(dict(dataset=dataset, config=config, split=split, offset=offset, length=page_size))
        url = "https://datasets-server.huggingface.co/rows?" + query
        raw, _ = snapshot.fetch(url, Path(str(cache_prefix) + f"_{offset}.json"), revision=revision)
        page, total = validate_page(json.loads(raw), offset, total)
        rows.extend(page)
    if len(rows) != total:
        raise ValueError("Extra rows in snapshot")
    return rows


def check_rows(rows, key, expected=None):
    ids = [row[key] for row in rows]
    if len(set(ids)) != len(ids) or (expected is not None and len(rows) != expected):
        raise ValueError(f"Unexpected count/duplicate IDs: {len(rows)}; expected {expected}")


def prepare_plus(snapshot):
    suites = {}
    for suite, (name, revision, count) in PLUS.items():
        url = f"https://github.com/evalplus/{name.lower()}_release/releases/download/{revision}/{name}.jsonl.gz"
        compressed, _ = snapshot.fetch(url, snapshot.root / f"datasets/functions/raw/{suite}.jsonl.gz")
        raw = gzip.decompress(compressed)
        rows = [json.loads(line) for line in raw.splitlines() if line]
        check_rows(rows, "task_id", count)
        path = snapshot.root / f"datasets/functions/raw/{suite}.jsonl"
        snapshot.store(path, raw, source_url=url, revision=revision)
        suites[suite] = {"path": f"raw/{suite}.jsonl", "sha256": sha256(raw), "revision": revision,
                         "evaluator_version": "0.3.1"}
        snapshot.catalog["datasets"][suite] = {"count": count, "languages": {"py": count}, "revision": revision}
        snapshot.save()
        print(f"Prepared {suite}: {count}", flush=True)
    return suites


def public_repo_task(row, dataset):
    language = "py" if dataset == "swebench_verified" else REPO_LANGUAGES[row["repo"].lower()]
    return {"dataset": dataset, "instance_id": row["instance_id"], "repo": row["repo"],
            "base_commit": row["base_commit"], "problem_statement": row["problem_statement"],
            "language": language}


def reuse_repository_snapshot(snapshot, name, dataset, count):
    """Keep existing pinned data and unused legacy metadata byte-for-byte."""
    target = snapshot.root / f"datasets/{name}"
    paths = [target / filename for filename in ("sealed.jsonl", "tasks.jsonl", "manifest.json")]
    if not all(str(path.relative_to(snapshot.root)) in snapshot.catalog["files"] for path in paths):
        return False
    payloads = {}
    for path in paths:
        record = snapshot.catalog["files"][str(path.relative_to(snapshot.root))]
        contents = path.read_bytes()
        if len(contents) != record["bytes"] or sha256(contents) != record["sha256"]:
            raise ValueError(f"Corrupt repository snapshot: {path}")
        payloads[path.name] = contents
    sealed = [json.loads(line) for line in payloads["sealed.jsonl"].splitlines() if line]
    public = [json.loads(line) for line in payloads["tasks.jsonl"].splitlines() if line]
    check_rows(sealed, "instance_id", count)
    expected_public = [public_repo_task(row, name) for row in sealed]
    if len(public) != count or any(any(row.get(key) != value for key, value in expected.items())
                                   for row, expected in zip(public, expected_public)):
        raise ValueError(f"Repository public snapshot changed: {name}")
    counts = dict(Counter(row["language"] for row in expected_public))
    identity = {"schema_version": 1, "dataset": name, "source": dataset, "revision": HF_REVISIONS[dataset],
                "count": count, "languages": counts, "tasks_path": "tasks.jsonl",
                "tasks_sha256": sha256(payloads["tasks.jsonl"]), "sealed_path": "sealed.jsonl",
                "sealed_sha256": sha256(payloads["sealed.jsonl"]), "harness_version": SWE_VERSION,
                "harness_commit": SWE_COMMIT}
    manifest = json.loads(payloads["manifest.json"])
    if any(manifest.get(key) != value for key, value in identity.items()):
        raise ValueError(f"Repository manifest changed: {name}")
    expected_catalog = {"count": count, "languages": counts, "revision": HF_REVISIONS[dataset]}
    if snapshot.catalog["datasets"].get(name) != expected_catalog:
        raise ValueError(f"Repository dataset catalog changed: {name}")
    return True


def prepare_swe(snapshot, name, dataset, count):
    if reuse_repository_snapshot(snapshot, name, dataset, count):
        print(f"Reused {name}: {count}", flush=True)
        return
    revision = HF_REVISIONS[dataset]
    rows = hf_rows(snapshot, dataset, "default", "test", snapshot.root / f"datasets/downloads/{name}", revision, 20)
    check_rows(rows, "instance_id", count)
    public = [public_repo_task(row, name) for row in rows]
    target = snapshot.root / f"datasets/{name}"
    raw, clean = jsonl(rows), jsonl(public)
    snapshot.store(target / "sealed.jsonl", raw, dataset=dataset, revision=revision)
    snapshot.store(target / "tasks.jsonl", clean, dataset=dataset, revision=revision)
    counts = dict(Counter(row["language"] for row in public))
    manifest = {"schema_version": 1, "dataset": name, "source": dataset, "revision": revision,
                "count": count, "languages": counts, "tasks_path": "tasks.jsonl", "tasks_sha256": sha256(clean),
                "sealed_path": "sealed.jsonl", "sealed_sha256": sha256(raw),
                "harness_version": SWE_VERSION, "harness_commit": SWE_COMMIT,
                "language_mapping": "official multilingual repository table; JS/TS grouped by primary repository language",
                "local_preparation_status": "repository checkouts and native environments require explicit preparation"}
    snapshot.store(target / "manifest.json", json_bytes(manifest))
    snapshot.catalog["datasets"][name] = {"count": count, "languages": counts, "revision": revision}
    snapshot.save()
    print(f"Prepared {name}: {count} {counts}", flush=True)


def prepare_legacy(snapshot):
    cache = snapshot.root / "datasets/function-cache"
    for repo, branch, filename, destination in [
        ("openai/human-eval", "master", "data/HumanEval.jsonl.gz", "HumanEval.jsonl.gz"),
        ("google-research/google-research", "master", "mbpp/mbpp.jsonl", "mbpp.jsonl"),
    ]:
        meta, _ = snapshot.fetch(f"https://api.github.com/repos/{repo}/commits/{branch}",
                                 snapshot.root / f"datasets/downloads/{destination}.commit.json")
        revision = json.loads(meta)["sha"]
        snapshot.fetch(f"https://raw.githubusercontent.com/{repo}/{revision}/{filename}", cache / destination)
    meta, _ = snapshot.fetch("https://huggingface.co/api/datasets/nuprl/MultiPL-E", snapshot.root / "datasets/downloads/multiple.meta.json")
    revision = json.loads(meta)["sha"]
    suite_revisions = {}
    for family in ("humaneval", "mbpp"):
        for language in ("rs", "js", "cpp"):
            suite = f"multiple_{language}" if family == "humaneval" else f"multiple_mbpp_{language}"
            config = f"{family}-{language}"
            actual_revision = MULTIPLE_VIEWER_REVISIONS.get(config, revision)
            rows = hf_rows(snapshot, "nuprl/MultiPL-E", config, "test", cache / suite, actual_revision)
            suite_revisions[suite] = actual_revision
            check_rows(rows, "name")
    from datasets_code import load_tasks
    suites = list(LEGACY_SUITES)
    tasks = load_tasks(suites, cache, limit=0)
    for suite in suites:
        selected = [task for task in tasks if task.dataset == suite]
        snapshot.catalog["datasets"][suite] = {"count": len(selected), "languages": dict(Counter(t.language for t in selected)),
                                               "viewer_revision": suite_revisions.get(suite)}
        print(f"Prepared {suite}: {len(selected)}", flush=True)
    snapshot.save()


def selected_suites(scope="functions"):
    if scope not in SCOPES:
        raise ValueError(f"Unknown preparation scope: {scope}")
    return ((*FUNCTION_SUITES,) if scope == "functions" else
            (*REPOSITORY_SUITES,) if scope == "repositories" else (*FUNCTION_SUITES, *REPOSITORY_SUITES))


def selected_file(relative, scope):
    """Classify only the files belonging to selected fixed suites."""
    if scope in ("functions", "all"):
        if relative in {"datasets/functions/manifest.json", "datasets/function-cache/HumanEval.jsonl.gz",
                        "datasets/function-cache/mbpp.jsonl", "datasets/downloads/HumanEval.jsonl.gz.commit.json",
                        "datasets/downloads/mbpp.jsonl.commit.json", "datasets/downloads/multiple.meta.json"}:
            return True
        if any(relative in (f"datasets/functions/raw/{suite}.jsonl", f"datasets/functions/raw/{suite}.jsonl.gz")
               for suite in PLUS):
            return True
        if re.fullmatch(r"datasets/function-cache/multiple_(?:mbpp_)?(?:rs|js|cpp)_\d+\.json", relative):
            return True
    if scope in ("repositories", "all"):
        for suite in REPOSITORY_SUITES:
            if relative in {f"datasets/{suite}/{name}" for name in ("tasks.jsonl", "sealed.jsonl", "manifest.json")}:
                return True
            if re.fullmatch(r"datasets/downloads/" + suite + r"_\d+\.json", relative):
                return True
    return False


def required_files(scope):
    paths = set()
    if scope in ("functions", "all"):
        paths.update(("datasets/function-cache/HumanEval.jsonl.gz", "datasets/function-cache/mbpp.jsonl",
                      "datasets/functions/manifest.json"))
        paths.update(f"datasets/function-cache/{suite}_0.json" for suite in LEGACY_SUITES if suite.startswith("multiple_"))
        paths.update(f"datasets/functions/raw/{suite}.jsonl" for suite in PLUS)
    if scope in ("repositories", "all"):
        paths.update(f"datasets/{suite}/{name}" for suite in REPOSITORY_SUITES
                     for name in ("tasks.jsonl", "sealed.jsonl", "manifest.json"))
    return paths


def verify_official_entries(root, catalog):
    """Validate selected entries without requiring unrelated shared-manifest bytes."""
    target = root / "datasets/functions/manifest.json"
    document = json.loads(target.read_text())
    if document.get("schema_version") != 1 or not isinstance(document.get("suites"), dict):
        raise ValueError("Unsupported official function manifest")
    for suite, (_, revision, _) in PLUS.items():
        entry = document["suites"].get(suite)
        if not isinstance(entry, dict):
            raise ValueError(f"Official function manifest missing {suite}")
        raw = (target.parent / entry.get("path", "")).resolve()
        expected = root / f"datasets/functions/raw/{suite}.jsonl"
        record = catalog["files"][str(expected.relative_to(root))]
        if (raw != expected or entry.get("sha256") != record["sha256"]
                or entry.get("revision") != revision or entry.get("evaluator_version") != "0.3.1"):
            raise ValueError(f"Selected official manifest integrity failure: {suite}")
        if "num_tasks" in entry and entry["num_tasks"] != catalog["datasets"][suite]["count"]:
            raise ValueError(f"Selected official manifest task count mismatch: {suite}")


def merge_official_manifest(snapshot, updates):
    """Add selected suites while preserving existing entries and unrelated data."""
    if set(updates) - set(PLUS):
        raise ValueError("Only selected EvalPlus suites may be prepared")
    path = snapshot.root / "datasets/functions/manifest.json"
    relative = str(path.relative_to(snapshot.root))
    if path.exists():
        document = json.loads(path.read_text())
        if document.get("schema_version") != 1 or not isinstance(document.get("suites"), dict):
            raise ValueError("Unsupported official function manifest")
    else:
        if relative in snapshot.catalog["files"]:
            raise ValueError("Shared official manifest is missing; restore it before merging new suites")
        document = {"schema_version": 1, "suites": {}}
    changed = not path.exists()
    for suite, entry in updates.items():
        previous = document["suites"].get(suite)
        if previous is not None:
            if not isinstance(previous, dict) or any(previous.get(key) != value for key, value in entry.items()):
                raise ValueError(f"Selected official snapshot changed: {suite}; use a new data root")
        else:
            document["suites"][suite] = entry
            changed = True
    if changed:
        contents = json_bytes(document)
        write(path, contents)
        snapshot.catalog["files"][relative] = {"sha256": sha256(contents), "bytes": len(contents)}
    elif relative not in snapshot.catalog["files"]:
        contents = path.read_bytes()
        snapshot.catalog["files"][relative] = {"sha256": sha256(contents), "bytes": len(contents)}
    snapshot.save()


def verify(root, scope="functions", *, quiet=False, verify_hashes=True):
    root = data_root(root)
    required = selected_suites(scope)
    catalog = json.loads((root / "datasets/catalog.json").read_text())
    if catalog.get("schema_version") != 1 or not isinstance(catalog.get("files"), dict):
        raise ValueError("Unsupported data catalog")
    missing = set(required) - catalog.get("datasets", {}).keys()
    if missing:
        raise ValueError(f"Incomplete {scope} preparation: {sorted(missing)}")
    missing_files = required_files(scope) - catalog["files"].keys()
    if missing_files:
        raise ValueError(f"Incomplete {scope} file catalog: {sorted(missing_files)}")
    selected = {relative: record for relative, record in catalog["files"].items() if selected_file(relative, scope)}
    for relative, record in selected.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Manifest path outside data root")
        if relative == "datasets/functions/manifest.json":
            # This shared file may contain an excluded suite. Selected entries
            # are checked against their independently hashed raw data below.
            continue
        if not record.get("sha256") or path.stat().st_size != record["bytes"]:
            raise ValueError(f"Integrity failure: {relative}")
        if verify_hashes and sha256(path.read_bytes()) != record["sha256"]:
            raise ValueError(f"Integrity failure: {relative}")
    if scope in ("functions", "all"):
        verify_official_entries(root, catalog)
    if not quiet:
        print(json.dumps({"scope": scope, "verified_files": len(selected), "hashes_verified": verify_hashes,
                          "datasets": {suite: catalog["datasets"][suite] for suite in required}}, ensure_ascii=False, indent=2))
    return catalog


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=data_root())
    parser.add_argument("--scope", choices=SCOPES, default="functions",
                        help="Function code generation, repository issue resolution, or both; library-use suites are excluded")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--download", action="store_true")
    action.add_argument("--verify", action="store_true")
    action.add_argument("--init", action="store_true")
    args = parser.parse_args(argv)
    if args.init:
        print(initialize(args.data_root))
    elif args.verify:
        verify(args.data_root, args.scope)
    else:
        snapshot = Snapshot(args.data_root)
        if args.scope in ("functions", "all"):
            merge_official_manifest(snapshot, prepare_plus(snapshot))
            prepare_legacy(snapshot)
        if args.scope in ("repositories", "all"):
            prepare_swe(snapshot, "swebench_verified", "SWE-bench/SWE-bench_Verified", 500)
            prepare_swe(snapshot, "swebench_multilingual", "SWE-bench/SWE-bench_Multilingual", 300)
        verify(args.data_root, args.scope)


if __name__ == "__main__":
    main()
