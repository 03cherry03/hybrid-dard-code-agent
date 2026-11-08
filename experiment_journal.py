"""Append-only experiment intent, observations and decisions (no model imports).

The journal is separate from benchmark configuration. A declared development
split is an experimenter's declaration, not automatic proof of data separation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def configuration_fingerprints(run: Path) -> dict:
    """Hash immutable settings, not evolving function phases/identities."""
    result = {}
    for name in ("config.json", "run.json"):
        path = run / name
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            result[name] = canonical_hash(value.get("settings", value))
    return result


def _decode_journal(content: str) -> list[dict]:
    entries = []
    previous = None
    for number, line in enumerate(content.splitlines(), 1):
        if not line.strip():
            raise ValueError(f"Blank journal entry at line {number}")
        entry = json.loads(line)
        recorded = entry.get("entry_sha256")
        payload = {k: v for k, v in entry.items() if k != "entry_sha256"}
        if (entry.get("sequence") != number or entry.get("previous_sha256") != previous
                or recorded != canonical_hash(payload)):
            raise ValueError(f"Journal integrity check failed at line {number}")
        if (number == 1) != (entry.get("kind") == "init"):
            raise ValueError(f"Invalid journal initialization at line {number}")
        entries.append(entry)
        previous = recorded
    return entries


def read_journal(run: Path) -> list[dict]:
    path = Path(run) / "experiment_journal.jsonl"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
        return _decode_journal(stream.read())


def append_entry(run: Path, kind: str, **details) -> dict:
    if kind not in ("init", "observe", "decide"):
        raise ValueError("Unknown journal entry kind")
    run = Path(run).resolve()
    if kind == "init":
        for field in ("purpose", "hypothesis", "change", "split"):
            if not isinstance(details.get(field), str) or not details[field].strip():
                raise ValueError(f"Initialization requires {field}")
        if details["split"] not in ("dev", "test"):
            raise ValueError("split must be dev or test")
    elif not isinstance(details.get("text"), str) or not details["text"].strip():
        raise ValueError("An observation or decision requires text")
    run.mkdir(parents=True, exist_ok=True)
    path = run / "experiment_journal.jsonl"
    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        content = stream.read()
        entries = _decode_journal(content)
        if content and not content.endswith("\n"):
            raise ValueError("Journal lacks a final newline; refusing ambiguous append")
        if kind == "init" and entries:
            raise ValueError("Experiment already initialized; append an observation or use a new run")
        if kind != "init" and not entries:
            raise ValueError("Initialize experiment intent before appending observations")
        payload = {
            "schema_version": 1, "sequence": len(entries) + 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(), "kind": kind,
            "previous_sha256": entries[-1]["entry_sha256"] if entries else None,
            "configuration_fingerprints": configuration_fingerprints(run),
            "details": details,
        }
        entry = {**payload, "entry_sha256": canonical_hash(payload)}
        stream.seek(0, os.SEEK_END)
        stream.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return entry


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Record intent before examining evaluation outcomes")
    init.add_argument("--run", type=Path, required=True)
    init.add_argument("--purpose", required=True)
    init.add_argument("--hypothesis", required=True)
    init.add_argument("--change", required=True, help="One intended change, or baseline")
    init.add_argument("--split", choices=("dev", "test"), required=True)
    init.add_argument("--parent-run", help="Prior run for provenance; never merges its measurements")
    for name in ("observe", "decide"):
        command = commands.add_parser(name)
        command.add_argument("--run", type=Path, required=True)
        command.add_argument("--text", required=True)
        command.add_argument("--evidence", action="append", default=[],
                             help="File:line, task ID or report finding; may be repeated")
        if name == "decide":
            command.add_argument("--next-change", help="Proposed single change for a separate dev run")
    return parser


def main(argv=None):
    args = vars(make_parser().parse_args(argv))
    kind, run = args.pop("command"), args.pop("run")
    try:
        entry = append_entry(run, kind, **args)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({"journal": str(run / "experiment_journal.jsonl"),
                      "sequence": entry["sequence"], "entry_sha256": entry["entry_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
