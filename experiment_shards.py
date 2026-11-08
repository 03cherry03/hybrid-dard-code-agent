"""Deterministic task partitions; no model, dataset or process side effects."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
import hashlib
import json
from pathlib import Path
from typing import TypeVar


SHARDING_ALGORITHM = "stratified_round_robin_v1"
TaskT = TypeVar("TaskT")


def select_manifest(tasks: Iterable[TaskT], path: Path | str | None,
                    expected_sha256: str | None = None) -> tuple[list[TaskT], dict | None]:
    """Select exact public task identities before sharding and pin their file.

    The manifest selects scope only; task contents remain the prepared dataset's
    contents. Source order is preserved regardless of JSON key ordering.
    """
    tasks = list(tasks)
    if path is None:
        if expected_sha256 is not None:
            raise ValueError("A selection manifest hash requires a manifest path")
        return tasks, None
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise ValueError("Selection manifest SHA256 changed after campaign planning")
    try:
        manifest = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Selection manifest must be valid JSON") from exc
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("Selection manifest requires schema_version 1")
    keys = manifest.get("task_keys")
    if not isinstance(keys, list) or not keys or any(
            not isinstance(key, list) or len(key) != 3 or
            any(not isinstance(value, str) or not value for value in key) for key in keys):
        raise ValueError("Selection manifest task_keys must be nonempty [dataset, language, task_id] lists")
    selected = {tuple(key) for key in keys}
    if len(selected) != len(keys):
        raise ValueError("Selection manifest contains duplicate task identities")
    available = {(task.dataset, task.language, task.task_id) for task in tasks}
    if len(available) != len(tasks):
        raise ValueError("Prepared selection contains duplicate task identities")
    missing = selected - available
    if missing:
        raise ValueError(f"Selection manifest identities are missing from selected suites: {sorted(missing)[:5]}")
    result = [task for task in tasks if (task.dataset, task.language, task.task_id) in selected]
    return result, {"path": str(path), "sha256": sha256, "schema_version": 1,
                    "task_count": len(result), "available_task_count": len(tasks)}


def validate_shard(num_shards: int, shard_index: int) -> None:
    if isinstance(num_shards, bool) or not isinstance(num_shards, int) or num_shards < 1:
        raise ValueError("num_shards must be a positive integer")
    if isinstance(shard_index, bool) or not isinstance(shard_index, int) or not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be an integer in [0, num_shards)")


def select_shard(tasks: Iterable[TaskT], num_shards: int = 1, shard_index: int = 0) -> list[TaskT]:
    """Balance dataset/language groups and total task counts without reordering.

    Groups are visited in sorted (dataset, language) order; each group's task
    order remains its original selection order. Carrying the round-robin offset
    between groups balances both each group and the overall task counts within
    one task. All backends/repeats for a selected task stay in the same shard.
    Counts balance, but runtime can still differ by task and language. Empty
    partitions are errors, never a request to execute the full selection.
    """
    validate_shard(num_shards, shard_index)
    tasks = list(tasks)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, task in enumerate(tasks):
        dataset, language = getattr(task, "dataset", None), getattr(task, "language", None)
        if not isinstance(dataset, str) or not isinstance(language, str):
            raise ValueError("Every task must provide string dataset and language attributes")
        groups[(dataset, language)].append(index)
    selected = set()
    offset = 0
    for group in sorted(groups):
        indices = groups[group]
        selected.update(index for position, index in enumerate(indices)
                        if (offset + position) % num_shards == shard_index)
        offset = (offset + len(indices)) % num_shards
    result = [task for index, task in enumerate(tasks) if index in selected]
    if not result:
        raise ValueError(f"Shard {shard_index}/{num_shards} is empty for {len(tasks)} selected tasks; "
                         "select more tasks or reduce num_shards")
    return result
