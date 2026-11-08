"""CPU integration checks for bounded grading without starving generation."""
from contextlib import ExitStack
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from iterative_tuning import AR, CPGRPO, candidate, digest, file_sha, task_id, write_json
from iteration_observability import verify_event_stream
from scripts import iterative_eval_worker as worker
from tradeoff_analysis import load_evaluations


class Backend:
    def __init__(self, generated):
        self.model = torch.nn.Linear(1, 1)
        self.generated = generated
        self.closed = False

    def metadata(self):
        return {"model": "cpu-fixture"}

    def generate(self, *args, **kwargs):
        return {"text": "pass", "wall_s": .01}

    def generate_messages(self, *args, **kwargs):
        self.model(torch.zeros(1, 1))
        self.generated.append(self)
        return {"text": "def identity(x): return x", "wall_s": .01}

    def close(self):
        self.closed = True


def fixture(tmp_path, count=6, configs=None, **options):
    rows = [dict(dataset="humaneval", family="humaneval", language="py",
        task_id=f"fixture/{i}", number=i, source_id=f"fixture/{i}",
        prompt="def identity(x):\n    '''Return x.'''", required_name="identity",
        test_source="assert identity(1) == 1") for i in range(count)]
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("".join(json.dumps(row)+"\n" for row in rows))
    configs = configs or [candidate(AR, decoder="ar", tokens=32, native_steps=1,
                                    forward_budget=32)]
    plan = {"tasks_path": str(tasks), "tasks_sha256": file_sha(tasks), "split": "validation",
        "output": str(tmp_path / "output"), "candidates": configs, "seed": 17,
        "deadline_unix": time.time()+60, "data_root": str(tmp_path),
        "generation_timeout": 10, "grading_timeout": 300, **options}
    path = tmp_path / "plan.json"
    write_json(path, plan)
    return SimpleNamespace(plan=path, shard=0), rows, plan


def patched_run(args, rows, factory, grade, states, waiting, failures):
    original_state = worker.record_worker_state

    def record(output, value):
        original_state(output, value)
        states.append(value)
        if value.get("stage") == "waiting_for_grading":
            waiting.set()

    def run():
        try:
            worker.run(args)
        except BaseException as exc:
            failures.append(exc)

    stack = ExitStack()
    stack.enter_context(patch("models.create_backend", side_effect=factory))
    stack.enter_context(patch("evaluation.preflight_local", return_value={"ready": True}))
    stack.enter_context(patch("evaluation.grade_final", side_effect=grade))
    stack.enter_context(patch.object(worker, "shard_tasks", return_value=rows))
    stack.enter_context(patch.object(worker, "record_worker_state", side_effect=record))
    thread = threading.Thread(target=run)
    thread.start()
    return stack, thread


@pytest.mark.parametrize("values", [
    {"grading_workers": True}, {"grading_workers": 0}, {"grading_workers": 9},
    {"grading_workers": 2.0}, {"grading_queue_size": True}, {"grading_queue_size": 1},
    {"grading_queue_size": 129}, {"grading_queue_size": 4.0},
    {"grading_workers": 8, "grading_queue_size": 7},
])
def test_invalid_concurrency_is_rejected(values):
    with pytest.raises(ValueError):
        worker.grading_options(values)


def test_legacy_defaults_and_new_bounds():
    assert worker.grading_options({}) == (2, 4)
    assert worker.grading_options({"grading_workers": 8, "grading_queue_size": 64}) == (8, 64)
    assert worker.grading_options({"grading_workers": 1, "grading_queue_size": 128}) == (1, 128)


def test_blocked_graders_allow_generation_up_to_bound_and_report_waiting(tmp_path):
    args, rows, plan = fixture(tmp_path, grading_workers=2, grading_queue_size=4)
    release, waiting = threading.Event(), threading.Event()
    generated, states, failures, timeouts = [], [], [], []
    backend = Backend(generated)

    def grade(*args, timeout):
        timeouts.append(timeout)
        assert release.wait(10)
        return {"passed": True, "status": "passed", "grading_wall_s": .01}

    stack, thread = patched_run(args, rows, lambda *a, **kw: backend, grade, states, waiting, failures)
    try:
        assert waiting.wait(5), failures
        assert len(generated) == 4
        current = states[-1]
        assert current["stage"] == "waiting_for_grading"
        assert current["grading_pending"] == 4
        assert current["grading_running"] + current["grading_ready"] + current["grading_queued"] == 4
        assert current["last_generation_completed_unix"] is not None
        assert current["grading_oldest_pending_seconds"] >= 0
    finally:
        release.set()
        thread.join(5)
        stack.close()
    assert not thread.is_alive()
    assert not failures
    assert len(generated) == 6
    assert timeouts == [300]*6
    assert len(list(Path(plan["output"]).glob("shard-0/cases/*/*.json"))) == 6
    assert states[-1]["status"] == "completed"
    for stream in Path(plan["output"]).glob("shard-0/events/*.jsonl"):
        verify_event_stream(stream)


def test_model_switch_does_not_wait_for_cpu_only_grading(tmp_path):
    configs = [candidate(AR, decoder="ar", tokens=32, native_steps=1),
               candidate(CPGRPO, decoder="standard", tokens=32, native_steps=2)]
    args, rows, _ = fixture(tmp_path, count=1, configs=configs)
    release, waiting = threading.Event(), threading.Event()
    generated, states, failures = [], [], []
    backends = [Backend(generated), Backend(generated)]
    remaining = iter(backends)

    def grade(*args, timeout):
        assert release.wait(10)
        return {"passed": True, "status": "passed"}

    stack, thread = patched_run(args, rows, lambda *a, **kw: next(remaining), grade, states, waiting, failures)
    try:
        assert waiting.wait(5), failures
        assert len(generated) == 2
        assert backends[0].closed
        assert states[-1]["waiting_reason"] == "final_drain"
    finally:
        release.set()
        thread.join(5)
        stack.close()
    assert not thread.is_alive()
    assert not failures
    assert all(backend.closed for backend in backends)


def test_deadline_does_not_wait_for_blocked_running_or_queued_graders(tmp_path):
    args, rows, plan = fixture(tmp_path, count=3, grading_workers=1, grading_queue_size=3)
    # The test expires the already-read plan only after the queue is blocked.
    original_read = worker.read_json
    release, waiting = threading.Event(), threading.Event()
    generated, states, failures = [], [], []
    backend = Backend(generated)
    calls = []

    def grade(*args, timeout):
        calls.append(timeout)
        assert release.wait(10)
        return {"passed": True, "status": "passed"}

    with patch.object(worker, "read_json", side_effect=lambda path, *a, **kw:
                      plan if Path(path) == args.plan else original_read(path, *a, **kw)):
        stack, thread = patched_run(args, rows, lambda *a, **kw: backend, grade, states, waiting, failures)
        try:
            assert waiting.wait(5), failures
            plan["deadline_unix"] = time.time() - 1
            thread.join(6)
            assert not thread.is_alive(), "collect or executor shutdown ignored the campaign deadline"
            assert len(failures) == 1 and isinstance(failures[0], TimeoutError)
            assert len(calls) == 1, "queued graders should be cancelled on deadline"
            assert backend.closed
            assert not list(Path(plan["output"]).glob("shard-0/cases/*/*.json"))
        finally:
            release.set()
            thread.join(5)
            stack.close()


def analysis_phase(root, seed, **options):
    root.mkdir()
    tasks = root / "tasks.jsonl"
    task = {"dataset": "fixture", "task_id": "1", "source_id": "source/1", "language": "py"}
    tasks.write_text(json.dumps(task)+"\n")
    config = candidate(AR, decoder="ar", tokens=32, native_steps=1)
    write_json(root / "evaluation_plan.json", {"split": "validation", "seed": seed,
        "tasks_path": str(tasks), "tasks_sha256": file_sha(tasks), "candidates": [config], **options})
    row = {"candidate_id": config["id"], "config_sha256": digest(config), "task_key": task_id(task),
        "task_sha256": digest(task), "source_id": task["source_id"], "language": "py",
        "passed": True, "status": "passed", "generation_seconds": 1., "forward_calls": 1,
        "forced_tokens": 0, "infrastructure_error": False}
    write_json(root / "shard-0/cases" / config["id"] / (task_id(task)+".json"), row)
    return root


def test_analysis_normalizes_legacy_concurrency_and_never_pools_changed_protocols(tmp_path):
    old = analysis_phase(tmp_path / "old", 1)
    explicit = analysis_phase(tmp_path / "explicit", 2, grading_workers=2, grading_queue_size=4)
    new = analysis_phase(tmp_path / "new", 3, grading_workers=8, grading_queue_size=64)
    records, _ = load_evaluations([old, explicit, new])
    assert len(records) == 2
    by_workers = {r["evaluation_protocol"]["grading_workers"]: r for r in records}
    assert by_workers[2]["seeds"] == {1, 2}
    assert by_workers[8]["seeds"] == {3}
