"""Offline review of resume, holdout sealing and hard stage deadlines."""
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from iterative_tuning import AR, CPGRPO, candidate, read_json, write_json
from scripts import iterative_experiments as controller


def summary(config, passed, seconds):
    return {"candidate": config, "cases": 100, "passed": passed,
            "pass_rate": passed / 100, "mean_generation_seconds": seconds,
            "by_language": {}}


@pytest.fixture
def controller_fixture(tmp_path, monkeypatch):
    ar = candidate(AR, decoder="ar", temperature=0, top_p=1, forward_budget=512)
    old = candidate(CPGRPO)
    trained = candidate(CPGRPO, temperature=.4)
    baseline = [summary(ar, 80, 2), summary(old, 40, 4)]
    improved = [summary(ar, 80, 2), summary(old, 40, 4), summary(trained, 50, 3)]
    plan = {"output": str(tmp_path), "source": str(tmp_path / "source"), "data_root": str(tmp_path), "data": str(tmp_path / "data"),
            "gpus": [{"uuid": "GPU-" + str(i)} for i in range(4)],
            "failure_limit": 3, "deadline_unix": 1_000_000, "final_reserve_seconds": 1000,
            "max_rounds": 1, "patience": 1, "accuracy_margin": .03,
            "initial_candidates": [ar, old]}
    write_json(tmp_path / "iteration_plan.json", plan)
    monkeypatch.setattr(controller, "verify_plan", lambda plan: None)
    monkeypatch.setattr(controller, "prepare_teacher_corpus", Mock())
    monkeypatch.setattr(controller.time, "time", lambda: 1000)
    monkeypatch.setattr(controller.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=""))
    train = Mock(return_value={"path": str(tmp_path / "adapter"), "sha256": {}})
    monkeypatch.setattr(controller, "train", train)
    monkeypatch.setattr(controller, "round_candidates", lambda *args: [trained])
    return plan, baseline, improved, train


def test_holdout_interruption_resumes_frozen_selection_without_training_again(controller_fixture, monkeypatch):
    plan, baseline, improved, train = controller_fixture
    seen_holdout = []
    fail_once = [True]

    def evaluate(plan, label, configs, split="validation", deadline=None):
        if split == "holdout":
            seen_holdout.append(copy.deepcopy(configs))
            assert Path(plan["output"], "final_selection.json").exists()
            if fail_once[0]:
                fail_once[0] = False
                raise RuntimeError("simulated interruption during holdout")
            return improved[:1] + improved[2:]
        return baseline if label == "baseline_validation" else improved

    monkeypatch.setattr(controller, "evaluate", evaluate)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        controller.execute(plan)
    frozen = read_json(Path(plan["output"]) / "final_selection.json")
    assert train.call_count == 1
    controller.execute(plan)
    assert train.call_count == 1
    assert seen_holdout[0] == seen_holdout[1]
    assert read_json(Path(plan["output"]) / "final_selection.json") == frozen
    assert read_json(Path(plan["output"]) / "iteration_state.json")["status"] == "completed"
    assert frozen["deployment_candidate"]["decoder"] == "ar"


def test_completed_controller_does_not_rerun_evaluation(controller_fixture, monkeypatch):
    plan, _, _, train = controller_fixture
    write_json(Path(plan["output"]) / "iteration_state.json", {"status": "completed"})
    evaluate = Mock()
    monkeypatch.setattr(controller, "evaluate", evaluate)
    controller.execute(plan)
    evaluate.assert_not_called()
    train.assert_not_called()


def test_changed_final_selection_stops_before_reading_holdout(controller_fixture, monkeypatch):
    plan, baseline, improved, _ = controller_fixture
    write_json(Path(plan["output"]) / "final_selection.json", {"different": "selection"})
    seen = []

    def evaluate(plan, label, configs, split="validation", deadline=None):
        seen.append(split)
        return baseline if label == "baseline_validation" else improved

    monkeypatch.setattr(controller, "evaluate", evaluate)
    with pytest.raises(ValueError, match="Immutable artifact changed"):
        controller.execute(plan)
    assert "holdout" not in seen


def test_deadline_terminates_every_live_worker(monkeypatch, tmp_path):
    children = [SimpleNamespace(poll=lambda: None, pid=i) for i in range(4)]
    terminate = Mock()
    monkeypatch.setattr(controller, "terminate", terminate)
    monkeypatch.setattr(controller.time, "time", lambda: 20)
    with pytest.raises(TimeoutError, match="deadline reached"):
        controller.wait_children({"output": str(tmp_path)}, children, 10, "validation")
    terminate.assert_called_once_with(children)


def test_retry_limit_prevents_new_training(controller_fixture, monkeypatch):
    plan, _, _, train = controller_fixture
    write_json(Path(plan["output"]) / "iteration_state.json", {"status": "failed", "failure_count": 3})
    evaluate = Mock()
    monkeypatch.setattr(controller, "evaluate", evaluate)
    with pytest.raises(RuntimeError, match="Retry limit reached"):
        controller.execute(plan)
    evaluate.assert_not_called()
    train.assert_not_called()


def test_post_stop_records_and_notifies_an_uncaught_service_loss(controller_fixture, monkeypatch):
    import experiment_notifications
    plan, _, _, _ = controller_fixture
    path = Path(plan["output"]) / "iteration_state.json"
    write_json(path, {"status": "running", "stage": "training", "failure_count": 0})
    monkeypatch.setenv("SERVICE_RESULT", "signal")
    notify = Mock(return_value={"smtp_accepted": True})
    monkeypatch.setattr(experiment_notifications, "notify", notify)
    monkeypatch.setattr(experiment_notifications, "flush_pending", Mock())
    assert controller.main(["--output", plan["output"], "--post-stop"]) == 0
    assert read_json(path)["status"] == "failed"
    assert read_json(path)["failure_count"] == 1
    notify.assert_called_once()
    controller.main(["--output", plan["output"], "--post-stop"])
    assert notify.call_count == 1


def test_training_resume_does_not_pass_a_second_initial_adapter(tmp_path, monkeypatch):
    corpus = tmp_path / "data" / "training_examples.jsonl"
    corpus.parent.mkdir()
    corpus.write_text('{"source_id":"train-only-fixture"}\n')
    plan = {"output": str(tmp_path), "source": str(tmp_path / "source"),
            "data_root": str(tmp_path), "data": str(tmp_path / "data"),
            "python": "/unused/python", "data_sha256": {"training_examples.jsonl": controller.file_sha(corpus)},
            "seed": 42, "train_steps": 100, "deadline_unix": 100000, "final_reserve_seconds": 1000,
            "gpus": [{"uuid": "GPU-" + str(i)} for i in range(4)]}
    output = tmp_path / "round-02" / "training"
    output.mkdir(parents=True)
    write_json(output / "latest.json", {"checkpoint": "checkpoint-000010"})
    previous = tmp_path / "initial_adapter"
    previous.mkdir()
    (previous / "adapter_config.json").write_text("{}")
    (previous / "adapter_model.safetensors").write_bytes(b"initial")
    incumbent = summary(candidate(CPGRPO, adapter=controller.adapter_identity(previous)), 50, 3)
    monkeypatch.setattr(controller, "verify_plan", lambda plan: None)
    monkeypatch.setattr(controller, "worker_env", lambda *args: {})
    popen = Mock(return_value=SimpleNamespace(pid=123))
    monkeypatch.setattr(controller.subprocess, "Popen", popen)

    def finish(*args):
        final = output / "final_adapter"
        final.mkdir()
        (final / "adapter_config.json").write_text("{}")
        (final / "adapter_model.safetensors").write_bytes(b"final")
        write_json(output / "training_state.json", {"status": "completed"})

    monkeypatch.setattr(controller, "wait_children", finish)
    controller.train(plan, 2, incumbent)
    command = popen.call_args.args[0]
    assert "--resume" in command
    assert "--initial-adapter" not in command
