"""Offline contracts for a source-preserving, reviewed logging handoff."""
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import upgrade_iterative_logging as upgrade


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    output, source = tmp_path / "experiment", tmp_path / "repository"
    old_source = output / "source"
    write(old_source / "scripts/iterative_experiments.py", "# original controller\n")
    write(source / "scripts/iterative_experiments.py", "# controller with events\n")
    write(source / "iteration_observability.py", "# local event writer\n")
    write(output / "data/train.json", "[]\n")
    plan = {"schema_version": 1, "output": str(output), "source": str(old_source),
            "source_sha256": {"scripts/iterative_experiments.py": upgrade.file_sha(old_source / "scripts/iterative_experiments.py")},
            "data_root": str(tmp_path / "data_root"), "data": str(output / "data"),
            "data_sha256": {"train.json": upgrade.file_sha(output / "data/train.json")},
            "package_versions": {"example": "1.0"}, "train_steps": 100,
            "seed": 10, "deadline_unix": 1234, "gpus": [{"uuid": "GPU-0"}],
            "initial_candidates": [{"id": "candidate-1", "temperature": .4}]}
    # Noncanonical whitespace ensures the archive preserves bytes, not just JSON.
    original = json.dumps(plan, indent=3).encode() + b"\n\n"
    (output / "iteration_plan.json").write_bytes(original)
    case = output / "validation/shard-0/cases/candidate-1/task-1.json"
    write(case, '{"passed": true}\n')
    write(output / "validation/evaluation_plan.json", '{"frozen": true}\n')
    verify = Mock()
    monkeypatch.setattr(upgrade, "verify_plan", verify)
    monkeypatch.setattr(upgrade.subprocess, "run", lambda command, **kwargs: SimpleNamespace(
        stdout="LoadState=loaded\nActiveState=inactive\nSubState=dead\nMainPID=0\n" if command[0] == "systemctl" else ""))
    return SimpleNamespace(output=output, source=source, plan=plan, original=original,
                           case=case, verify=verify, audit=output / "audit" / upgrade.MIGRATION)


def prepare(setup):
    return upgrade.prepare(setup.output, setup.source, "experiment.service")


def approve(setup, **changes):
    receipt = json.loads((setup.audit / "migration_receipt.json").read_text())
    validation = {"status": "passed", "prepared_plan_sha256": receipt["prepared_plan_sha256"], **changes}
    write(setup.output / "audit/prelaunch_logging_validation.json", json.dumps(validation))


def test_prepare_archives_exact_bytes_and_preserves_live_files(setup, monkeypatch):
    command = Mock(side_effect=AssertionError("offline preparation must not touch services"))
    monkeypatch.setattr(upgrade.subprocess, "run", command)
    old_source = (Path(setup.plan["source"]) / "scripts/iterative_experiments.py").read_bytes()
    receipt = prepare(setup)
    assert (setup.output / "iteration_plan.json").read_bytes() == setup.original
    assert (setup.audit / "previous_iteration_plan.json").read_bytes() == setup.original
    assert (Path(setup.plan["source"]) / "scripts/iterative_experiments.py").read_bytes() == old_source
    assert setup.case.read_text() == '{"passed": true}\n'
    new = json.loads((setup.audit / "prepared_iteration_plan.json").read_bytes())
    assert {k for k in new if new.get(k) != setup.plan.get(k)} == upgrade.PLAN_CHANGES
    assert new["previous_plan"]["sha256"] == upgrade.sha(setup.original)
    assert receipt["previous_case_count"] == 1
    assert "+# controller with events" in (setup.audit / "source_diff.patch").read_text()
    command.assert_not_called()


def test_preparation_is_idempotent_without_overwriting_snapshot(setup):
    first = prepare(setup)
    write(setup.source / "scripts/iterative_experiments.py", "# newer work must not overwrite prepared code\n")
    assert prepare(setup) == first
    assert (setup.output / upgrade.SNAPSHOT / "scripts/iterative_experiments.py").read_text() == "# controller with events\n"


def test_preparation_requires_existing_original_plan(setup):
    (setup.output / "iteration_plan.json").unlink()
    with pytest.raises(ValueError, match="Existing regular"):
        prepare(setup)
    assert not (setup.output / upgrade.SNAPSHOT).exists()


def test_preparation_rejects_source_symlinks(setup):
    (setup.source / "outside.py").symlink_to(setup.output / "iteration_plan.json")
    with pytest.raises(ValueError, match="regular files"):
        prepare(setup)


def test_activation_changes_only_root_plan_and_keeps_new_completed_cases(setup):
    prepare(setup)
    approve(setup)
    new_case = setup.case.with_name("task-2.json")
    write(new_case, '{"passed": false}\n')
    result = upgrade.activate(setup.output)
    assert result["status"] == "activated"
    assert result["previous_case_count"] == 2
    assert (setup.output / "iteration_plan.json").read_bytes() == (setup.audit / "prepared_iteration_plan.json").read_bytes()
    assert (setup.audit / "previous_iteration_plan.json").read_bytes() == setup.original
    assert setup.case.read_text() == '{"passed": true}\n'
    assert new_case.read_text() == '{"passed": false}\n'
    assert setup.verify.call_count >= 3


@pytest.mark.parametrize("changes", [{"status": "failed"}, {"prepared_plan_sha256": "bad"}])
def test_activation_requires_passed_hash_bound_validation(setup, changes):
    prepare(setup)
    approve(setup, **changes)
    with pytest.raises(ValueError, match="Passed validation receipt"):
        upgrade.activate(setup.output)
    assert (setup.output / "iteration_plan.json").read_bytes() == setup.original


@pytest.mark.parametrize("service,gpus", [
    ("LoadState=loaded\nActiveState=active\nMainPID=100\n", ""),
    ("LoadState=loaded\nActiveState=inactive\nMainPID=100\n", ""),
    ("LoadState=not-found\nActiveState=inactive\nMainPID=0\n", ""),
    ("LoadState=loaded\nActiveState=inactive\nMainPID=0\n", "GPU-0, 100\n"),
])
def test_activation_refuses_running_or_unknown_service_and_busy_gpu(setup, monkeypatch, service, gpus):
    prepare(setup)
    approve(setup)
    monkeypatch.setattr(upgrade.subprocess, "run", lambda cmd, **kwargs: SimpleNamespace(stdout=service if cmd[0] == "systemctl" else gpus))
    with pytest.raises(RuntimeError):
        upgrade.activate(setup.output)
    assert (setup.output / "iteration_plan.json").read_bytes() == setup.original


def test_activation_rejects_changed_completed_case(setup):
    prepare(setup)
    approve(setup)
    write(setup.case, '{"passed": false}\n')
    with pytest.raises(ValueError, match="Previously completed"):
        upgrade.activate(setup.output)
    assert (setup.output / "iteration_plan.json").read_bytes() == setup.original


def test_activation_refuses_existing_controller_lock_even_if_status_is_stale(setup):
    prepare(setup)
    approve(setup)
    with (setup.output / "iteration.lock").open("a") as running:
        fcntl.flock(running, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            upgrade.activate(setup.output)
    assert (setup.output / "iteration_plan.json").read_bytes() == setup.original


def test_activation_rejects_root_plan_drift(setup):
    prepare(setup)
    approve(setup)
    changed = dict(setup.plan, seed=999)
    (setup.output / "iteration_plan.json").write_bytes(upgrade.json_bytes(changed))
    with pytest.raises(ValueError, match="Root plan differs"):
        upgrade.activate(setup.output)
    assert json.loads((setup.output / "iteration_plan.json").read_bytes())["seed"] == 999


def test_activation_rejects_snapshot_modification(setup):
    prepare(setup)
    approve(setup)
    write(setup.output / upgrade.SNAPSHOT / "iteration_observability.py", "# modified\n")
    with pytest.raises(ValueError, match="Prepared source inventory changed"):
        upgrade.activate(setup.output)


def test_activation_rejects_parameter_change_even_if_receipt_hash_is_rebound(setup):
    prepare(setup)
    plan_path = setup.audit / "prepared_iteration_plan.json"
    plan = json.loads(plan_path.read_text())
    plan["train_steps"] += 1
    plan_path.write_bytes(upgrade.json_bytes(plan))
    receipt_path = setup.audit / "migration_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["prepared_plan_sha256"] = upgrade.file_sha(plan_path)
    receipt_path.write_bytes(upgrade.json_bytes(receipt))
    approve(setup)
    with pytest.raises(ValueError, match="experiment parameters"):
        upgrade.activate(setup.output)


def test_activation_rechecks_idle_immediately_before_replace(setup, monkeypatch):
    prepare(setup)
    approve(setup)
    check = Mock(side_effect=[{}, RuntimeError("service resumed")])
    monkeypatch.setattr(upgrade, "require_idle", check)
    with pytest.raises(RuntimeError, match="service resumed"):
        upgrade.activate(setup.output)
    assert (setup.output / "iteration_plan.json").read_bytes() == setup.original
    assert json.loads((setup.audit / "activation_receipt.json").read_text())["status"] == "activation_intent"


def test_activation_recovers_after_replace_before_final_receipt(setup, monkeypatch):
    prepare(setup)
    approve(setup)
    real_write = upgrade.atomic_bytes

    def interrupted_write(path, content):
        if Path(path).name == "activation_receipt.json" and json.loads(content)["status"] == "activated":
            raise OSError("simulated post-replace interruption")
        return real_write(path, content)

    monkeypatch.setattr(upgrade, "atomic_bytes", interrupted_write)
    with pytest.raises(OSError):
        upgrade.activate(setup.output)
    assert (setup.output / "iteration_plan.json").read_bytes() == (setup.audit / "prepared_iteration_plan.json").read_bytes()
    monkeypatch.setattr(upgrade, "atomic_bytes", real_write)
    assert upgrade.activate(setup.output)["status"] == "activated"


def test_activation_never_stops_starts_or_modifies_units(setup, monkeypatch):
    prepare(setup)
    approve(setup)
    commands = []

    def command(args, **kwargs):
        commands.append(args)
        return SimpleNamespace(stdout="LoadState=loaded\nActiveState=inactive\nMainPID=0\n" if args[0] == "systemctl" else "")

    monkeypatch.setattr(upgrade.subprocess, "run", command)
    upgrade.activate(setup.output)
    assert all(command[:3] == ["systemctl", "--user", "show"] or command[0] == "nvidia-smi" for command in commands)
