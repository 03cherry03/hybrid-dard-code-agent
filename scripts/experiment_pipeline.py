"""Pure stage planning for the complete paired four-GPU experiment pipeline.

No imports of model libraries, environment probes, subprocess execution or
filesystem writes occur here. The caller owns supervision, durable state,
notifications and the runtime environment. In particular, it must set
HYBRID_DARD_DATA_ROOT/HYBRID_DARD_ROOT to the planned data root before running
the preparation script.
"""
from __future__ import annotations

from pathlib import Path
import sys


def build_stages(*, repo: Path, data_root: Path, output: Path,
                 gpus: str = "0,1,2,3", prepare: bool = False,
                 python: str = sys.executable, scope: str = "functions",
                 priority_codegen: bool = False, deadline_unix: float | None = None,
                 focused_codegen: bool = False) -> list[dict]:
    """Return deterministic named argv stages, without starting any work.

    Preparation is opt-in. Existing prepared datasets are verified, never
    downloaded. Native repository preparation finishes before either repository
    plan is created, preserving environment locks across smoke/full runs.
    Each four-GPU stage itself executes all three paired backends. Test failures
    in generated code are measured outcomes; runner/infrastructure failures are
    conveyed by nonzero subprocess status to the supervisor.
    """
    repo, data_root, output = (Path(path).expanduser().resolve()
                              for path in (repo, data_root, output))
    if not isinstance(prepare, bool):
        raise ValueError("prepare must be a boolean")
    if scope not in ("functions", "repositories", "all"):
        raise ValueError("scope must be functions, repositories or all")
    if priority_codegen and (scope != "functions" or deadline_unix is None):
        raise ValueError("priority code generation requires functions scope and an absolute deadline")
    if focused_codegen and (scope != "functions" or deadline_unix is None or priority_codegen):
        raise ValueError("focused code generation requires functions scope, an absolute deadline, and no priority-codegen flag")
    include_functions = scope in ("functions", "all")
    include_repositories = scope in ("repositories", "all")
    if not isinstance(python, str) or not python.strip():
        raise ValueError("python must be a nonempty executable path/name")
    if not isinstance(gpus, str):
        raise ValueError("gpus must contain four comma-separated GPU identifiers")
    selected = [value.strip() for value in gpus.split(",")]
    if len(selected) != 4 or any(not value for value in selected) or len(set(selected)) != 4:
        raise ValueError("gpus must contain four distinct, nonempty identifiers")
    gpus = ",".join(selected)
    stages: list[dict] = []

    def add(name, command, *, downloads=False):
        stages.append({"name": name, "command": [str(value) for value in command],
                       "cwd": str(repo), "downloads": downloads})

    def script(name):
        return repo / "scripts" / name

    def gpu_run(name, kind, directory, *, limit, repeats, tokens, standard_steps,
                block_size=128, dard_steps=192, split="test"):
        command = ["bash", script("run_four_gpu.sh"), kind,
                   "--data-root", data_root, "--gpus", gpus,
                   "--limit", limit, "--repeats", repeats, "--seed", 42,
                   "--tokens", tokens, "--standard-steps", standard_steps,
                   "--block-size", block_size, "--dard-steps", dard_steps,
                   "--quant", "bf16", "--diagnostics", "summary", "--split", split,
                   "--output", directory, "--execute", "--fail-fast"]
        if kind == "repository":
            command += ["--task-timeout", 600, "--max-steps", 30]
        else:
            from datasets_code import CODE_GENERATION_SUITES
            command += ["--suites", ",".join(CODE_GENERATION_SUITES)]
        add(name, command)

    def postprocess(name, kind, directory, action):
        # Stored campaign plans supply generation budgets/GPU bindings. They
        # must not be re-created from defaults during CPU-only postprocessing.
        add(name, ["bash", script("run_four_gpu.sh"), kind,
                   "--output", directory, action])

    verified = data_root / "datasets/swebench_verified/tasks.jsonl"
    multilingual = data_root / "datasets/swebench_multilingual/tasks.jsonl"
    add("gpu_idle_check", [python, script("all_experiments.py"), "--check-gpus", "--gpus", gpus])
    add("data_verify", [python, repo / "prepare_data.py", "--data-root", data_root, "--scope", scope, "--verify"])
    if include_repositories:
        add("repository_data_check", [python, repo / "repository_runner.py", "check-data",
                                      "--tasks", verified, multilingual, "--limit", 0])

    if prepare:
        add("setup_" + scope, ["bash", script("setup_env.sh"), scope], downloads=True)
        add("download_models", [data_root / "envs/inference/bin/python", script("download_models.py"),
                                "--data-root", data_root, "--download"], downloads=True)
        for suite in (("swebench_verified", "swebench_multilingual") if include_repositories else ()):
            name = "prepare_verified_repositories" if suite == "swebench_verified" else "prepare_multilingual_repositories"
            add(name, [data_root / "envs/swebench/bin/python", script("prepare_repositories.py"), "--data-root", data_root,
                       "--suite", suite, "--limit", 0, "--prepare"], downloads=True)

    if focused_codegen:
        add("focused_codegen", [python, script("focused_experiments.py"), "functions",
            "--data-root", data_root, "--gpus", gpus, "--output", output / "functions",
            "--deadline-unix", deadline_unix, "--execute"])
        add("focused_analysis", [python, script("focused_experiments.py"), "functions",
            "--output", output / "functions", "--analyze"])
        return stages

    function_smoke = output / "function-smoke"
    repository_smoke = output / "repository-smoke"
    functions = output / "functions"
    repositories = output / "repositories"
    if include_functions:
        gpu_run("function_smoke", "functions", function_smoke,
                limit=2 if priority_codegen else 1, repeats=1,
                tokens=512 if priority_codegen else 128, standard_steps=512 if priority_codegen else 128,
                block_size=128 if priority_codegen else 64, dard_steps=192 if priority_codegen else 96, split="dev")
        if priority_codegen:
            stages[-1]["command"] += ["--backends", "ar,dard"]
            for mode in ("priority", "standard"):
                directory = output / ("functions-" + mode)
                add("functions_" + mode, [python, script("priority_experiments.py"), "functions",
                    "--mode", mode, "--data-root", data_root, "--output", directory,
                    "--calibration", function_smoke, "--gpus", gpus,
                    "--deadline-unix", deadline_unix, "--execute"])
            add("priority_analysis", [python, script("priority_experiments.py"), "functions",
                "--output", output, "--analyze"])
            return stages
    # A per-file limit of two selects four total SWE tasks, ensuring that the
    # deterministic four-way partition never produces an empty worker.
    if include_repositories:
        gpu_run("repository_smoke", "repository", repository_smoke,
                limit=2, repeats=1, tokens=1024, standard_steps=128, split="dev")
        postprocess("repository_smoke_grade", "repository", repository_smoke, "--grade")
    if include_functions:
        gpu_run("functions_full", "functions", functions,
                limit=0, repeats=3, tokens=512, standard_steps=512)
    if include_repositories:
        gpu_run("repositories_full", "repository", repositories,
                limit=0, repeats=3, tokens=1024, standard_steps=128)
        postprocess("repositories_grade", "repository", repositories, "--grade")
    if include_functions:
        postprocess("functions_analysis", "functions", functions, "--analyze")
    if include_repositories:
        postprocess("repositories_analysis", "repository", repositories, "--analyze")
    return stages
