"""Bounded native commands shared by function and repository evaluations.

Process groups and resource limits bound accidental hangs; they are not a host
filesystem or network security boundary. Limits are set in an exec wrapper,
never in preexec_fn in the multithreaded inference process.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import threading
import time

_COMMAND_GROUP_ENV = "HYBRID_DARD_COMMAND_PROCESS_GROUP"


def _process_start_time(pid: int) -> str | None:
    try:
        # comm can contain spaces and parentheses; fields after the final ')'
        # start at process state (field 3), and starttime is field 22.
        return (Path("/proc") / str(pid) / "stat").read_text().rsplit(") ", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def _remember_descendants(owned: dict[int, str]) -> None:
    pending = list(owned)
    visited = set()
    while pending:
        pid = pending.pop()
        if pid in visited or _process_start_time(pid) != owned[pid]:
            continue
        visited.add(pid)
        try:
            threads = list((Path("/proc") / str(pid) / "task").iterdir())
        except OSError:
            continue
        for thread in threads:
            try:
                children = [int(value) for value in (thread / "children").read_text().split()]
            except (OSError, ValueError):
                continue
            for child in children:
                identity = _process_start_time(child)
                if identity is not None:
                    owned.setdefault(child, identity)
                    pending.append(child)


def _stop_owned_processes(owned: dict[int, str]) -> None:
    _remember_descendants(owned)
    for pid, identity in reversed(list(owned.items())):
        if _process_start_time(pid) != identity:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@dataclass
class CommandResult:
    returncode: int
    output: str
    wall_s: float = 0.0
    timed_out: bool = False
    truncated: bool = False

    @property
    def output_truncated(self) -> bool:
        return self.truncated


def environment_fingerprint(prefix: str | Path) -> str:
    """Detect ordinary prefix mutations using paths, metadata and link targets.

    This deliberately excludes Python bytecode caches, which imports can create.
    It is a reproducibility check, not a cryptographic content attestation.
    """
    root = Path(prefix).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Environment prefix must be a directory")
    digest = hashlib.sha256()
    for current, directories, files in os.walk(root, followlinks=False):
        directories[:] = sorted(name for name in directories if name != "__pycache__")
        for name in sorted(directories + files):
            if name.endswith((".pyc", ".pyo")):
                continue
            path = Path(current) / name
            relative = str(path.relative_to(root))
            stat = path.lstat()
            if path.is_symlink():
                entry = (relative, "link", os.readlink(path))
            elif path.is_file():
                entry = (relative, "file", stat.st_size, stat.st_mtime_ns, stat.st_mode)
            else:
                entry = (relative, "directory", stat.st_mode)
            digest.update(json.dumps(entry, separators=(",", ":")).encode() + b"\n")
    return digest.hexdigest()


def clone_environment(prefix: str | Path, destination: str | Path, *, timeout: float) -> Path:
    """Relocate an offline Conda clone without sharing writable package files."""
    source, destination = Path(prefix).resolve(strict=True), Path(destination).resolve()
    if not (source / "conda-meta").is_dir():
        raise RuntimeError("Prepared native environment must be a Conda prefix")
    if destination.exists():
        raise ValueError("Trial environment destination already exists")
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda:
        raise RuntimeError("Conda is required to clone the prepared local environment")
    result = run_bounded_command([conda, "create", "--offline", "--copy", "--yes", "--clone",
                                  str(source), "--prefix", str(destination)],
                                 timeout=timeout, max_output_bytes=8000)
    if result.returncode or result.timed_out:
        raise RuntimeError("Local environment cloning failed: " + result.output[-4000:])
    return destination


def native_environment(prefix: str | Path, workspace: str | Path,
                       runtime_dir: str | Path) -> dict[str, str]:
    """Use the prepared prefix and keep ordinary tool caches inside this trial."""
    prefix, workspace, runtime = Path(prefix), Path(workspace), Path(runtime_dir)
    for name in ("home", "tmp", "cache", "cargo", "go", "gradle", "npm"):
        (runtime / name).mkdir(parents=True, exist_ok=True)
    seed_home = prefix / "native-home"
    if seed_home.exists():
        shutil.copytree(seed_home, runtime / "home", dirs_exist_ok=True, symlinks=True)
    env = dict(os.environ)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONUSERBASE", None)
    env.update(PATH=str(prefix / "bin") + os.pathsep + env.get("PATH", ""),
               CONDA_PREFIX=str(prefix), VIRTUAL_ENV=str(prefix),
               PYTHONPATH=os.pathsep.join((str(workspace), str(workspace / "src"))),
               PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
               HOME=str(runtime / "home"), TMPDIR=str(runtime / "tmp"),
               XDG_CACHE_HOME=str(runtime / "cache"),
               CARGO_TARGET_DIR=str(runtime / "cargo"), GOCACHE=str(runtime / "go"),
               GRADLE_USER_HOME=str(runtime / "gradle"), npm_config_cache=str(runtime / "npm"),
               CUDA_VISIBLE_DEVICES="", GIT_CONFIG_NOSYSTEM="1",
               GIT_CONFIG_GLOBAL=os.devnull)
    config_path = prefix / "native-environment.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        def expand(value):
            if not isinstance(value, str) or "\0" in value:
                raise ValueError("Native environment values must be strings")
            for key, replacement in (("PREFIX", prefix), ("WORKSPACE", workspace), ("RUNTIME", runtime)):
                value = value.replace("${" + key + "}", str(replacement))
            return value
        env.update({key: expand(value) for key, value in config.get("environment", {}).items()})
        entries = [expand(value) for value in config.get("path_entries", [])]
        if entries:
            env["PATH"] = os.pathsep.join(entries + [env["PATH"]])
    return env


def run_bounded_command(argv: list[str], *, cwd: Path | str | None = None,
                        env: dict[str, str] | None = None, timeout: float,
                        max_output_bytes: int = 40000, input_text: str | None = None,
                        memory_bytes: int | None = None, cpu_seconds: int | None = None,
                        max_processes: int | None = None) -> CommandResult:
    """Drain stdout/stderr with bounded retention and always stop the process group."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise TimeoutError("No time remains for command")
    if not argv or any(not isinstance(arg, str) or "\0" in arg for arg in argv):
        raise ValueError("Command must be a nonempty list of strings")
    if not isinstance(max_output_bytes, int) or max_output_bytes < 0:
        raise ValueError("Output limit must be a nonnegative integer")
    limits = {name: value for name, value in (("RLIMIT_AS", memory_bytes),
              ("RLIMIT_CPU", cpu_seconds), ("RLIMIT_NPROC", max_processes)) if value is not None}
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1
           for value in limits.values()):
        raise ValueError("Resource limits must be positive integers")
    # Nested native workers inherit the outer supervising group. Otherwise a
    # worker killed at its deadline could leave its evaluator in a new session.
    inherited_group = os.environ.get(_COMMAND_GROUP_ENV)
    group_owned = inherited_group != str(os.getpgrp())
    command = [sys.executable, str(Path(__file__).resolve()), "--limit-wrapper",
               json.dumps(limits), *argv]
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=cwd, env=env, start_new_session=group_owned,
        stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    identity = _process_start_time(process.pid)
    owned = {process.pid: identity} if identity is not None else {}

    def stop_group():
        _stop_owned_processes(owned)
        if group_owned:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    if input_text is not None:
        def write_input():
            try:
                process.stdin.write(input_text.encode())
                process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                pass
            finally:
                try:
                    process.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
        threading.Thread(target=write_input, daemon=True).start()
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    chunks, retained, total, timed_out = [], 0, 0, False
    try:
        while selector.get_map():
            _remember_descendants(owned)
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                timed_out = True
                break
            for key, _ in selector.select(min(remaining, .1)):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                total += len(chunk)
                keep = chunk[:max(0, max_output_bytes - retained)]
                if keep:
                    chunks.append(keep)
                retained += len(keep)
        if not timed_out:
            _remember_descendants(owned)
            try:
                process.wait(timeout=max(.001, timeout - (time.monotonic() - started)))
            except subprocess.TimeoutExpired:
                timed_out = True
    finally:
        # Also terminate background children when the leader exits successfully.
        stop_group()
        process.wait()
        selector.close()
        process.stdout.close()
    return CommandResult(process.returncode, b"".join(chunks).decode("utf-8", errors="replace"),
                         round(time.monotonic() - started, 6), timed_out, total > retained)


if __name__ == "__main__":
    if len(sys.argv) < 4 or sys.argv[1] != "--limit-wrapper":
        raise SystemExit("Internal native execution wrapper")
    import resource
    for name, value in json.loads(sys.argv[2]).items():
        resource.setrlimit(getattr(resource, name), (value, value))
    os.environ[_COMMAND_GROUP_ENV] = str(os.getpgrp())
    os.execvpe(sys.argv[3], sys.argv[3:], os.environ)
