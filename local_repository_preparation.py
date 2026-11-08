"""Pinned native repository setup, environment provenance, and grading controls.

Preparation is an explicit user-run operation. It installs isolated prefixes,
then requires the original and reference-patched controls before publishing a
ready lock. Agents never receive reference patches, test lists, or control logs.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

from local_execution import environment_fingerprint, native_environment, run_bounded_command

RECIPES = Path(__file__).with_name("native_repository_recipes.json")
WORKER = Path(__file__).with_name("local_swe_worker.py")
HARNESS_VERSION = "5.0.1"
HARNESS_COMMIT = "87ab1f6ced28f75ba73ca899dc759b019310944a"


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value, private=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    if private:
        temporary.chmod(0o600)
    temporary.replace(path)


def preparation_source_hashes():
    return {name: file_hash(Path(__file__).with_name(name)) for name in
            ("local_repository_preparation.py", "local_swe_worker.py", "local_execution.py", "native_toolchains.py")}


def verify_prepared_environment(task):
    record = asdict(task) if is_dataclass(task) else dict(task)
    prefix, source = Path(record["environment_path"]), Path(record["source_path"])
    preparation = prefix.parent / "preparation.json"
    if not preparation.exists() or file_hash(preparation) != record["preparation_sha256"]:
        raise ValueError("Native preparation provenance is missing or changed")
    metadata = json.loads(preparation.read_text())
    if metadata.get("status") != "ready" or metadata.get("base_commit") != record["base_commit"]:
        raise ValueError("Native environment has not passed preparation controls")
    if metadata.get("preparation_source_sha256") != preparation_source_hashes():
        raise ValueError("Native preparation/adapter code changed; original/reference controls must be rerun")
    plan = metadata["plan"]
    current_recipe = recipe_for(plan["recipe"], plan["dataset"])
    if canonical_hash(current_recipe) != record["recipe_sha256"] or plan["recipe_sha256"] != record["recipe_sha256"]:
        raise ValueError("Native recipe changed; original/reference controls must be rerun")
    if metadata.get("system_state", {}) != snapshot_system_dependencies(plan.get("system_packages", [])):
        raise ValueError("Native system dependencies changed; original/reference controls must be rerun")
    if record.get("environment_fingerprint_kind") != "metadata_v1":
        raise ValueError("Unknown native environment fingerprint policy")
    if environment_fingerprint(prefix) != record["environment_sha256"]:
        raise ValueError("Prepared environment changed; prepare a new experiment environment")
    if environment_fingerprint(source) != record["source_sha256"]:
        raise ValueError("Prepared source changed; refuse a comparison with different starting files")
    config = json.loads((prefix / "native-environment.json").read_text()) if (prefix / "native-environment.json").exists() else {}
    for toolchain in config.get("external_toolchains", []):
        if environment_fingerprint(toolchain["path"]) != toolchain["sha256"]:
            raise ValueError("Prepared shared native toolchain changed: " + toolchain["path"])
    identity = canonical_hash({key: record[key] for key in
                               ("environment_sha256", "source_sha256", "recipe_sha256")})
    if identity != record["environment_id"]:
        raise ValueError("Native environment identity mismatch")
    return record


def recipe_for(row, suite, catalog=None):
    catalog = catalog or json.loads(RECIPES.read_text())
    key = suite + "/" + row["repo"] + "@" + row["version"]
    if key not in catalog["recipes"]:
        raise ValueError("No pinned native installation recipe for " + key)
    return catalog["recipes"][key]


def conda_packages(recipe):
    """Use versioned language toolchains in each task's isolated prefix."""
    result = ["python=" + recipe.get("python", "3.11"), "pip", "git", "make", "pkg-config"]
    toolchain = recipe.get("toolchain", {})
    # Historical Go/Ruby/PHP patch versions are absent from conda-forge. Their
    # checksummed upstream builds are prepared separately and shared immutably.
    names = {"node_version": "nodejs", "java_version": "openjdk", "maven_version": "maven", "rust_version": "rust"}
    for key, name in names.items():
        if key in toolchain:
            result.append(name + "=" + toolchain[key])
    if "java_version" in toolchain:
        result += ["ant", "unzip", "zip"]
    if "node_version" in toolchain:
        result += ["yarn<2", "jq"]
        install = recipe.get("install", [])
        if any("pnpm" in line for line in ([install] if isinstance(install, str) else install)):
            result.append("pnpm")
    if "php_version" in toolchain:
        result += ["libxml2", "libzip", "libpng", "icu"]
    if "ruby_version" in toolchain:
        result += ["libffi", "openssl", "libxml2", "libxslt", "jq"]
    if "python" not in recipe:
        result += ["c-compiler", "cxx-compiler", "cmake", "ninja", "autoconf", "automake", "libtool", "curl", "tcl", "bison", "flex"]
    packages = recipe.get("packages", "")
    if packages and packages not in ("requirements.txt", "environment.yml"):
        result.extend(shlex.split(packages))
    return list(dict.fromkeys(result))


APT_CONDA = {"imagemagick": "imagemagick", "ffmpeg": "ffmpeg", "libqhull-dev": "qhull",
             "libfreetype6-dev": "freetype", "pkg-config": "pkg-config", "graphviz": "graphviz",
             "libffi-dev": "libffi", "libssl-dev": "openssl", "zlib1g-dev": "zlib",
             "libxml2-dev": "libxml2", "libxslt1-dev": "libxslt",
             "build-essential": "cxx-compiler", "autoconf": "autoconf", "automake": "automake",
             "libtool": "libtool", "bison": "bison", "flex": "flex", "tcl": "tcl",
             "tcl8.6": "tcl", "libonig-dev": "oniguruma", "gettext": "gettext"}


def setup_commands(recipe):
    """Separate native package requirements from repository-local shell edits."""
    extra, required_system, commands = [], [], []
    apt = list(recipe.get("apt-pkgs", []))
    raw = recipe.get("pre_install", [])
    raw = [raw] if isinstance(raw, str) else list(raw)
    if recipe.get("repo_install"):
        raw.insert(0, recipe["repo_install"])
    for line in raw:
        if "apt-get" in line:
            if "tzdata=" in line:
                # Use per-prefix timezone data instead of changing the host OS.
                extra.append("tzdata=2022a")
                continue
            part = line.rsplit("install", 1)[-1]
            apt.extend(x for x in shlex.split(part) if not x.startswith("-") and x not in ("&&", "true"))
        elif "/etc/locale.gen" in line or line.startswith("locale-gen"):
            commands.append("locale -a | tr '[:upper:]' '[:lower:]' | grep -qx en_us.utf8")
            required_system.append("locales")
        else:
            commands.append(line)
    for package in apt:
        if package == "locales":
            commands.append("locale -a | tr '[:upper:]' '[:lower:]' | grep -qx en_us.utf8")
            required_system.append("locales")
        elif package in APT_CONDA:
            extra.append(APT_CONDA[package])
        else:
            required_system.append(package)
    for key in ("install", "build", "post_build_cleanup"):
        value = recipe.get(key, [])
        commands.extend([value] if isinstance(value, str) else value)
    commands = [re.sub(r"(?<![A-Za-z0-9_/])mvnd(?=\s|$)", "mvn", command) for command in commands]
    if recipe.get("repo") == "django/django":
        required_system.append("locales")
    if recipe.get("repo") == "preactjs/preact":
        required_system.append("google-chrome-stable")
    return list(dict.fromkeys(extra)), list(dict.fromkeys(required_system)), commands


def validate_controls(row, controls):
    """Reject native drift before any model results can be graded."""
    from local_swe_worker import test_ids
    gold, baseline = controls["reference_patch"], controls["original_base"]
    if (gold.get("resolved") is not True or gold.get("status") != "native_official_parser_report"
            or not gold.get("parser_output_valid")):
        raise RuntimeError("Reference patch did not pass native controls; environment cannot grade candidates")
    expected_fail = test_ids(row["FAIL_TO_PASS"])
    expected_pass = test_ids(row["PASS_TO_PASS"])
    observed = baseline.get("observed_expected_tests", {})
    if baseline.get("status") != "native_official_parser_report" or not baseline.get("parser_output_valid"):
        raise RuntimeError("Original base did not execute control tests")
    if row.get("eval_type", "pass_and_fail") == "pass_and_fail" and any(
            observed.get(case) is None for case in expected_fail + expected_pass):
        raise RuntimeError("Original base did not execute every expected control test")
    baseline_tests = baseline["official_report"]["tests_status"]
    if baseline_tests["PASS_TO_PASS"]["failure"]:
        raise RuntimeError("Original base regressed PASS_TO_PASS control tests; repair native dependencies")
    if any(observed.get(case) not in ("FAILED", "ERROR") for case in expected_fail):
        raise RuntimeError("Original base did not reproduce every FAIL_TO_PASS failure; native control is invalid")


def clean_requirements(text, replacements):
    for old, new in replacements:
        text = re.sub(r"^" + re.escape(old) + r"([<>=!~]=?.*|$)\n?",
                      (new + "\n") if new else "", text, flags=re.MULTILINE)
    return text


def requirements_from_git(source, commit, candidates, replacements, runner):
    def read(path):
        result = runner(["git", "show", commit + ":" + path], cwd=source,
                        timeout=60, max_output_bytes=2 * 1024 * 1024)
        if result.returncode or result.truncated:
            return None
        return result.output
    def expand(path, visited):
        if path in visited:
            raise ValueError("Recursive requirement include")
        text = read(path)
        if text is None:
            raise ValueError("Missing requirement file at pinned setup commit: " + path)
        output = []
        for line in text.splitlines():
            if line.strip().startswith(("-e .", "#", ".[test")):
                continue
            if line.strip().startswith("-r"):
                relative = str(Path(path).parent / line.strip()[2:].strip())
                output.append(expand(relative, visited | {path}))
            else:
                output.append(line)
        return "\n".join(output) + "\n"
    for path in candidates:
        if read(path) is not None:
            return clean_requirements(expand(path, set()), replacements)
    raise ValueError("No requirements file at pinned setup commit; tried " + repr(candidates))


def fetch_text(url):
    with urllib.request.urlopen(url, timeout=90) as response:
        return response.read().decode()


def find_chrome():
    candidate = os.environ.get("CHROME_BIN")
    if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
        return str(Path(candidate).resolve())
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        candidate = shutil.which(name)
        if candidate:
            return str(Path(candidate).resolve())
    return None


def snapshot_system_dependencies(packages, *, command_runner=run_bounded_command):
    """Record actual host prerequisites used by a task, including browser choice."""
    result = {}
    for package in sorted(set(packages)):
        if package == "google-chrome-stable":
            browser = find_chrome()
            if browser is None:
                raise RuntimeError("Native browser tests require Chrome/Chromium; set CHROME_BIN")
            probe = command_runner([browser, "--version"], timeout=30, max_output_bytes=10000)
            if probe.returncode or probe.timed_out or probe.truncated:
                raise RuntimeError("Configured native browser did not report its version")
            result[package] = {"path": browser, "version": probe.output.strip(),
                               "launcher_sha256": file_hash(browser)}
            continue
        probe = command_runner(["dpkg-query", "-W", "-f=${Status}\t${Version}", package],
                               timeout=30, max_output_bytes=10000)
        if probe.returncode or probe.timed_out or "install ok installed" not in probe.output:
            raise RuntimeError("Native system dependency missing. Administrator command: sudo apt-get install " + shlex.quote(package))
        result[package] = probe.output.strip()
        if package == "locales":
            locale = command_runner(["locale", "-a"], timeout=30, max_output_bytes=10000)
            if locale.returncode or "en_us.utf8" not in locale.output.lower().splitlines():
                raise RuntimeError("en_US.UTF-8 locale is missing. Administrator command: sudo locale-gen en_US.UTF-8")
            result["en_US.UTF-8"] = True
    return result


def make_plan(row, suite, data_root, catalog=None):
    recipe = recipe_for(row, suite, catalog)
    extra, system, commands = setup_commands(recipe)
    root = Path(data_root).resolve()
    instance = row["instance_id"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", instance) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", row["repo"]):
        raise ValueError("Unsafe repository identity")
    target = root / "repositories" / suite / instance
    return {"schema_version": 1, "backend": "native", "dataset": suite, "instance_id": instance,
            "repo": row["repo"], "base_commit": row["base_commit"],
            "source_path": str(target / "source"),
            "environment_path": str(root / "envs" / "repositories" / suite / instance / "prefix"),
            "recipe_sha256": canonical_hash(recipe), "recipe": recipe,
            "conda_packages": list(dict.fromkeys(conda_packages(recipe) + extra)),
            "system_packages": system, "setup_commands": commands,
            "controls": ["original_base", "reference_patch"],
            "grading_policy": "native environment; pinned official parser and F2P/P2P rules"}


def prepare_one(row, suite, root, *, harness_python, conda, timeout=7200, test_timeout=1800,
                command_runner=run_bounded_command, fetcher=fetch_text):
    plan = make_plan(row, suite, root)
    source, prefix = Path(plan["source_path"]), Path(plan["environment_path"])
    state_dir = prefix.parent; state_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(root) / "results" / "repository-preparation" / suite / row["instance_id"]
    log_dir.mkdir(parents=True, exist_ok=True); log_dir.chmod(0o700)
    write_json(log_dir / "plan.json", plan)
    log_path = log_dir / "preparation.log"
    started = time.monotonic()
    def run(argv, *, cwd=None, env=None, input_text=None):
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Native preparation exceeded its per-task deadline")
        result = command_runner([str(x) for x in argv], cwd=cwd, env=env, input_text=input_text,
                                timeout=remaining, max_output_bytes=4 * 1024 * 1024)
        with log_path.open("a") as stream:
            stream.write("\n$ " + shlex.join(str(x) for x in argv) + "\n" + result.output)
        if result.returncode or result.timed_out or result.truncated:
            raise RuntimeError("Native setup command failed; inspect " + str(log_path))
        return result
    try:
        if source.exists() or prefix.exists():
            raise ValueError("Incomplete native preparation directories exist. Archive/remove only this task's "
                             "source and prefix after inspecting its log, then rerun: " + str(source) + " ; " + str(prefix))
        if plan["system_packages"]:
            missing = []
            for package in plan["system_packages"]:
                if package == "google-chrome-stable" and find_chrome():
                    continue
                result = command_runner(["dpkg-query", "-W", "-f=${Status}", package],
                                        timeout=30, max_output_bytes=2048)
                if result.returncode or "install ok installed" not in result.output:
                    missing.append(package)
            if missing:
                raise RuntimeError("Native system dependencies missing. Administrator command: sudo apt-get install " + shlex.join(missing))
        system_state = snapshot_system_dependencies(plan["system_packages"], command_runner=command_runner)
        mirror = Path(root) / "repositories" / "mirrors" / (row["repo"].replace("/", "__") + ".git")
        mirror.parent.mkdir(parents=True, exist_ok=True)
        if not mirror.exists():
            run(["git", "clone", "--mirror", "https://github.com/" + row["repo"] + ".git", mirror])
        run(["git", "--git-dir", mirror, "cat-file", "-e", row["base_commit"] + "^{commit}"])
        source.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--shared", "--no-checkout", mirror, source])
        run(["git", "-c", "core.hooksPath=/dev/null", "checkout", "--detach", row["base_commit"]], cwd=source)
        run(["git", "config", "core.hooksPath", "/dev/null"], cwd=source)
        recipe = plan["recipe"]
        upstream = "https://raw.githubusercontent.com/SWE-bench/" + recipe["upstream_repo"] + "/" + recipe["upstream_revision"]
        env_file = None
        if "python" in recipe:
            try:
                text = fetcher(upstream + "/tasks/" + row["instance_id"] + "/environment.yml")
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    raise
            else:
                text = re.sub(r"^(name|prefix):.*\n?", "", text, flags=re.MULTILINE)
                env_file = log_dir / "environment.yml"; env_file.write_text(text)
        if env_file:
            run([conda, "env", "create", "--yes", "--prefix", prefix, "--file", env_file])
            extra, _, _ = setup_commands(recipe)
            if extra:
                run([conda, "install", "--yes", "-c", "conda-forge", "--prefix", prefix, *extra])
        else:
            run([conda, "create", "--yes", "-c", "conda-forge", "--prefix", prefix, *plan["conda_packages"]])
        environment = {"CONDA_PREFIX": "${PREFIX}", "VIRTUAL_ENV": "${PREFIX}",
                       "CARGO_HOME": "${RUNTIME}/home/.cargo", "GOPATH": "${RUNTIME}/home/go",
                       "GRADLE_USER_HOME": "${RUNTIME}/home/.gradle",
                       "npm_config_cache": "${RUNTIME}/home/.npm", "npm_config_prefix": "${PREFIX}",
                       "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        if recipe.get("repo") == "matplotlib/matplotlib":
            environment.update(TZDIR="${PREFIX}/share/zoneinfo", PYTHONTZPATH="${PREFIX}/share/zoneinfo")
        if recipe.get("repo") == "preactjs/preact":
            chrome = find_chrome()
            if chrome is None:
                raise RuntimeError("Native browser tests require Chrome/Chromium; set CHROME_BIN to its executable")
            environment.update(CHROME_BIN=chrome, CHROME_PATH=chrome)
        if "java_version" in recipe.get("toolchain", {}):
            environment["JAVA_HOME"] = "${PREFIX}/lib/jvm"
        from native_toolchains import prepare_native_tools, compiler_environment
        native_config = prepare_native_tools(recipe, prefix, root, conda=conda, run=run)
        environment.update(native_config["environment"])
        environment.update({key: value.replace(str(prefix), "${PREFIX}")
                            for key, value in compiler_environment(prefix).items()})
        native_config["environment"] = environment
        write_json(prefix / "native-environment.json", native_config)
        runtime = log_dir / "setup-runtime"
        env = native_environment(prefix, source, runtime)
        # Resolve conda's actual Java layout; distribution variants are recorded.
        if "JAVA_HOME" in environment:
            javac = prefix / "bin" / "javac"
            if javac.exists():
                java_home = javac.resolve().parent.parent
                environment["JAVA_HOME"] = str(java_home).replace(str(prefix), "${PREFIX}")
                write_json(prefix / "native-environment.json", native_config)
                env = native_environment(prefix, source, runtime)
        if not env_file and recipe.get("packages") == "requirements.txt":
            requirements = requirements_from_git(source, row.get("environment_setup_commit") or row["base_commit"],
                recipe["requirements_paths"], recipe.get("requirements_replacements", []), command_runner)
            path = log_dir / "requirements.txt"; path.write_text(requirements)
            run([prefix / "bin/python", "-m", "pip", "install", "-r", path], cwd=source, env=env)
        elif not env_file and recipe.get("packages") == "environment.yml":
            commit = row.get("environment_setup_commit") or row["base_commit"]
            for path in recipe["environment_paths"]:
                result = command_runner(["git", "show", commit + ":" + path], cwd=source,
                                        timeout=60, max_output_bytes=2 * 1024 * 1024)
                if result.returncode == 0:
                    text = re.sub(r"^(name|prefix):.*\n?", "", result.output, flags=re.MULTILINE)
                    env_path = log_dir / "environment.yml"; env_path.write_text(text)
                    run([conda, "env", "update", "--prefix", prefix, "--file", env_path])
                    break
            else:
                raise ValueError("Missing environment.yml at pinned environment setup commit")
        if not env_file and recipe.get("pip_packages"):
            run([prefix / "bin/python", "-m", "pip", "install", *recipe["pip_packages"]], cwd=source, env=env)
        fixture_hashes = {}
        for key, target in (("cargo_lock", "Cargo.lock"), ("composer_json", "composer.json"), ("composer_lock", "composer.lock")):
            if recipe.get(key):
                text = fetcher(upstream + "/dockerfile_gen/fixtures/" + recipe[key])
                (source / target).write_text(text)
                fixture_hashes[target] = hashlib.sha256(text.encode()).hexdigest()
        setup = "set -euo pipefail\n" + "\n".join(plan["setup_commands"]) + "\n"
        # Preparation commands must never install to host system paths.
        if re.search(r"\b(sudo|apt-get|apt)\b|/etc/locale.gen|/usr/local/", setup):
            raise ValueError("Native recipe contains an unmapped host installation command; inspect plan.json")
        script = log_dir / "setup.sh"; script.write_text(setup)
        run(["bash", script], cwd=source, env=env)
        native_home = prefix / "native-home"
        shutil.copytree(runtime / "home", native_home, symlinks=True)
        run([conda, "list", "--explicit", "--prefix", prefix])
        resolved = run([prefix / "bin/python", "-m", "pip", "freeze", "--all"], env=env)
        (log_dir / "pip-freeze.txt").write_text(resolved.output)
        prepared = {key: plan[key] for key in ("source_path", "environment_path", "base_commit", "recipe_sha256")}
        prepared.update(python_path=str(prefix / "bin/python"), environment_fingerprint_kind="metadata_v1",
                        environment_sha256=environment_fingerprint(prefix), source_sha256=environment_fingerprint(source))
        prepared["environment_id"] = canonical_hash({key: prepared[key] for key in ("environment_sha256", "source_sha256", "recipe_sha256")})
        controls = {}
        for name, patch in (("original_base", ""), ("reference_patch", row["patch"])):
            request = {"task": row, "prepared": prepared, "patch": patch,
                       "timeout": test_timeout, "verify_prepared": False}
            request_path = log_dir / (name + ".request.private.json")
            write_json(request_path, request, private=True)
            case_dir = log_dir / name
            run([harness_python, WORKER, "--request", request_path, "--output", case_dir])
            controls[name] = json.loads((case_dir / "report.json").read_text())
        validate_controls(row, controls)
        # Controls run in copied environments and may not alter prepared inputs.
        if prepared["environment_sha256"] != environment_fingerprint(prefix) or prepared["source_sha256"] != environment_fingerprint(source):
            raise RuntimeError("Native control modified the prepared environment/source")
        metadata = {"status": "ready", "base_commit": row["base_commit"], "plan": plan,
                    "preparation_source_sha256": preparation_source_hashes(),
                    "system_state": system_state,
                    "environment_id": prepared["environment_id"], "fixture_sha256": fixture_hashes,
                    "controls": {key: {"resolved": value["resolved"], "report_sha256": file_hash(log_dir / key / "report.json")}
                                 for key, value in controls.items()},
                    "harness_version": HARNESS_VERSION, "harness_commit": HARNESS_COMMIT,
                    "wall_s": time.monotonic() - started,
                    "environment_fingerprint_kind": "metadata_v1",
                    "limitations": "Native toolchain/environment adaptation; not official environment equivalence."}
        write_json(state_dir / "preparation.json", metadata)
        prepared["preparation_sha256"] = file_hash(state_dir / "preparation.json")
        verify_prepared_environment(prepared)
        return prepared
    except Exception as exc:
        write_json(log_dir / "failure.json", {"status": "preparation_failed", "error": type(exc).__name__ + ": " + str(exc),
                   "instance_id": row["instance_id"], "wall_s": time.monotonic() - started})
        raise
