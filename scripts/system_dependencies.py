#!/usr/bin/env python3
"""Check native system prerequisites; install only with explicit --install."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_repository_preparation import make_plan
from storage import data_root

SUITES = ("swebench_verified", "swebench_multilingual")
BROWSER_PACKAGES = {"google-chrome-stable", "google-chrome", "chromium", "chromium-browser"}
CHROME_URL = "https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb"
BROWSER_NAMES = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")
# Chromium's official packaging reads these defaults before creating APT sources:
# https://github.com/chromium/chromium/blob/main/chrome/installer/linux/common/apt.include
DISABLE_BROWSER_REPOSITORY = '''from pathlib import Path
import re
p=Path('/etc/default/google-chrome')
text=p.read_text() if p.exists() else ''
for key in ('repo_add_once','repo_reenable_on_distupgrade'):
    text=re.sub(r'^\\s*'+key+r'=.*\\n?', '', text, flags=re.M)
    text += '\\n'+key+'="false"\\n'
p.write_text(text)
'''


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def requirements(root):
    """Read all prepared instances, validate their snapshots, and union recipes."""
    root = Path(root)
    packages, sources, task_count = set(), [], 0
    for suite in SUITES:
        directory = root / "datasets" / suite
        manifest_path, sealed = directory / "manifest.json", directory / "sealed.jsonl"
        manifest = json.loads(manifest_path.read_text())
        actual = digest(sealed)
        if actual != manifest.get("sealed_sha256"):
            raise ValueError(f"Prepared sealed dataset checksum mismatch: {suite}")
        rows = [json.loads(line) for line in sealed.read_text().splitlines() if line.strip()]
        if len(rows) != manifest.get("count") or not rows:
            raise ValueError(f"Prepared dataset count mismatch or empty dataset: {suite}")
        if len({row["instance_id"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate prepared instances: {suite}")
        for row in rows:
            packages.update(make_plan(row, suite, root)["system_packages"])
        task_count += len(rows)
        sources.append({"dataset": suite, "count": len(rows), "sealed_sha256": actual,
                        "manifest_sha256": digest(manifest_path)})
    if any(not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9+.-]*(?::[a-z0-9]+)?", name)
           for name in packages):
        raise ValueError("Invalid native system package name in recipe")
    return {"packages": sorted(packages - BROWSER_PACKAGES),
            "browser_required": bool(packages & BROWSER_PACKAGES),
            "locale_required": "locales" in packages, "task_count": task_count, "datasets": sources}


def probe(command, *, timeout=15):
    try:
        result = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=timeout)
        return {"returncode": result.returncode, "output": result.stdout.strip(), "error_type": None}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "output": "", "error_type": type(exc).__name__}


def browser_state(required):
    candidates = []
    configured = os.environ.get("CHROME_BIN")
    if configured:
        candidates.append((configured, "CHROME_BIN"))
    candidates.extend((name, name) for name in BROWSER_NAMES)
    seen = set()
    for candidate, source in candidates:
        path = shutil.which(candidate)
        if not path or path in seen:
            continue
        seen.add(path)
        result = probe([path, "--version"], timeout=10)
        if result["returncode"] == 0 and re.search(r"\b(?:Chrome|Chromium)\b", result["output"], re.I):
            return {"required": required, "available": True, "path": str(Path(path).resolve()),
                    "version": result["output"][:500], "source": source}
    return {"required": required, "available": False, "path": None, "version": None, "source": None}


def inspect_system(root):
    """Return installed package/browser/locale evidence without writes or SMTP."""
    required = requirements(root)
    installed, missing = [], []
    for package in required["packages"]:
        result = probe(["dpkg-query", "-W", "-f=${Status}\t${Version}", package])
        status, _, version = result["output"].partition("\t")
        ready = result["returncode"] == 0 and status == "install ok installed" and bool(version)
        installed.append({"package": package, "installed": ready, "version": version if ready else None,
                          "error_type": result["error_type"]})
        if not ready:
            missing.append(package)
    browser = browser_state(required["browser_required"])
    result = probe(["locale", "-a"])
    names = [line.strip() for line in result["output"].splitlines()]
    english = next((name for name in names if name.lower().replace("-", "") == "en_us.utf8"), None)
    locale = {"required": required["locale_required"], "available": result["returncode"] == 0 and bool(english),
              "name": english, "error_type": result["error_type"]}
    ready = not missing and (not browser["required"] or browser["available"]) and (not locale["required"] or locale["available"])
    return {"schema_version": 1, "ready": ready, "requirements": required, "packages": installed,
            "missing_packages": missing, "browser": browser, "locale": locale}


def validate_root(value):
    root = Path(value).expanduser().resolve()
    if root == Path("/data") or not root.is_relative_to("/data"):
        raise ValueError("System preparation data root must be below /data")
    return root


def download_chrome(root):
    """Fetch the official standalone package and record the actual downloaded SHA."""
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "amd64"):
        raise ValueError("The supplied Chrome package supports Linux amd64 only; supply CHROME_BIN for this host")
    directory = Path(root) / "cache/system"
    archive, metadata = directory / "google-chrome-stable_current_amd64.deb", directory / "chrome-download.json"
    directory.mkdir(parents=True, exist_ok=True)
    if archive.exists() and metadata.exists():
        record = json.loads(metadata.read_text())
        if record.get("url") != CHROME_URL or record.get("sha256") != digest(archive):
            raise ValueError("Cached browser package provenance/checksum mismatch")
        return archive, record
    temporary = archive.with_suffix(".deb.part")
    try:
        with urllib.request.urlopen(CHROME_URL, timeout=90) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    archive.chmod(0o644)
    record = {"url": CHROME_URL, "sha256": digest(archive), "bytes": archive.stat().st_size,
              "downloaded_unix": time.time()}
    write_json(metadata, record)
    return archive, record


def install_system(root, initial, log):
    """Explicit terminal-only installation. Commands keep sudo authentication on the terminal."""
    def run(command):
        with log.open("a") as stream:
            stream.write("$ " + shlex.join(command) + "\n")
        # Preserve terminal stdin/stdout for sudo. Never collect a password.
        try:
            result = subprocess.run(command, check=True)
        except BaseException as exc:
            with log.open("a") as stream:
                stream.write("error_type=" + type(exc).__name__ + " exit_code="
                             + str(getattr(exc, "returncode", None)) + "\n")
            raise
        with log.open("a") as stream:
            stream.write("exit_code=" + str(result.returncode) + "\n")
    missing = initial["missing_packages"]
    need_browser = initial["browser"]["required"] and not initial["browser"]["available"]
    if missing or need_browser:
        run(["sudo", "apt-get", "update"])
    if missing:
        run(["sudo", "apt-get", "install", "-y", "--", *missing])
    if need_browser:
        archive, _ = download_chrome(root)
        package = probe(["dpkg-deb", "--field", str(archive), "Package", "Architecture"])
        fields = dict(line.split(":", 1) for line in package["output"].splitlines() if ":" in line)
        if (package["returncode"] != 0 or fields.get("Package", "").strip() != "google-chrome-stable"
                or fields.get("Architecture", "").strip() != "amd64"):
            raise ValueError("Downloaded file is not the expected Chrome amd64 package")
        run(["sudo", "python3", "-c", DISABLE_BROWSER_REPOSITORY])
        run(["sudo", "apt-get", "install", "-y", "--", str(archive)])
    if initial["locale"]["required"] and not initial["locale"]["available"]:
        run(["sudo", "locale-gen", "en_US.UTF-8"])
    return inspect_system(root)


class PreparationInterrupted(Exception):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(signal.Signals(signum).name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=data_root())
    parser.add_argument("--output", type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="Read-only prerequisite check (default)")
    action.add_argument("--install", action="store_true", help="Explicit terminal sudo installation")
    action.add_argument("--print", dest="print_plan", action="store_true", help="Print required system packages only")
    args = parser.parse_args(argv)
    root = validate_root(args.data_root)
    output = args.output.expanduser().resolve() if args.output else root / "envs/locks/system.json"
    if not output.is_relative_to(root):
        raise ValueError("System inventory output must remain below the data root")
    invocation = uuid.uuid4().hex
    directory = root / "results/system-preparation" / invocation
    previous = {}
    if args.install:
        directory.mkdir(parents=True, exist_ok=True); directory.chmod(0o700)
        def interrupted(signum, frame):
            raise PreparationInterrupted(signum)
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous[signum] = signal.signal(signum, interrupted)
    try:
        if args.print_plan:
            print(json.dumps(requirements(root), indent=2, sort_keys=True))
            return 0
        state = inspect_system(root)
        if args.install and not state["ready"]:
            state = install_system(root, state, directory / "installation.log")
        if not state["ready"]:
            print(json.dumps(state, indent=2, sort_keys=True))
            print("System prerequisites are missing. Run in a terminal: bash scripts/setup_system_dependencies.sh "
                  + "--data-root " + shlex.quote(str(root)), file=sys.stderr)
            if args.install:
                raise RuntimeError("System prerequisites remain unavailable after installation")
            return 1
        download = root / "cache/system/chrome-download.json"
        if download.exists():
            state["browser_download"] = json.loads(download.read_text())
        # A successful explicit output request or installation publishes actual
        # measured inventory; an ordinary check performs no filesystem writes.
        if args.output is not None or args.install:
            write_json(output, state)
        print(json.dumps(state, indent=2, sort_keys=True))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        signum = getattr(exc, "signum", signal.SIGINT if isinstance(exc, KeyboardInterrupt) else None)
        code = 128 + signum if signum else getattr(exc, "returncode", 1) or 1
        event = {"event_id": "system-preparation:" + invocation, "invocation_id": invocation,
                 "stage": "system_dependencies", "exit_code": code, "error_type": type(exc).__name__,
                 "output": str(directory), "log_path": str(directory / "failure.json")}
        if args.install:
            write_json(directory / "failure.json", event)
            from experiment_notifications import notify
            receipt = notify(directory, event)
            print(json.dumps({"failure_stage": "system_dependencies", "error_type": type(exc).__name__,
                              "smtp_accepted": receipt.get("smtp_accepted", False),
                              "email_pending": receipt.get("pending", True)}), file=sys.stderr)
        else:
            print("System prerequisite check failed: " + type(exc).__name__ + ". Run: "
                  "bash scripts/setup_system_dependencies.sh --data-root " + shlex.quote(str(root)), file=sys.stderr)
        return code
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
