"""Explicit, checksummed native toolchain preparation for historical SWE tasks.

Only prepare_repositories --prepare calls these installers. Source builds are
cached by exact version under /data; measured trials use the prepared tools.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import tarfile
import urllib.request

from local_execution import environment_fingerprint

# Published by https://go.dev/dl/?mode=json&include=all (linux-amd64 archives).
GO_SHA256 = {
    "1.18.10": "5e05400e4c79ef5394424c0eff5b9141cb782da25f64f79d54c98af0a37f8d49",
    "1.19.13": "4643d4c29c55f53fa0349367d7f1bb5ca554ea6ef528c146825b0f8464e2e668",
    "1.20.14": "ff445e48af27f93f66bd949ae060d97991c83e11289009d311f25426258f9c44",
    "1.21.13": "502fc16d5910562461e6a6631fb6377de2322aad7304bf2bcd23500ba9dab4a7",
    "1.22.12": "4fa4f869b0f7fc6bb1eb2660e74657fbf04cdd290b5aef905585c86051b34d43",
    "1.23.8": "45b87381172a58d62c977f27c4683c8681ef36580abecd14fd124d24ca306d3f",
}
# https://www.ruby-lang.org/en/news/2025/03/26/ruby-3-1-7-released/
# https://www.ruby-lang.org/en/news/2025/04/09/ruby-3-3-8-released/
RUBY_SHA256 = {
    "3.1.7": "0556acd69f141ddace03fa5dd8d76e7ea0d8f5232edf012429579bcdaab30e7b",
    "3.3.8": "5ae28a87a59a3e4ad66bc2931d232dbab953d0aa8f6baf3bc4f8f80977c89cab",
}
# https://www.php.net/releases/index.php?json&version=8.3.16
PHP_SHA256 = {"8.3.16": "61441627dca50cf0173e3f054ffe8c4f5db6552555c43cab87a8ecacfd201c7e"}
# The pinned upstream PHP setup uses Composer 2.8.5.
# https://getcomposer.org/download/2.8.5/composer.phar.sha256sum
COMPOSER_VERSION = "2.8.5"
COMPOSER_SHA256 = "9cef18212e222351aeb476b81de7b2a5383f775336474467bf5c7ccfe84ab0cc"


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compiler_environment(prefix):
    """Conda compiler packages need explicit prefixed tools without activation."""
    prefix = Path(prefix)
    patterns = {"CC": ("*-conda-linux-gnu-cc", "*-conda-linux-gnu-gcc"),
                "CXX": ("*-conda-linux-gnu-c++", "*-conda-linux-gnu-g++"),
                "AR": ("*-conda-linux-gnu-ar",), "RANLIB": ("*-conda-linux-gnu-ranlib",),
                "FC": ("*-conda-linux-gnu-gfortran",)}
    values = {}
    for name, globs in patterns.items():
        for glob in globs:
            matches = sorted((prefix / "bin").glob(glob))
            if matches:
                values[name] = str(matches[0])
                break
    sysroots = sorted(prefix.glob("*-conda-linux-gnu/sysroot"))
    if sysroots:
        values["CONDA_BUILD_SYSROOT"] = str(sysroots[0])
    return values


def download_verified(url, path, expected):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if checksum(path) != expected:
            raise ValueError("Cached native toolchain archive checksum mismatch: " + str(path))
        return path
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=90) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        if checksum(temporary) != expected:
            raise ValueError("Native toolchain download checksum mismatch: " + url)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def prepare_toolchain(kind, version, root, *, conda, run):
    from local_repository_preparation import write_json
    root = Path(root).resolve()
    parent = root / "toolchains" / (kind + "-" + version)
    parent.mkdir(parents=True, exist_ok=True)
    prefix = parent / "prefix"
    manifest = parent / "manifest.json"
    expected = {"go": GO_SHA256, "ruby": RUBY_SHA256, "php": PHP_SHA256}[kind][version]
    installer_hash = checksum(Path(__file__))
    with (parent / "preparation.lock").open("a") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX)
        if manifest.exists():
            value = json.loads(manifest.read_text())
            if value.get("installer_sha256") != installer_hash or value.get("archive_sha256") != expected:
                raise ValueError("Shared toolchain recipe changed; archive and prepare again: " + str(parent))
            if environment_fingerprint(prefix) != value["sha256"]:
                raise ValueError("Prepared shared toolchain changed: " + str(prefix))
            return value
        if prefix.exists():
            raise ValueError("Incomplete shared toolchain exists; inspect logs and archive " + str(parent))
        if kind == "go":
            if platform.machine() not in ("x86_64", "amd64"):
                raise ValueError("Pinned Go binaries require Linux x86_64")
            filename = "go" + version + ".linux-amd64.tar.gz"
            url, digest = "https://go.dev/dl/" + filename, GO_SHA256[version]
        elif kind == "ruby":
            filename = "ruby-" + version + ".tar.gz"
            url = "https://cache.ruby-lang.org/pub/ruby/" + ".".join(version.split(".")[:2]) + "/" + filename
            digest = RUBY_SHA256[version]
        elif kind == "php":
            filename = "php-" + version + ".tar.gz"
            url, digest = "https://www.php.net/distributions/" + filename, PHP_SHA256[version]
        else:
            raise ValueError("Unknown native toolchain " + kind)
        archive = download_verified(url, root / "cache" / "native-toolchains" / filename, digest)
        build = parent / "build"; build.mkdir(exist_ok=True)
        with tarfile.open(archive) as stream:
            stream.extractall(build, filter="data")
        commands = []
        if kind == "go":
            prefix.mkdir(); shutil.move(str(build / "go"), str(prefix / "go"))
        else:
            packages = ["c-compiler", "cxx-compiler", "make", "pkg-config", "autoconf", "bison", "re2c",
                        "openssl", "libffi", "zlib", "bzip2", "libyaml", "readline", "ncurses"]
            if kind == "php":
                packages += ["libxml2", "libzip", "libpng", "libjpeg-turbo", "freetype", "icu",
                             "libcurl", "oniguruma", "sqlite", "libiconv", "gettext", "libsodium", "gmp"]
            run([conda, "create", "--yes", "-c", "conda-forge", "--prefix", prefix, *packages])
            env = dict(os.environ)
            env.update(PATH=str(prefix / "bin") + os.pathsep + env.get("PATH", ""),
                       PKG_CONFIG_PATH=str(prefix / "lib/pkgconfig"),
                       CFLAGS="-I" + str(prefix / "include"),
                       LDFLAGS="-L" + str(prefix / "lib") + " -Wl,-rpath," + str(prefix / "lib"))
            compilers = compiler_environment(prefix)
            if not all(name in compilers for name in ("CC", "CXX")):
                raise RuntimeError("Native source build requires the prepared conda C/C++ compiler executables")
            env.update(compilers)
            compiler_versions = {}
            for name in ("CC", "CXX", "AR", "RANLIB"):
                if name in compilers:
                    result = run([compilers[name], "--version"], env=env)
                    compiler_versions[name] = {"path": compilers[name], "version_output": result.output}
            source = build / (kind + "-" + version)
            configure = ["./configure", "--prefix=" + str(prefix)]
            if kind == "ruby":
                configure += ["--disable-install-doc", "--enable-shared", "--with-openssl-dir=" + str(prefix)]
            else:
                configure += ["--disable-cgi", "--enable-mbstring", "--enable-intl", "--enable-bcmath",
                              "--enable-exif", "--enable-gd", "--enable-ftp", "--enable-pcntl", "--with-gmp",
                              "--with-jpeg", "--with-freetype", "--with-zip",
                              "--with-curl", "--with-openssl", "--with-zlib", "--with-bz2", "--with-sodium",
                              "--with-gettext", "--with-pdo-sqlite", "--with-sqlite3"]
            commands = [configure, ["make", "-j", str(min(8, os.cpu_count() or 1))], ["make", "install"]]
            for command in commands:
                run(command, cwd=source, env=env)
            if kind == "php":
                composer = download_verified("https://getcomposer.org/download/" + COMPOSER_VERSION + "/composer.phar",
                    root / "cache" / "native-toolchains" / ("composer-" + COMPOSER_VERSION + ".phar"), COMPOSER_SHA256)
                shutil.copy2(composer, prefix / "bin/composer"); (prefix / "bin/composer").chmod(0o755)
        shutil.rmtree(build)
        value = {"kind": kind, "version": version, "path": str(prefix),
                 "archive_url": url, "archive_sha256": digest, "build_commands": commands,
                 "sha256": environment_fingerprint(prefix), "fingerprint_kind": "metadata_v1",
                 "compiler_identity": compiler_versions if kind != "go" else {}, "installer_sha256": installer_hash,
                 "composer": {"version": COMPOSER_VERSION, "sha256": COMPOSER_SHA256} if kind == "php" else None}
        write_json(manifest, value)
        return value


def prepare_native_tools(recipe, task_prefix, root, *, conda, run):
    records, environment, paths = [], {}, []
    for kind in ("go", "ruby", "php"):
        version = recipe.get("toolchain", {}).get(kind + "_version")
        if not version:
            continue
        record = prepare_toolchain(kind, version, root, conda=conda, run=run)
        records.append(record); prefix = Path(record["path"])
        if kind == "go":
            paths.append(str(prefix / "go/bin"))
            environment.update(GOROOT=str(prefix / "go"), GOTOOLCHAIN="local")
        elif kind == "ruby":
            paths.append(str(prefix / "bin"))
            # Ruby embeds its original installation prefix. Its shared runtime
            # remains immutable; installed gems and bundle state are per trial.
            abi = ".".join(version.split(".")[:2]) + ".0"
            environment.update(GEM_HOME="${RUNTIME}/home/.gem",
                               GEM_PATH="${RUNTIME}/home/.gem:" + str(prefix / "lib/ruby/gems" / abi),
                               BUNDLE_USER_HOME="${RUNTIME}/home/.bundle")
        else:
            paths.append(str(prefix / "bin"))
            environment.update(COMPOSER_HOME="${RUNTIME}/home/.composer", COMPOSER_ALLOW_SUPERUSER="0")
    return {"external_toolchains": records, "environment": environment, "path_entries": paths}
