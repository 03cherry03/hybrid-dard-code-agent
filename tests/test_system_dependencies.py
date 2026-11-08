"""System preparation tests simulate all probes, installs, downloads and mail."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import system_dependencies as system


class SystemDependencyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for suite in system.SUITES:
            directory = self.root / "datasets" / suite
            directory.mkdir(parents=True)
            sealed = directory / "sealed.jsonl"
            sealed.write_text(json.dumps({"instance_id": suite + "-1", "patch": "SEALED_REFERENCE"}) + "\n")
            (directory / "manifest.json").write_text(json.dumps({"count": 1,
                "sealed_sha256": hashlib.sha256(sealed.read_bytes()).hexdigest()}))
        self.enterContext(patch.object(system, "validate_root", return_value=self.root))
        self.mail = self.enterContext(patch("experiment_notifications.notify",
            return_value={"smtp_accepted": True, "pending": False}))
        self.plan = self.enterContext(patch.object(system, "make_plan", return_value={
            "system_packages": ["dvipng", "locales", "google-chrome-stable"]}))

    @staticmethod
    def state(ready=False):
        return {"ready": ready, "requirements": {}, "packages": [],
                "missing_packages": [] if ready else ["dvipng", "locales"],
                "browser": {"required": True, "available": ready},
                "locale": {"required": True, "available": ready}}

    def quiet_main(self, args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return system.main(["--data-root", str(self.root), *args])

    def test_recipe_union_verifies_full_prepared_snapshots_without_exposing_sealed_data(self):
        required = system.requirements(self.root)
        self.assertEqual(required["packages"], ["dvipng", "locales"])
        self.assertEqual(required["task_count"], 2)
        self.assertTrue(required["browser_required"])
        self.assertTrue(required["locale_required"])
        self.assertEqual(self.plan.call_count, 2)
        self.assertNotIn("SEALED_REFERENCE", json.dumps(required))
        sealed = self.root / "datasets" / system.SUITES[0] / "sealed.jsonl"
        sealed.write_text(sealed.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            system.requirements(self.root)

    def test_read_only_checker_accepts_chromium_alternative_and_records_actual_versions(self):
        def command(argv, **kwargs):
            self.assertGreater(kwargs["timeout"], 0)
            if argv[0] == "dpkg-query":
                return SimpleNamespace(returncode=0, stdout="install ok installed\t1.fixture", stderr="")
            if argv[0] == "locale":
                return SimpleNamespace(returncode=0, stdout="C\nen_US.utf8\n", stderr="")
            self.assertEqual(argv, ["/usr/bin/chromium", "--version"])
            return SimpleNamespace(returncode=0, stdout="Chromium 130.fixture", stderr="")
        before = sorted(str(path) for path in self.root.rglob("*"))
        with patch.object(system.subprocess, "run", side_effect=command), \
             patch.object(system.shutil, "which", side_effect=lambda name: "/usr/bin/chromium" if name == "chromium" else None), \
             patch.object(system.urllib.request, "urlopen") as network:
            state = system.inspect_system(self.root)
        self.assertTrue(state["ready"])
        self.assertEqual(state["browser"]["source"], "chromium")
        self.assertEqual(state["packages"][0]["version"], "1.fixture")
        self.assertEqual(before, sorted(str(path) for path in self.root.rglob("*")))
        network.assert_not_called()
        self.mail.assert_not_called()

    def test_probe_timeout_is_missing_dependency_not_an_unbounded_browser_start(self):
        with patch.object(system.subprocess, "run", side_effect=subprocess.TimeoutExpired("browser", 10)), \
             patch.object(system.shutil, "which", return_value="/browser"):
            state = system.browser_state(True)
        self.assertFalse(state["available"])

    def test_check_failure_never_installs_writes_inventory_or_sends_mail(self):
        output = self.root / "envs/locks/system.json"
        with patch.object(system, "inspect_system", return_value=self.state()), \
             patch.object(system, "install_system") as install:
            result = self.quiet_main(["--check", "--output", str(output)])
        self.assertEqual(result, 1)
        self.assertFalse(output.exists())
        install.assert_not_called()
        self.mail.assert_not_called()

    def test_only_ready_check_publishes_requested_inventory(self):
        output = self.root / "envs/locks/system.json"
        with patch.object(system, "inspect_system", return_value=self.state(True)):
            self.assertEqual(self.quiet_main(["--check", "--output", str(output)]), 0)
        self.assertTrue(json.loads(output.read_text())["ready"])
        self.mail.assert_not_called()

    def test_explicit_install_uses_missing_packages_local_browser_archive_and_locale(self):
        archive = self.root / "cache/system/chrome.deb"
        log = self.root / "installation.log"
        with patch.object(system.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run, \
             patch.object(system, "download_chrome", return_value=(archive, {"sha256": "fixture"})), \
             patch.object(system, "probe", return_value={"returncode": 0,
                "output": "Package: google-chrome-stable\nArchitecture: amd64"}), \
             patch.object(system, "inspect_system", return_value=self.state(True)):
            self.assertTrue(system.install_system(self.root, self.state(), log)["ready"])
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(commands[0], ["sudo", "apt-get", "update"])
        self.assertIn(["sudo", "apt-get", "install", "-y", "--", "dvipng", "locales"], commands)
        self.assertIn(["sudo", "apt-get", "install", "-y", "--", str(archive)], commands)
        self.assertIn(["sudo", "locale-gen", "en_US.UTF-8"], commands)
        defaults = [command for command in commands if command[:3] == ["sudo", "python3", "-c"]]
        self.assertEqual(len(defaults), 1)
        self.assertIn("repo_add_once", defaults[0][-1])
        self.assertIn('false', defaults[0][-1])
        self.assertTrue(all(call.kwargs == {"check": True} for call in run.call_args_list))

    def test_install_command_failure_records_returncode_without_exception_text(self):
        log = self.root / "installation.log"
        with patch.object(system.subprocess, "run", side_effect=subprocess.CalledProcessError(
                23, "apt-get", stderr="fixture-private-detail")):
            with self.assertRaises(subprocess.CalledProcessError):
                system.install_system(self.root, self.state(), log)
        self.assertIn("error_type=CalledProcessError exit_code=23", log.read_text())
        self.assertNotIn("fixture-private-detail", log.read_text())

    def test_installed_browser_is_never_downloaded_or_replaced(self):
        state = self.state(True)
        with patch.object(system.subprocess, "run") as run, \
             patch.object(system, "download_chrome") as download, \
             patch.object(system, "inspect_system", return_value=state):
            system.install_system(self.root, state, self.root / "installation.log")
        run.assert_not_called()
        download.assert_not_called()

    def test_install_failure_notifies_with_safe_metadata_and_persists_failure(self):
        secret = "fixture-private-error-never-in-mail"
        with patch.object(system, "inspect_system", return_value=self.state()), \
             patch.object(system, "install_system", side_effect=RuntimeError(secret)):
            self.assertEqual(self.quiet_main(["--install"]), 1)
        self.mail.assert_called_once()
        directory, event = self.mail.call_args.args
        self.assertEqual(event["error_type"], "RuntimeError")
        self.assertEqual(event["output"], str(directory))
        self.assertEqual(event["log_path"], str(directory / "failure.json"))
        self.assertNotIn(secret, json.dumps(event))
        self.assertTrue((directory / "failure.json").is_file())
        self.assertFalse((self.root / "envs/locks/system.json").exists())

    def test_check_and_plan_errors_do_not_send_email(self):
        with patch.object(system, "inspect_system", side_effect=ValueError("fixture")):
            self.assertEqual(self.quiet_main(["--check"]), 1)
        with patch.object(system, "requirements", return_value={"packages": ["locales"]}), \
             patch.object(system, "inspect_system") as inspect:
            self.assertEqual(self.quiet_main(["--print"]), 0)
        inspect.assert_not_called()
        self.mail.assert_not_called()


if __name__ == "__main__":
    unittest.main()
