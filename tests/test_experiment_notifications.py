"""SMTP contract tests use local temporary credentials and mocked transports."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import smtplib
import stat
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import experiment_notifications as mail


class NotificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.credential = self.root / "credential"
        self.credential.write_text("fake-test-password\n")
        self.credential.chmod(0o600)
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {"HYBRID_DARD_MAIL_PASSWORD_FILE": str(self.credential)}).start()
        patch.object(mail.time, "sleep").start()
        mail._accepted_without_receipt.clear()
        mail._sensitive_literals.clear()
        self.addCleanup(mail._accepted_without_receipt.clear)
        self.addCleanup(mail._sensitive_literals.clear)
        self.smtp = MagicMock()
        self.smtp.__enter__.return_value = self.smtp
        self.smtp.__exit__.return_value = False
        self.smtp.noop.return_value = (250, b"ok")
        self.smtp.send_message.return_value = {}
        self.transport = patch.object(mail.smtplib, "SMTP_SSL", return_value=self.smtp).start()
        self.event = {"event_id": "campaign-1:invocation-1:failed-setup", "invocation_id": "invocation-1",
                      "stage": "model_setup", "exit_code": 1, "error_type": "RuntimeError",
                      "host": "fixture", "output": "/data/fixture", "log_path": "/data/fixture/setup.log",
                      "command": ["python3", "prepare.py"]}

    def test_metadata_does_not_read_credentials_or_contact_smtp(self):
        with patch.object(Path, "read_text", side_effect=AssertionError("No secret read")):
            status = mail.configuration_status()
        self.assertTrue(status["metadata_ready"])
        self.assertFalse(status["authentication_attempted"])
        self.assertFalse(status["email_sent"])
        self.transport.assert_not_called()

    def test_credential_environment_precedence_and_legacy_default(self):
        with patch.dict(os.environ, {"LERRA_MAIL_PASSWORD_FILE": str(self.root / "legacy")}):
            self.assertEqual(mail.credential_path(), self.credential)
            with patch.dict(os.environ, {"HYBRID_DARD_MAIL_PASSWORD_FILE": ""}):
                self.assertEqual(mail.credential_path(), self.root / "legacy")
        with patch.dict(os.environ, {"HYBRID_DARD_MAIL_PASSWORD_FILE": "", "LERRA_MAIL_PASSWORD_FILE": ""}), \
                patch.object(Path, "home", return_value=self.root):
            self.assertEqual(mail.credential_path(), self.root / ".config/lerra_ft_comparison/gmail_app_password")

    def test_private_regular_owned_nonempty_credential_is_required(self):
        self.credential.chmod(0o644)
        self.assertFalse(mail.configuration_status()["metadata_ready"])
        with self.assertRaisesRegex(mail.NotificationError, "UnsafeCredentialFile"):
            mail.preflight()
        self.transport.assert_not_called()
        self.credential.chmod(0o600)
        self.credential.write_text("   \n")
        with self.assertRaisesRegex(mail.NotificationError, "InvalidCredentialFile"):
            mail.preflight()
        target = self.root / "target"
        self.credential.rename(target)
        self.credential.symlink_to(target)
        self.assertFalse(mail.configuration_status()["metadata_ready"])
        with self.assertRaises(mail.NotificationError):
            mail.preflight()
        self.transport.assert_not_called()

    def test_preflight_authenticates_with_tls_and_noop_but_does_not_send_mail(self):
        result = mail.preflight()
        self.assertTrue(result["authenticated"])
        self.assertFalse(result["email_sent"])
        self.smtp.login.assert_called_once_with(mail.RECIPIENT, "fake-test-password")
        self.smtp.noop.assert_called_once()
        self.smtp.send_message.assert_not_called()
        args, kwargs = self.transport.call_args
        self.assertEqual(args, ("smtp.gmail.com", 465))
        self.assertEqual(kwargs["timeout"], 15)
        self.assertTrue(kwargs["context"].check_hostname)

    def test_preflight_exception_never_contains_server_reply_or_secret(self):
        self.smtp.login.side_effect = smtplib.SMTPAuthenticationError(535, b"fake-test-password secret-reply")
        with self.assertRaises(mail.NotificationError) as caught:
            mail.preflight()
        self.assertEqual(str(caught.exception), "SMTPAuthenticationError")
        self.assertNotIn("fake-test-password", repr(caught.exception))
        self.smtp.send_message.assert_not_called()

    def test_success_persists_private_receipt_and_omits_log_and_secret_content(self):
        event = {**self.event, "log_tail": "SECRET LOG", "environment": {"TOKEN": "ENV_SECRET"},
                 "command": ["run", "--smtp-password", "ARGV_SECRET", "--api-key=KEY_SECRET"]}
        receipt = mail.notify(self.root, event)
        self.assertTrue(receipt["smtp_accepted"])
        self.assertFalse(receipt["pending"])
        self.assertTrue(receipt["receipt_persisted"])
        self.assertEqual(receipt["attempts"], 1)
        saved_path = next((self.root / "notifications").glob("*.json"))
        self.assertEqual(stat.S_IMODE(saved_path.stat().st_mode), 0o600)
        saved = json.loads(saved_path.read_text())
        self.assertTrue(saved["smtp_accepted"])
        message = self.smtp.send_message.call_args.args[0]
        self.assertEqual(message["To"], "leejwian@gmail.com")
        self.assertTrue(message["Subject"].startswith("[Hybrid-DARD failure]"))
        for secret in ("SECRET LOG", "ENV_SECRET", "ARGV_SECRET", "KEY_SECRET", "fake-test-password"):
            self.assertNotIn(secret, str(message))
            self.assertNotIn(secret, saved_path.read_text())

    def test_shell_command_strings_are_not_transmitted(self):
        mail.notify(self.root, {**self.event, "command": "password='DO_NOT_SEND' arbitrary shell"})
        message = str(self.smtp.send_message.call_args.args[0])
        self.assertNotIn("DO_NOT_SEND", message)
        self.assertIn("command omitted", message)

    def test_explicit_delivery_test_is_clearly_labelled_without_claiming_failure(self):
        event = {**self.event, 'event_id': 'explicit-notification-test',
                 'stage': 'notification_test', 'exit_code': 0, 'error_type': 'ExplicitTest',
                 'message': 'untrusted freeform content is still omitted'}
        receipt = mail.notify(self.root, event)
        self.assertTrue(receipt['smtp_accepted'])
        message = self.smtp.send_message.call_args.args[0]
        self.assertEqual(message['Subject'], '[Hybrid-DARD notification test] Delivery check')
        self.assertIn('No experiment failed or was started by this test.', message.get_content())
        self.assertNotIn('untrusted freeform content', message.get_content())
        self.assertNotIn('[Hybrid-DARD failure]', message['Subject'])

    def test_failure_during_test_stage_is_still_labelled_failure(self):
        mail.notify(self.root, {**self.event, 'event_id': 'failed-test-command',
                    'stage': 'notification_test', 'exit_code': 1, 'error_type': 'RuntimeError'})
        message = self.smtp.send_message.call_args.args[0]
        self.assertTrue(message['Subject'].startswith('[Hybrid-DARD failure]'))
        self.assertNotIn('No experiment failed', message.get_content())

    def test_retry_rejects_nonfinite_or_invalid_deadline_budget_without_transport(self):
        for budget in (0, -1, True, None, '60', float('nan'), float('inf')):
            with self.subTest(budget=budget), self.assertRaisesRegex(mail.NotificationError, 'InvalidRetryBudget'):
                mail.flush_pending(self.root, deadline_seconds=budget)
        self.transport.assert_not_called()

    def test_three_failures_stay_pending_and_flush_reuses_message_id(self):
        self.transport.side_effect = OSError("fake-test-password server-secret")
        failed = mail.notify(self.root, self.event)
        self.assertEqual(self.transport.call_count, 3)
        self.assertFalse(failed["smtp_accepted"])
        self.assertTrue(failed["pending"])
        self.assertEqual(failed["error_type"], "OSError")
        self.assertNotIn("server-secret", json.dumps(failed))
        self.transport.side_effect = None
        result = mail.flush_pending(self.root)
        self.assertEqual(result["processed"], 1)
        self.assertEqual(result["smtp_accepted"], 1)
        saved = json.loads(next((self.root / "notifications").glob("*.json")).read_text())
        self.assertEqual(saved["attempts"], 4)
        self.assertEqual(saved["message_id"], failed["message_id"])
        self.assertEqual(self.smtp.send_message.call_args.args[0]["Message-ID"], failed["message_id"])

    def test_accepted_receipt_deduplicates_across_process_memory_reset(self):
        first = mail.notify(self.root, self.event)
        mail._accepted_without_receipt.clear()
        second = mail.notify(self.root, self.event)
        self.assertTrue(first["smtp_accepted"] and second["smtp_accepted"])
        self.smtp.send_message.assert_called_once()

    def test_acceptance_followed_by_smtp_quit_failure_is_not_retried(self):
        self.smtp.__exit__.side_effect = OSError("server-secret")
        receipt = mail.notify(self.root, self.event)
        self.assertTrue(receipt["smtp_accepted"])
        self.assertEqual(receipt["cleanup_error_type"], "OSError")
        self.assertIsNone(receipt["error_type"])
        self.smtp.send_message.assert_called_once()

    def test_unwritable_outbox_still_attempts_smtp(self):
        blocked = self.root / "not-a-directory"
        blocked.write_text("fixture")
        receipt = mail.notify(blocked, self.event)
        self.assertTrue(receipt["smtp_accepted"])
        self.assertFalse(receipt["receipt_persisted"])
        self.assertEqual(receipt["storage_error_type"], "NotADirectoryError")
        self.smtp.send_message.assert_called_once()

    def test_failed_receipt_write_after_acceptance_uses_in_process_dedup(self):
        def no_persist(path, receipt):
            receipt["receipt_persisted"] = False
            return False
        with patch.object(mail, "_persist", side_effect=no_persist):
            first = mail.notify(self.root, self.event)
            second = mail.notify(self.root, self.event)
        self.assertTrue(first["smtp_accepted"] and second["smtp_accepted"])
        self.assertFalse(second["receipt_persisted"])
        self.smtp.send_message.assert_called_once()

    def test_busy_event_lock_does_not_race_a_second_delivery(self):
        @contextmanager
        def locked(path):
            raise mail._OutboxBusy()
            yield
        with patch.object(mail, "_event_lock", locked):
            receipt = mail.notify(self.root, self.event)
        self.assertEqual(receipt["error_type"], "OutboxBusy")
        self.assertTrue(receipt["pending"])
        self.transport.assert_not_called()

    def test_flush_record_budget_and_corrupt_record_are_reported_safely(self):
        self.transport.side_effect = OSError("fake-test-password")
        mail.notify(self.root, self.event)
        mail.notify(self.root, {**self.event, "event_id": "second-failure"})
        self.transport.side_effect = None
        result = mail.flush_pending(self.root, max_records=1)
        self.assertEqual(result["processed"], 1)
        self.assertEqual(result["smtp_accepted"], 1)
        self.assertEqual(result["pending_remaining"], 1)
        self.assertTrue(result["max_records_reached"])
        self.assertTrue(result["remaining_scan_complete"])
        (self.root / "notifications/000-corrupt.json").write_text("fake-test-password not JSON")
        second = mail.flush_pending(self.root)
        self.assertEqual(second["errors"][0]["error_type"], "JSONDecodeError")
        self.assertEqual(second["pending_remaining"], 0)
        self.assertFalse(second["remaining_scan_complete"])
        self.assertNotIn("fake-test-password", json.dumps(second))

    def test_flush_expired_budget_retains_unattempted_pending_count(self):
        self.transport.side_effect = OSError("fixture unavailable")
        mail.notify(self.root, self.event)
        self.transport.reset_mock()
        with patch.object(mail.time, "monotonic", side_effect=[0, 2, 2]):
            result = mail.flush_pending(self.root, deadline_seconds=1)
        self.assertTrue(result["deadline_reached"])
        self.assertEqual(result["processed"], 0)
        self.assertEqual(result["pending_remaining"], 1)
        self.transport.assert_not_called()

    def test_flush_unreadable_outbox_is_unknown_and_not_empty(self):
        with patch.object(mail.os, "scandir", side_effect=PermissionError("secret error body")):
            result = mail.flush_pending(self.root)
        self.assertIsNone(result["pending_remaining"])
        self.assertFalse(result["remaining_scan_complete"])
        self.assertEqual(result["errors"], [{"record": "<outbox>", "error_type": "PermissionError"}])
        self.assertNotIn("secret error body", json.dumps(result))

    def test_refused_recipient_is_pending_and_does_not_record_server_body(self):
        self.smtp.send_message.return_value = {mail.RECIPIENT: (550, b"secret refusal reply")}
        receipt = mail.notify(self.root, self.event)
        self.assertFalse(receipt["smtp_accepted"])
        self.assertTrue(receipt["pending"])
        self.assertEqual(receipt["error_type"], "SMTPRecipientRefused")
        self.assertEqual(receipt["attempts"], 3)
        self.assertNotIn("refusal reply", json.dumps(receipt))

    def test_missing_or_oversized_event_identity_never_sends(self):
        for event in ({}, {"event_id": "\n"}, {"event_id": "a"*257}):
            with self.assertRaisesRegex(mail.NotificationError, "MissingEventIdentity"):
                mail.notify(self.root, event)
        self.transport.assert_not_called()


if __name__ == "__main__":
    unittest.main()
