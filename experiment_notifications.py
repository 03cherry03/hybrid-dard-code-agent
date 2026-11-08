"""Failure mail using the existing private LERRA Gmail credential.

Import and configuration_status() are local-only. SMTP is contacted only by
preflight(), notify() or flush_pending(). No credentials or log contents are
included in messages, receipts or exceptions. SMTP acceptance is not proof of
inbox delivery; a crash after acceptance but before receipt persistence can
still cause a duplicate retry with the same Message-ID.
"""
from __future__ import annotations

from contextlib import contextmanager
from email.message import EmailMessage
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import smtplib
import ssl
import stat
import tempfile
import time


RECIPIENT = "leejwian@gmail.com"
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465
SMTP_TIMEOUT_SECONDS = 15
DELIVERY_ATTEMPTS = 3
DELIVERY_DEADLINE_SECONDS = 50
LOCK_WAIT_SECONDS = 2
RETRY_DELAY_SECONDS = 1
_accepted_without_receipt: dict[str, dict] = {}
_sensitive_literals: set[str] = set()
_SENSITIVE = re.compile(r"(?i)(password|passwd|token|secret|api[-_]?key|authorization|credential)")


class NotificationError(RuntimeError):
    """Contains a class/code only, never SMTP server replies or credentials."""

    def __init__(self, error_type):
        self.error_type = _safe_type(error_type)
        super().__init__(self.error_type)


def _safe_type(value):
    value = str(value)
    return value if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", value) else "NotificationError"


def credential_path() -> Path:
    return Path(os.environ.get("HYBRID_DARD_MAIL_PASSWORD_FILE") or
                os.environ.get("LERRA_MAIL_PASSWORD_FILE") or
                Path.home() / ".config/lerra_ft_comparison/gmail_app_password").expanduser()


def configuration_status() -> dict:
    """Check credential metadata without reading the secret or using the network."""
    path = credential_path()
    try:
        info = path.lstat()
        regular = stat.S_ISREG(info.st_mode)
        private = not bool(info.st_mode & 0o077)
        owned = info.st_uid == os.getuid()
        nonempty = 0 < info.st_size <= 4096
    except OSError:
        regular = private = owned = nonempty = False
    return {"recipient": RECIPIENT, "smtp_host": SMTP_HOST, "smtp_port": SMTP_PORT,
            "credential_path": str(path), "regular": regular, "private": private,
            "owned": owned, "nonempty": nonempty,
            "metadata_ready": regular and private and owned and nonempty,
            "authentication_attempted": False, "email_sent": False}


def _password() -> str:
    try:
        fd = os.open(credential_path(), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "r", encoding="utf-8") as stream:
            info = os.fstat(stream.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or
                    info.st_uid != os.getuid() or not 0 < info.st_size <= 4096):
                raise NotificationError("UnsafeCredentialFile")
            value = stream.read(4097).strip().replace(" ", "")
    except NotificationError:
        raise
    except Exception as exc:
        raise NotificationError(type(exc).__name__) from None
    if not value or len(value) > 4096 or "\n" in value or "\r" in value:
        raise NotificationError("InvalidCredentialFile")
    _sensitive_literals.add(value)
    return value


def preflight() -> dict:
    """Authenticate and issue NOOP; do not send a verification message."""
    try:
        password = _password()
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ssl.create_default_context(),
                              timeout=SMTP_TIMEOUT_SECONDS) as smtp:
            smtp.login(RECIPIENT, password)
            code, _ = smtp.noop()
            if code != 250:
                raise NotificationError("SMTPNoopRejected")
    except NotificationError:
        raise
    except Exception as exc:
        raise NotificationError(type(exc).__name__) from None
    return {"recipient": RECIPIENT, "authentication_attempted": True,
            "authenticated": True, "email_sent": False}


def _sanitize_text(value, limit=2048):
    value = str(value)
    for secret in _sensitive_literals:
        if secret:
            value = value.replace(secret, "[REDACTED]")
    value = re.sub(r"(?i)\b(Bearer|Basic)\s+\S+", r"\1 [REDACTED]", value)
    value = re.sub(r"(?i)((?:password|passwd|token|secret|api[-_]?key|authorization|credential)\s*[=:]\s*)\S+",
                   r"\1[REDACTED]", value)
    value = re.sub(r"(https?://[^\s?]+)\?\S+", r"\1?[REDACTED]", value)
    return value.replace("\x00", "")[:limit]


def _safe_command(command):
    if not isinstance(command, (list, tuple)):
        # A shell string is not reliably parseable. Do not transmit it.
        return "[command omitted: expected argv list]"
    result, hide_next = [], False
    for item in command[:100]:
        value = str(item)
        if hide_next:
            result.append("[REDACTED]")
            hide_next = False
        elif _SENSITIVE.search(value.split("=", 1)[0]):
            if "=" in value:
                result.append(_sanitize_text(value.split("=", 1)[0]) + "=[REDACTED]")
            else:
                result.append(_sanitize_text(value))
                hide_next = True
        else:
            result.append(_sanitize_text(value))
    return result


def _safe_event(event):
    if (not isinstance(event, dict) or not isinstance(event.get("event_id"), str) or
            not event["event_id"].strip() or len(event["event_id"]) > 256 or
            any(char in event["event_id"] for char in ("\n", "\r", "\x00"))):
        raise NotificationError("MissingEventIdentity")
    result = {"event_id": _sanitize_text(event["event_id"], 256)}
    for name in ("invocation_id", "stage", "signal", "host", "output", "log_path"):
        if name in event:
            result[name] = _sanitize_text(event[name])
    if "exit_code" in event:
        result["exit_code"] = event["exit_code"] if type(event["exit_code"]) is int else None
    if "error_type" in event:
        result["error_type"] = _safe_type(event["error_type"])
    if "command" in event:
        result["command"] = _safe_command(event["command"])
    return result


def _new_receipt(event):
    event = _safe_event(event)
    key = hashlib.sha256(event["event_id"].encode()).hexdigest()
    return {"schema_version": 1, "event": event, "recipient": RECIPIENT,
            "message_id": f"<hybrid-dard-{key}@local.invalid>",
            "smtp_accepted": False, "pending": True, "receipt_persisted": False,
            "attempts": 0, "error_type": None, "created_utc_epoch": time.time()}


def _persist(path, receipt):
    if path is None:
        receipt["receipt_persisted"] = False
        return False
    temporary = None
    try:
        fd, temporary = tempfile.mkstemp(prefix=".notification-", suffix=".tmp", dir=path.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            saved = {**receipt, "receipt_persisted": True}
            json.dump(saved, stream, sort_keys=True, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        receipt["receipt_persisted"] = True
        receipt.pop("storage_error_type", None)
        return True
    except Exception as exc:
        receipt["receipt_persisted"] = False
        receipt["storage_error_type"] = _safe_type(type(exc).__name__)
        return False
    finally:
        if temporary:
            try:
                Path(temporary).unlink(missing_ok=True)
            except OSError:
                pass


class _OutboxBusy(Exception):
    pass


@contextmanager
def _event_lock(path):
    deadline = time.monotonic() + LOCK_WAIT_SECONDS
    fd = os.open(path.with_suffix(".lock"), os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(fd, "a") as stream:
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise _OutboxBusy()
                time.sleep(0.05)
        yield


def _read_receipt(path):
    record = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(record, dict) or record.get("schema_version") != 1 or
            record.get("recipient") != RECIPIENT or not isinstance(record.get("event"), dict) or
            type(record.get("smtp_accepted")) is not bool or type(record.get("pending")) is not bool or
            record["pending"] == record["smtp_accepted"] or
            type(record.get("attempts")) is not int or record["attempts"] < 0):
        raise NotificationError("InvalidOutboxRecord")
    event = _safe_event(record["event"])
    expected = hashlib.sha256(event["event_id"].encode()).hexdigest()
    if path.stem != expected or record.get("message_id") != f"<hybrid-dard-{expected}@local.invalid>":
        raise NotificationError("InvalidOutboxIdentity")
    record["event"] = event
    record["error_type"] = _safe_type(record["error_type"]) if record.get("error_type") else None
    return record


def _message(receipt):
    message = EmailMessage()
    message["From"] = message["To"] = RECIPIENT
    message["Message-ID"] = receipt["message_id"]
    event = receipt["event"]
    is_test = (event.get("stage") == "notification_test" and event.get("error_type") == "ExplicitTest"
               and event.get("exit_code") == 0)
    if is_test:
        message["Subject"] = "[Hybrid-DARD notification test] Delivery check"
        introduction = ("This is an explicitly requested email delivery test. "
                        "No experiment failed or was started by this test.\n\n")
    else:
        message["Subject"] = " ".join(
            f"[Hybrid-DARD failure] {event.get('stage', 'unknown')} exit={event.get('exit_code', 'unknown')}".splitlines())[:220]
        introduction = ""
    message.set_content(introduction + json.dumps(event, ensure_ascii=False, indent=2) +
                        "\n\nLog files remain local; their contents are not attached.\n")
    return message


def _deliver(path, receipt, *, deadline=None):
    if receipt["smtp_accepted"]:
        return receipt
    cache_key = receipt["message_id"]
    remembered = _accepted_without_receipt.get(cache_key)
    if remembered is not None:
        receipt = dict(remembered)
        _persist(path, receipt)
        return receipt
    deadline = min(deadline or float("inf"), time.monotonic() + DELIVERY_DEADLINE_SECONDS)
    for attempt in range(DELIVERY_ATTEMPTS):
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            receipt["error_type"] = receipt.get("error_type") or "DeliveryDeadline"
            _persist(path, receipt)
            break
        receipt["attempts"] += 1
        receipt["last_attempt_utc_epoch"] = time.time()
        _persist(path, receipt)
        accepted = False
        try:
            password = _password()
            # Sanitize once more after the secret is loaded, before serializing.
            receipt["event"] = _safe_event(receipt["event"])
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ssl.create_default_context(),
                                  timeout=max(0.1, min(SMTP_TIMEOUT_SECONDS, remaining))) as smtp:
                smtp.login(RECIPIENT, password)
                if smtp.send_message(_message(receipt)):
                    raise NotificationError("SMTPRecipientRefused")
                accepted = True
        except Exception as exc:
            if accepted:
                receipt["cleanup_error_type"] = _safe_type(type(exc).__name__)
            else:
                receipt["error_type"] = (exc.error_type if isinstance(exc, NotificationError)
                                         else _safe_type(type(exc).__name__))
        if accepted:
            receipt.update(smtp_accepted=True, pending=False, error_type=None,
                           accepted_utc_epoch=time.time())
            _accepted_without_receipt[cache_key] = dict(receipt)
            _persist(path, receipt)
            return receipt
        receipt["pending"] = True
        _persist(path, receipt)
        if attempt+1 < DELIVERY_ATTEMPTS:
            time.sleep(max(0, min(RETRY_DELAY_SECONDS, deadline-time.monotonic())))
    return receipt


def notify(directory: Path, event: dict) -> dict:
    """Persist and attempt a failure notification; never raise SMTP error text."""
    receipt = _new_receipt(event)
    outbox = Path(directory) / "notifications"
    path = outbox / (hashlib.sha256(receipt["event"]["event_id"].encode()).hexdigest() + ".json")
    try:
        outbox.mkdir(parents=True, exist_ok=True, mode=0o700)
        with _event_lock(path):
            if path.exists():
                previous = _read_receipt(path)
                if previous["event"] != receipt["event"]:
                    previous["event_conflict"] = True
                return _deliver(path, previous)
            _persist(path, receipt)
            return _deliver(path, receipt)
    except _OutboxBusy:
        receipt["error_type"] = "OutboxBusy"
        return receipt
    except Exception as exc:
        # Unwritable/broken local storage must not prevent an SMTP attempt.
        receipt["storage_error_type"] = _safe_type(type(exc).__name__)
        return _deliver(None, receipt)


def flush_pending(directory: Path, *, max_records=20, deadline_seconds=60) -> dict:
    """Retry bounded pending outbox work; no secret values or log bodies returned."""
    if (type(max_records) is not int or max_records < 1
            or isinstance(deadline_seconds, bool) or not isinstance(deadline_seconds, (int, float))
            or not math.isfinite(deadline_seconds) or deadline_seconds <= 0):
        raise NotificationError("InvalidRetryBudget")
    outbox = Path(directory) / "notifications"
    result = {"processed": 0, "smtp_accepted": 0, "pending": 0, "pending_remaining": 0,
              "errors": [], "deadline_reached": False, "max_records_reached": False,
              "remaining_scan_complete": True}
    try:
        with os.scandir(outbox) as entries:
            paths = sorted(Path(entry.path) for entry in entries if entry.name.endswith(".json"))
    except FileNotFoundError:
        return result
    except OSError as exc:
        result.update(pending_remaining=None, remaining_scan_complete=False)
        result["errors"].append({"record": "<outbox>", "error_type": _safe_type(type(exc).__name__)})
        return result
    deadline = time.monotonic()+deadline_seconds
    for path in paths:
        if result["processed"] >= max_records or time.monotonic() >= deadline:
            result["deadline_reached"] = time.monotonic() >= deadline
            break
        try:
            with _event_lock(path):
                receipt = _read_receipt(path)
                if receipt["smtp_accepted"] or not receipt["pending"]:
                    continue
                result["processed"] += 1
                receipt = _deliver(path, receipt, deadline=deadline)
                result["smtp_accepted"] += int(receipt["smtp_accepted"])
                result["pending"] += int(receipt["pending"])
        except Exception as exc:
            result["errors"].append({"record": path.name, "error_type":
                exc.error_type if isinstance(exc, NotificationError) else _safe_type(type(exc).__name__)})
    # This is a fresh read-only snapshot, including records not attempted due
    # to the retry budget. Unknown/corrupt records must not imply an empty queue.
    try:
        with os.scandir(outbox) as entries:
            remaining_paths = sorted(Path(entry.path) for entry in entries if entry.name.endswith(".json"))
        for path in remaining_paths:
            try:
                record = _read_receipt(path)
                result["pending_remaining"] += int(record["pending"])
            except Exception as exc:
                result["remaining_scan_complete"] = False
                error = {"record": path.name, "error_type":
                    exc.error_type if isinstance(exc, NotificationError) else _safe_type(type(exc).__name__)}
                if error not in result["errors"]:
                    result["errors"].append(error)
    except OSError as exc:
        result.update(pending_remaining=None, remaining_scan_complete=False)
        result["errors"].append({"record": "<outbox>", "error_type": _safe_type(type(exc).__name__)})
    result["max_records_reached"] = result["processed"] >= max_records and (
        result["pending_remaining"] != 0 or not result["remaining_scan_complete"])
    return result
