"""Audit trail: who did what, as one PHI-free JSON object per line on stdout.

Imports no streamlit. Two halves:

- **Builders** (`session_start`, `transcription_run`, ...) return an `AuditEvent`
  whose fields are fixed per event and whose values can only be counts, real bools,
  members of fixed sets, or random 32-hex tokens. There is no free-text field, so a
  filename, transcript, keyterm, URL, or exception message has nowhere to go.
- **`emit(actor, event)`** re-validates the actor and every field (so a hand-built
  event gets the same checks), prepends the envelope — `v, ts, event, outcome, user,
  auth, session` — and writes one line through the dedicated `nova.audit` logger,
  whose single handler writes to stdout. Streamlit's own logs go to stderr, so a
  log shipper can route by stream.

`user` is the signed-in staff email (workforce identity, which HIPAA audit controls
ask for; not patient PHI), the literal "anonymous", or null. `AuditSchemaError`
names the rejected FIELD, never its value: the value may be the very PHI being
refused, and the message can reach Streamlit's stderr log.

No `from __future__ import annotations` here: `run_options`' signature is compared
with `build_options`' parameter by parameter, annotations included.
"""

import json
import logging
import re
import sys
import threading
from collections.abc import Callable, Container, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Literal, get_args

from nova.access import Decision, DenyReason, normalize_email
from nova.config import (
    DEFAULT_DIARIZE,
    DEFAULT_DICTATION,
    DEFAULT_MEASUREMENTS,
    DEFAULT_MODEL,
    DEFAULT_SMART_FORMAT,
    LANGUAGES,
    MODELS,
    REDACT_GROUPS,
)

LOGGER_NAME = "nova.audit"
SCHEMA_VERSION = 1
# The handler is found again by NAME, not isinstance: Streamlit re-imports nova/ on a
# source change (hot reload), which defines a new handler class while the process-wide
# logger keeps the old handler — an isinstance guard would then add a second one.
_HANDLER_NAME = "nova.audit.stdout"
ANONYMOUS = "anonymous"

AuthKind = Literal["oidc", "anonymous", "none"]
InputKind = Literal["upload", "record"]
RejectReason = Literal[
    "too_many_files", "all_oversize", "recording_unreadable", "recording_too_long"
]
Outcome = Literal["success", "partial", "failure", "denied", "rejected"]

_HEX32 = re.compile(r"[0-9a-f]{32}")


class AuditSchemaError(ValueError):
    """An audit value failed validation. The message names the field, never the value."""

    def __init__(self, field_name: str) -> None:
        super().__init__(f"audit field {field_name!r} rejected")
        self.field_name = field_name


# Field validators: each returns the value unchanged (or normalized) or raises.
def _count(name: str, value: object) -> int:
    # `type(...) is int` rejects bool (an int subclass) and MagicMock alike.
    if type(value) is not int or value < 0:
        raise AuditSchemaError(name)
    return value


def _optional_count(name: str, value: object) -> int | None:
    return None if value is None else _count(name, value)


def _flag(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise AuditSchemaError(name)
    return value


def _member(name: str, value: object, allowed: Container[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise AuditSchemaError(name)
    return value


def _token(name: str, value: object) -> str:
    if not isinstance(value, str) or not _HEX32.fullmatch(value):
        raise AuditSchemaError(name)
    return value


def _model(name: str, value: object) -> str:
    return _member(name, value, MODELS)


def _language(name: str, value: object) -> str | None:
    return None if value is None else _member(name, value, LANGUAGES)


def _groups(name: str, value: object) -> list[str]:
    if type(value) is not list:
        raise AuditSchemaError(name)
    return [_member(name, group, REDACT_GROUPS) for group in value]


def _input_kind(name: str, value: object) -> str:
    return _member(name, value, get_args(InputKind))


def _deny_reason(name: str, value: object) -> str:
    return _member(name, value, get_args(DenyReason))


def _reject_reason(name: str, value: object) -> str:
    return _member(name, value, get_args(RejectReason))


def _check_actor(user: object, auth: object, session: object) -> None:
    if not isinstance(auth, str) or auth not in get_args(AuthKind):
        raise AuditSchemaError("auth")
    _token("session", session)
    if auth == "anonymous":
        valid = user == ANONYMOUS
    elif auth == "none":
        valid = user is None
    else:  # oidc: a normalized email, or None (a sign-in that carried no usable one)
        valid = user is None or normalize_email(user) == user
    if not valid:
        raise AuditSchemaError("user")


@dataclass(frozen=True)
class Actor:
    """Who is acting: a normalized email under "oidc", "anonymous" under "anonymous",
    None under "none" (not signed in); `session` is a random 32-hex token."""

    user: str | None
    auth: AuthKind
    session: str

    def __post_init__(self) -> None:
        _check_actor(self.user, self.auth, self.session)


@dataclass(frozen=True)
class RunOptions:
    """The auditable projection of the Features options: flags and counts only —
    keyterms become `n_keyterms`, so their text never enters the audit layer."""

    model: str
    language: str | None
    smart_format: bool
    diarize: bool
    dictation: bool
    measurements: bool
    redact: tuple[str, ...]
    n_keyterms: int


@dataclass(frozen=True)
class AuditEvent:
    """A validated event, built by the builders below (`emit` re-checks it anyway)."""

    event: str
    outcome: Outcome
    fields: Mapping[str, object]


_Check = Callable[[str, object], object]


def _int(fields: Mapping[str, object], name: str) -> int:
    value = fields[name]
    if type(value) is not int:
        raise AuditSchemaError(name)
    return value


# Cross-field checks, run after every field passed its own validator.
def _check_run(fields: Mapping[str, object], outcome: str) -> None:
    n_items, n_ok = _int(fields, "n_items"), _int(fields, "n_ok")
    if n_items < 1:
        raise AuditSchemaError("n_items")
    if n_ok > n_items:
        raise AuditSchemaError("n_ok")
    if _int(fields, "n_failed") != n_items - n_ok:
        raise AuditSchemaError("n_failed")
    if outcome != _run_outcome(n_items, n_ok):
        raise AuditSchemaError("outcome")


def _check_position(fields: Mapping[str, object], outcome: str) -> None:
    if _int(fields, "result_index") >= _int(fields, "n_results"):
        raise AuditSchemaError("result_index")


def _check_download(fields: Mapping[str, object], outcome: str) -> None:
    n_results = _int(fields, "n_results")
    if n_results < 1:
        raise AuditSchemaError("n_results")
    if _int(fields, "n_edited") > n_results:
        raise AuditSchemaError("n_edited")


@dataclass(frozen=True)
class _Spec:
    outcomes: frozenset[str]
    fields: Mapping[str, _Check]  # in output order
    consistent: Callable[[Mapping[str, object], str], None] | None = None


_RUN_OPTION_FIELDS: dict[str, _Check] = {
    "model": _model,
    "language": _language,
    "smart_format": _flag,
    "diarize": _flag,
    "dictation": _flag,
    "measurements": _flag,
    "redact": _groups,
    "n_keyterms": _count,
}
_POSITION: dict[str, _Check] = {
    "run": _token,
    "result_index": _count,
    "n_results": _count,
}
# Every event's exact field set (in output order) and the outcomes it may carry.
_SCHEMA: dict[str, _Spec] = {
    "session_start": _Spec(frozenset({"success"}), {}),
    "access_denied": _Spec(frozenset({"denied"}), {"reason": _deny_reason}),
    "logout": _Spec(frozenset({"success"}), {}),
    "transcription_run": _Spec(
        frozenset({"success", "partial", "failure"}),
        {
            "run": _token,
            "input_kind": _input_kind,
            "n_items": _count,
            "n_ok": _count,
            "n_failed": _count,
            "n_skipped": _count,
            **_RUN_OPTION_FIELDS,
        },
        _check_run,
    ),
    "transcription_rejected": _Spec(
        frozenset({"rejected"}),
        {"input_kind": _input_kind, "reason": _reject_reason, "n_items": _count},
    ),
    "review_signed_off": _Spec(
        frozenset({"success"}),
        {**_POSITION, "edited": _flag, "n_flagged": _optional_count},
        _check_position,
    ),
    "review_reopened": _Spec(frozenset({"success"}), _POSITION, _check_position),
    "transcript_downloaded": _Spec(
        frozenset({"success"}),
        {"run": _token, "n_results": _count, "n_edited": _count},
        _check_download,
    ),
}


def _validated(event: object, outcome: object, fields: object) -> dict[str, object]:
    """The event's fields, checked against its schema and put in output order.

    Unknown events and fields are refused without naming them: a hand-built event's
    stray key could itself be PHI.
    """
    spec = _SCHEMA.get(event) if isinstance(event, str) else None
    if spec is None:
        raise AuditSchemaError("event")
    if not isinstance(outcome, str) or outcome not in spec.outcomes:
        raise AuditSchemaError("outcome")
    if not isinstance(fields, Mapping) or set(fields) != set(spec.fields):
        raise AuditSchemaError("fields")
    checked = {name: check(name, fields[name]) for name, check in spec.fields.items()}
    if spec.consistent is not None:
        spec.consistent(checked, outcome)
    return checked


def _event(event: str, outcome: Outcome, **fields: object) -> AuditEvent:
    return AuditEvent(
        event, outcome, MappingProxyType(_validated(event, outcome, fields))
    )


def _run_outcome(n_items: int, n_ok: int) -> Outcome:
    if n_ok == n_items:
        return "success"
    return "failure" if n_ok == 0 else "partial"


def run_options(
    *,
    model: str = DEFAULT_MODEL,
    keyterms: list[str] | None = None,
    language: str | None = None,
    smart_format: bool = DEFAULT_SMART_FORMAT,
    dictation: bool = DEFAULT_DICTATION,
    measurements: bool = DEFAULT_MEASUREMENTS,
    diarize: bool = DEFAULT_DIARIZE,
    redact: list[str] | None = None,
) -> RunOptions:
    """Project the Features options onto what the audit trail may record.

    Mirrors `build_options`' signature (pinned by a test), so the UI passes it the
    same `_feature_opts()` dict and a misspelled key raises TypeError. Keyterms are
    reduced to a count; the model must be a known one; the language must be a known
    one (or unset); redact groups
    must be known and come back sorted; the four toggles must be real bools.
    """
    if keyterms is not None and not isinstance(keyterms, list | tuple):
        raise AuditSchemaError("keyterms")
    if redact is not None and not isinstance(redact, list | tuple):
        raise AuditSchemaError("redact")
    groups = {_member("redact", group, REDACT_GROUPS) for group in redact or ()}
    return RunOptions(
        model=_model("model", model),
        language=_language("language", language or None),
        smart_format=_flag("smart_format", smart_format),
        diarize=_flag("diarize", diarize),
        dictation=_flag("dictation", dictation),
        measurements=_flag("measurements", measurements),
        redact=tuple(sorted(groups)),
        n_keyterms=len(keyterms or ()),
    )


def actor_for(decision: Decision, session: str) -> Actor:
    """The audit identity for an access-gate decision.

    Allowed or denied sign-ins are "oidc" with their normalized email (None when a
    denied token carried no usable one); anonymous local development is "anonymous";
    a visitor who has not signed in (login) or a blocked deploy is "none" with no user.
    """
    match decision.kind:
        case "allow" | "deny":
            return Actor(decision.email, "oidc", session)
        case "anonymous":
            return Actor(ANONYMOUS, "anonymous", session)
        case "login" | "blocked":
            return Actor(None, "none", session)
    raise AuditSchemaError("kind")


def gate_event(decision: Decision) -> AuditEvent | None:
    """The gate's audit event: session_start when access is granted, access_denied
    (with the decision's reason) when refused, and nothing for the sign-in screen."""
    if decision.kind in ("allow", "anonymous"):
        return session_start()
    if decision.kind in ("deny", "blocked"):
        if decision.reason is None:  # every refusal carries one (nova.access)
            raise AuditSchemaError("reason")
        return access_denied(reason=decision.reason)
    return None


def session_start() -> AuditEvent:
    """Access granted to a new session (after sign-in, this is the login record)."""
    return _event("session_start", "success")


def access_denied(*, reason: DenyReason) -> AuditEvent:
    """Access refused: a signed-in account outside the policy, or a blocked deploy."""
    return _event("access_denied", "denied", reason=reason)


def logout() -> AuditEvent:
    """The Sign out button."""
    return _event("logout", "success")


def transcription_run(
    *,
    run: str,
    input_kind: InputKind,
    n_items: int,
    n_ok: int,
    n_skipped: int,
    options: RunOptions,
) -> AuditEvent:
    """A finished Run: counts and options only. `n_failed` and the outcome (success /
    partial / failure) are derived here; `n_skipped` counts oversize uploads."""
    n_items, n_ok = _count("n_items", n_items), _count("n_ok", n_ok)
    if n_ok > n_items:
        raise AuditSchemaError("n_ok")
    return _event(
        "transcription_run",
        _run_outcome(n_items, n_ok),
        run=run,
        input_kind=input_kind,
        n_items=n_items,
        n_ok=n_ok,
        n_failed=n_items - n_ok,
        n_skipped=n_skipped,
        model=options.model,
        language=options.language,
        smart_format=options.smart_format,
        diarize=options.diarize,
        dictation=options.dictation,
        measurements=options.measurements,
        redact=list(options.redact),
        n_keyterms=options.n_keyterms,
    )


def transcription_rejected(
    *, input_kind: InputKind, reason: RejectReason, n_items: int
) -> AuditEvent:
    """A Run refused before any audio was sent."""
    return _event(
        "transcription_rejected",
        "rejected",
        input_kind=input_kind,
        reason=reason,
        n_items=n_items,
    )


def review_signed_off(
    *,
    run: str,
    result_index: int,
    n_results: int,
    edited: bool,
    n_flagged: int | None,
) -> AuditEvent:
    """A result marked "Reviewed against the audio". `edited`: its export text differs
    from Deepgram's; `n_flagged`: its low-confidence word count (None if unavailable)."""
    return _event(
        "review_signed_off",
        "success",
        run=run,
        result_index=result_index,
        n_results=n_results,
        edited=edited,
        n_flagged=n_flagged,
    )


def review_reopened(*, run: str, result_index: int, n_results: int) -> AuditEvent:
    """A signed-off result made unreviewed again."""
    return _event(
        "review_reopened",
        "success",
        run=run,
        result_index=result_index,
        n_results=n_results,
    )


def transcript_downloaded(*, run: str, n_results: int, n_edited: int) -> AuditEvent:
    """One generated download file: how many results, and how many were edited."""
    return _event(
        "transcript_downloaded",
        "success",
        run=run,
        n_results=n_results,
        n_edited=n_edited,
    )


class _StdoutHandler(logging.StreamHandler):
    """Writes to whatever `sys.stdout` is at EMIT time, not at construction.

    A handler that pins the stream it was built with never reaches a later
    replacement of `sys.stdout` (pytest's capsys swaps it per test). `Handler.handle`
    holds the handler lock around `emit`, so the swap is race-free across Streamlit's
    per-session script threads; `StreamHandler.emit` then writes the line and flushes.
    """

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stdout
        super().emit(record)

    def handleError(self, record: logging.LogRecord) -> None:
        """Re-raise a failed write instead of logging's default — a traceback on
        stderr, then carry on — so an audit line that cannot be written fails the
        action it records (`emit` raises). The error is the stream's own (a closed or
        broken stdout), which carries no event content."""
        error = sys.exc_info()[1]
        if error is not None:
            raise error


_setup_lock = threading.Lock()


def audit_logger() -> logging.Logger:
    """The `nova.audit` logger, with exactly one stdout handler installed.

    A plain stdlib logger — never `streamlit.logger.get_logger`, which would add
    Streamlit's stderr handler, and whose `logger.level` setting (often "error" in
    production) would drop these INFO lines. Idempotent across reruns, threads, and
    hot reloads (the handler is found by name). Level, `propagate=False` and
    `disabled=False` are re-asserted on every call, so a later
    `logging.config.dictConfig(disable_existing_loggers=True)` cannot switch the trail
    off. Called lazily from `emit`, never at import.
    """
    logger = logging.getLogger(LOGGER_NAME)
    with _setup_lock:
        if not any(h.get_name() == _HANDLER_NAME for h in logger.handlers):
            handler = _StdoutHandler(sys.stdout)
            handler.set_name(_HANDLER_NAME)
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.disabled = False
    return logger


def _timestamp(now: datetime | None) -> str:
    moment = datetime.now(UTC) if now is None else now
    if not isinstance(moment, datetime) or moment.utcoffset() is None:
        raise AuditSchemaError("ts")  # a naive time would be silently mislabeled UTC
    return (
        moment.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    )


def emit(actor: Actor, event: AuditEvent, *, now: datetime | None = None) -> None:
    """Write one audit line: the envelope, then the event's fields, as compact JSON.

    Everything is re-validated here — the actor and every field — so this is the
    single choke point whatever built the arguments. The actor is read by attribute
    (not isinstance-checked): after a hot reload the session's stored Actor is an
    instance of the previous module's class. Raises if the line is invalid
    (`AuditSchemaError`) or cannot be written (the stream's error), so callers fail
    closed.
    """
    _check_actor(actor.user, actor.auth, actor.session)
    fields = _validated(event.event, event.outcome, event.fields)
    record: dict[str, object] = {
        "v": SCHEMA_VERSION,
        "ts": _timestamp(now),
        "event": event.event,
        "outcome": event.outcome,
        "user": actor.user,
        "auth": actor.auth,
        "session": actor.session,
        **fields,
    }
    line = json.dumps(record, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    audit_logger().info(line)
