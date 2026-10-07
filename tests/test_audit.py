"""The audit trail (`nova.audit`), tested directly — no streamlit.

Lines are read from stdout with `capsys` (the handler looks up `sys.stdout` at emit
time), never `caplog`: the audit logger does not propagate, and caplog then sees its
records only depending on test order — and never the stdout line a shipper reads.
Where each event fires in the UI is tested in test_streamlit_app.py.
"""

import inspect
import io
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta, timezone
from types import MappingProxyType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from nova import audit
from nova.access import Decision
from nova.audit import (
    Actor,
    AuditEvent,
    AuditSchemaError,
    RunOptions,
    access_denied,
    actor_for,
    emit,
    gate_event,
    logout,
    review_reopened,
    review_signed_off,
    run_options,
    session_start,
    transcript_downloaded,
    transcription_rejected,
    transcription_run,
)
from nova.transcribe import build_options
from tests.helpers import audit_lines

NOW = datetime(2026, 9, 28, 14, 3, 12, 345000, tzinfo=UTC)
TS = "2026-09-28T14:03:12.345Z"
SESSION = "a" * 32
RUN = "b" * 32
ACTOR = Actor(user="dr.smith@hospital.org", auth="oidc", session=SESSION)
ENVELOPE = ["v", "ts", "event", "outcome", "user", "auth", "session"]
# Typed Any: these tests pass values the signatures forbid, as a caller bug would.
PHI: Any = "Jane_Doe_MRN12345.wav"
OPTIONS = run_options(
    model="nova-3-pharma",  # not the default, so the line proves it is carried through
    keyterms=["hydroxyzine", "cetirizine"],
    language="en-US",
    diarize=True,
    redact=["pii", "numbers"],
)


def _hand_built_options(**overrides: Any) -> RunOptions:
    """RunOptions built directly, skipping `run_options`' checks, as a caller bug would."""
    fields: dict[str, Any] = {
        "model": "nova-3-medical",
        "language": None,
        "smart_format": True,
        "diarize": False,
        "dictation": False,
        "measurements": False,
        "redact": (),
        "n_keyterms": 0,
    }
    return RunOptions(**(fields | overrides))


def _any(value: object) -> Any:
    """`value`, typed Any — for handing a function what its signature forbids."""
    return value


def _line(event, capsys, actor=ACTOR):
    emit(actor, event, now=NOW)
    out, err = capsys.readouterr()
    assert err == ""
    (line,) = audit_lines(out)
    return line


def _envelope(event, outcome, actor=ACTOR):
    return {
        "v": 1,
        "ts": TS,
        "event": event,
        "outcome": outcome,
        "user": actor.user,
        "auth": actor.auth,
        "session": actor.session,
    }


class TestBuilders:
    """Every builder's exact line — values AND key order (the envelope first, then
    the event's fields in schema order)."""

    @pytest.mark.parametrize(
        ("event", "expected"),
        [
            (session_start(), {**_envelope("session_start", "success")}),
            (logout(), {**_envelope("logout", "success")}),
            (
                access_denied(reason="domain_not_allowed"),
                {
                    **_envelope("access_denied", "denied"),
                    "reason": "domain_not_allowed",
                },
            ),
            (
                transcription_run(
                    run=RUN,
                    input_kind="upload",
                    n_items=3,
                    n_ok=2,
                    n_skipped=1,
                    options=OPTIONS,
                ),
                {
                    **_envelope("transcription_run", "partial"),
                    "run": RUN,
                    "input_kind": "upload",
                    "n_items": 3,
                    "n_ok": 2,
                    "n_failed": 1,
                    "n_skipped": 1,
                    "model": "nova-3-pharma",
                    "language": "en-US",
                    "smart_format": True,
                    "diarize": True,
                    "dictation": False,
                    "measurements": False,
                    "redact": ["numbers", "pii"],
                    "n_keyterms": 2,
                },
            ),
            (
                transcription_rejected(
                    input_kind="record", reason="recording_too_long", n_items=1
                ),
                {
                    **_envelope("transcription_rejected", "rejected"),
                    "input_kind": "record",
                    "reason": "recording_too_long",
                    "n_items": 1,
                },
            ),
            (
                review_signed_off(
                    run=RUN, result_index=1, n_results=2, edited=True, n_flagged=3
                ),
                {
                    **_envelope("review_signed_off", "success"),
                    "run": RUN,
                    "result_index": 1,
                    "n_results": 2,
                    "edited": True,
                    "n_flagged": 3,
                },
            ),
            (
                review_reopened(run=RUN, result_index=0, n_results=2),
                {
                    **_envelope("review_reopened", "success"),
                    "run": RUN,
                    "result_index": 0,
                    "n_results": 2,
                },
            ),
            (
                transcript_downloaded(run=RUN, n_results=2, n_edited=1),
                {
                    **_envelope("transcript_downloaded", "success"),
                    "run": RUN,
                    "n_results": 2,
                    "n_edited": 1,
                },
            ),
        ],
        ids=lambda v: v.event if isinstance(v, AuditEvent) else "",
    )
    def test_exact_line(self, event, expected, capsys):
        line = _line(event, capsys)

        assert line == expected
        assert list(line) == list(expected)  # key order too
        assert list(line)[: len(ENVELOPE)] == ENVELOPE

    def test_line_is_compact_ascii_json(self, capsys):
        emit(Actor("anonymous", "anonymous", SESSION), session_start(), now=NOW)

        out = capsys.readouterr().out
        assert out == (
            '{"v":1,"ts":"2026-09-28T14:03:12.345Z","event":"session_start",'
            '"outcome":"success","user":"anonymous","auth":"anonymous",'
            f'"session":"{SESSION}"}}\n'
        )

    def test_sign_off_without_highlighting_has_null_flag_count(self, capsys):
        event = review_signed_off(
            run=RUN, result_index=0, n_results=1, edited=False, n_flagged=None
        )

        assert _line(event, capsys)["n_flagged"] is None

    @pytest.mark.parametrize(
        ("n_ok", "outcome"), [(3, "success"), (1, "partial"), (0, "failure")]
    )
    def test_run_outcome_is_derived(self, n_ok, outcome, capsys):
        event = transcription_run(
            run=RUN,
            input_kind="record",
            n_items=3,
            n_ok=n_ok,
            n_skipped=0,
            options=OPTIONS,
        )

        line = _line(event, capsys)
        assert (line["outcome"], line["n_failed"]) == (outcome, 3 - n_ok)

    def test_timestamp_is_utc_with_milliseconds(self, capsys):
        plus_eight = datetime(
            2026, 9, 28, 22, 3, 12, 345678, tzinfo=timezone(timedelta(hours=8))
        )

        emit(ACTOR, session_start(), now=plus_eight)

        (line,) = audit_lines(capsys.readouterr().out)
        assert line["ts"] == TS

    def test_default_timestamp_is_now(self, capsys):
        before = datetime.now(UTC)
        emit(ACTOR, session_start())
        after = datetime.now(UTC)

        (line,) = audit_lines(capsys.readouterr().out)
        ts = datetime.fromisoformat(line["ts"].replace("Z", "+00:00"))
        assert before - timedelta(milliseconds=1) <= ts <= after


def _message(call) -> str:
    with pytest.raises(AuditSchemaError) as exc:
        call()
    return str(exc.value)


class TestRejections:
    """Anything outside the schema raises, and the message names the field only —
    the rejected value may be the very PHI being refused."""

    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(
                lambda: transcription_rejected(
                    input_kind=PHI,
                    reason="too_many_files",
                    n_items=1,
                ),
                id="filename-as-enum",
            ),
            pytest.param(
                lambda: transcription_rejected(
                    input_kind="upload",
                    reason=PHI,
                    n_items=1,
                ),
                id="filename-as-reason",
            ),
            pytest.param(
                lambda: access_denied(reason=PHI),
                id="filename-as-deny-reason",
            ),
            pytest.param(
                lambda: review_reopened(run=PHI, result_index=0, n_results=1),
                id="filename-as-run",
            ),
            pytest.param(
                lambda: run_options(model=PHI),
                id="unknown-model",
            ),
            pytest.param(
                lambda: run_options(language=PHI),
                id="unknown-language",
            ),
            pytest.param(
                lambda: run_options(redact=["pii", PHI]),
                id="unknown-redact-group",
            ),
            pytest.param(
                lambda: Actor(user=PHI, auth="oidc", session=SESSION),
                id="non-email-user",
            ),
        ],
    )
    def test_message_never_contains_the_value(self, call):
        message = _message(call)

        assert PHI not in message
        assert "Jane" not in message

    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(
                lambda: transcription_rejected(
                    input_kind="upload", reason="too_many_files", n_items=-1
                ),
                id="negative-count",
            ),
            pytest.param(
                lambda: transcription_rejected(
                    input_kind="upload",
                    reason="too_many_files",
                    n_items=True,  # a bool is an int subclass — still refused
                ),
                id="bool-as-count",
            ),
            pytest.param(
                lambda: review_signed_off(
                    run=RUN,
                    result_index=0,
                    n_results=1,
                    edited=MagicMock(),
                    n_flagged=0,
                ),
                id="mock-as-flag",
            ),
            pytest.param(
                lambda: run_options(diarize=_any(1)),
                id="int-as-flag",
            ),
            pytest.param(
                lambda: review_reopened(run="b" * 31, result_index=0, n_results=1),
                id="short-run",
            ),
            pytest.param(
                lambda: review_reopened(run="B" * 32, result_index=0, n_results=1),
                id="non-lowercase-hex-run",
            ),
            pytest.param(
                lambda: review_reopened(run=RUN, result_index=2, n_results=2),
                id="index-past-results",
            ),
            pytest.param(
                lambda: transcription_run(
                    run=RUN,
                    input_kind="upload",
                    n_items=1,
                    n_ok=2,
                    n_skipped=0,
                    options=OPTIONS,
                ),
                id="n-ok-over-n-items",
            ),
            pytest.param(
                lambda: transcription_run(
                    run=RUN,
                    input_kind="upload",
                    n_items=0,
                    n_ok=0,
                    n_skipped=0,
                    options=OPTIONS,
                ),
                id="empty-run",
            ),
            pytest.param(
                lambda: transcription_run(
                    run=RUN,
                    input_kind="upload",
                    n_items=1,
                    n_ok=1,
                    n_skipped=0,
                    options=_hand_built_options(language=PHI),
                ),
                id="hand-built-options",
            ),
            pytest.param(
                lambda: transcription_run(
                    run=RUN,
                    input_kind="upload",
                    n_items=1,
                    n_ok=1,
                    n_skipped=0,
                    options=_hand_built_options(model=PHI),
                ),
                id="hand-built-model",
            ),
            pytest.param(
                lambda: transcript_downloaded(run=RUN, n_results=1, n_edited=2),
                id="more-edited-than-results",
            ),
            pytest.param(
                lambda: Actor(user="anonymous", auth="oidc", session=SESSION),
                id="anonymous-under-oidc",
            ),
            pytest.param(
                lambda: Actor(
                    user="dr@hospital.org", auth="anonymous", session=SESSION
                ),
                id="email-under-anonymous",
            ),
            pytest.param(
                lambda: Actor(user="dr@hospital.org", auth="none", session=SESSION),
                id="email-under-none",
            ),
            pytest.param(
                lambda: Actor(user="Dr@Hospital.org", auth="oidc", session=SESSION),
                id="unnormalized-email",
            ),
            pytest.param(
                lambda: Actor(user=None, auth="oidc", session="not-a-session"),
                id="bad-session",
            ),
            pytest.param(
                lambda: Actor(user=None, auth=_any("sso"), session=SESSION),
                id="unknown-auth",
            ),
            pytest.param(
                lambda: emit(ACTOR, session_start(), now=datetime(2026, 9, 28)),
                id="naive-now",
            ),
        ],
    )
    def test_rejected(self, call):
        _message(call)

    def test_hand_built_event_with_an_extra_field_is_refused_unnamed(self, capsys):
        event = AuditEvent("session_start", "success", {PHI: 1})

        message = _message(lambda: emit(ACTOR, event, now=NOW))

        assert PHI not in message
        assert capsys.readouterr().out == ""

    @pytest.mark.parametrize(
        "event",
        [
            pytest.param(
                AuditEvent("access_denied", "denied", {"reason": "Jane Doe"}),
                id="free-text-in-a-known-field",
            ),
            pytest.param(AuditEvent(PHI, "success", {}), id="unknown-event"),
            pytest.param(AuditEvent("session_start", "denied", {}), id="wrong-outcome"),
            pytest.param(AuditEvent("access_denied", "denied", {}), id="missing-field"),
            pytest.param(
                AuditEvent(
                    "transcription_run",
                    "success",  # but one of two failed
                    MappingProxyType(
                        {
                            **transcription_run(
                                run=RUN,
                                input_kind="upload",
                                n_items=2,
                                n_ok=1,
                                n_skipped=0,
                                options=OPTIONS,
                            ).fields
                        }
                    ),
                ),
                id="inconsistent-outcome",
            ),
        ],
    )
    def test_emit_revalidates_hand_built_events(self, event, capsys):
        message = _message(lambda: emit(ACTOR, event, now=NOW))

        assert PHI not in message and "Jane" not in message
        assert capsys.readouterr().out == ""

    def test_emit_revalidates_the_actor_by_attribute(self, capsys):
        # After a hot reload the session's Actor is the previous module's class, so
        # emit reads it by attribute — and still checks it.
        duck = _any(
            SimpleNamespace(user="dr@hospital.org", auth="oidc", session=SESSION)
        )
        emit(duck, session_start(), now=NOW)
        assert audit_lines(capsys.readouterr().out)[0]["user"] == "dr@hospital.org"

        forged = _any(SimpleNamespace(user=PHI, auth="oidc", session=SESSION))
        message = _message(lambda: emit(forged, session_start(), now=NOW))
        assert PHI not in message
        assert capsys.readouterr().out == ""


class TestEmit:
    def test_writes_exactly_one_line_to_stdout_only(self, capsys):
        emit(ACTOR, logout(), now=NOW)

        out, err = capsys.readouterr()
        assert out.count("\n") == 1 and out.endswith("\n")
        assert err == ""

    def test_a_line_that_cannot_be_written_raises(self, monkeypatch, capsys):
        # logging's default would print a traceback to stderr and carry on, silently
        # dropping the line; the audit handler re-raises so the caller fails closed.
        closed = io.StringIO()
        closed.close()
        monkeypatch.setattr(sys, "stdout", closed)

        with pytest.raises(ValueError, match="closed file"):
            emit(ACTOR, logout(), now=NOW)

        monkeypatch.undo()
        assert capsys.readouterr().err == ""

    def test_logger_is_isolated_from_the_root_logger(self):
        logger = audit.audit_logger()

        assert logger.name == "nova.audit"
        assert logger.propagate is False
        # pytest's own capture handlers attach here in-process (the subprocess test
        # below pins the exact list), so count ours by name.
        assert [h.get_name() for h in logger.handlers].count("nova.audit.stdout") == 1

    def test_handler_survives_reload_and_dictconfig(self):
        # Streamlit re-imports nova/ on a source change, and an operator's
        # dictConfig(disable_existing_loggers=True) would disable the logger: neither
        # may duplicate the handler or silence the trail. Run in a fresh interpreter,
        # never in this shared pytest process.
        code = """
import importlib
import logging
import logging.config

import nova.audit

for _ in range(5):
    nova.audit.audit_logger()
importlib.reload(nova.audit)
logging.config.dictConfig({"version": 1, "disable_existing_loggers": True})

actor = nova.audit.Actor("anonymous", "anonymous", "c" * 32)
nova.audit.emit(actor, nova.audit.session_start())
logger = logging.getLogger("nova.audit")
names = [h.get_name() for h in logger.handlers]
print("HANDLERS", names, logger.propagate, logger.disabled)
"""
        root = os.path.dirname(os.path.dirname(__file__))
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=root,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        lines = result.stdout.splitlines()
        assert len(lines) == 2, lines  # one audit line, then the report
        assert json.loads(lines[0])["event"] == "session_start"
        assert lines[1] == "HANDLERS ['nova.audit.stdout'] False False"
        assert result.stderr == ""


class TestRunOptions:
    def test_signature_mirrors_build_options(self):
        # The UI passes both the same _feature_opts dict: matching signatures turn a
        # misspelled key into a TypeError instead of a silently unaudited option.
        assert inspect.signature(run_options).parameters == (
            inspect.signature(build_options).parameters
        )

    def test_keyterms_become_a_count(self):
        options = run_options(keyterms=["hydroxyzine"])

        assert options.n_keyterms == 1
        assert "hydroxyzine" not in repr(options)

    def test_redact_is_sorted_and_deduplicated(self):
        assert run_options(redact=["pii", "numbers", "pii"]).redact == (
            "numbers",
            "pii",
        )

    def test_defaults(self):
        assert run_options() == RunOptions(
            model="nova-3-medical",
            language=None,
            smart_format=True,
            diarize=False,
            dictation=False,
            measurements=False,
            redact=(),
            n_keyterms=0,
        )

    def test_empty_language_is_unset(self):
        assert run_options(language="").language is None


class TestGate:
    @pytest.mark.parametrize(
        ("decision", "actor"),
        [
            (
                Decision("allow", email="dr@hospital.org"),
                Actor("dr@hospital.org", "oidc", SESSION),
            ),
            (Decision("anonymous"), Actor("anonymous", "anonymous", SESSION)),
            (
                Decision("deny", reason="domain_not_allowed", email="x@evil.org"),
                Actor("x@evil.org", "oidc", SESSION),
            ),
            (
                Decision("deny", reason="email_missing"),
                Actor(None, "oidc", SESSION),
            ),
            (Decision("login"), Actor(None, "none", SESSION)),
            (
                Decision("blocked", reason="auth_not_configured"),
                Actor(None, "none", SESSION),
            ),
        ],
    )
    def test_actor_for(self, decision, actor):
        assert actor_for(decision, SESSION) == actor

    @pytest.mark.parametrize(
        ("decision", "expected"),
        [
            (Decision("allow", email="dr@hospital.org"), session_start()),
            (Decision("anonymous"), session_start()),
            (
                Decision("deny", reason="email_unverified", email="dr@hospital.org"),
                access_denied(reason="email_unverified"),
            ),
            (
                Decision("blocked", reason="auth_misconfigured"),
                access_denied(reason="auth_misconfigured"),
            ),
            (Decision("login"), None),
        ],
    )
    def test_gate_event(self, decision, expected):
        assert gate_event(decision) == expected

    @pytest.mark.parametrize("kind", ["deny", "blocked"])
    def test_a_reasonless_refusal_is_refused(self, kind):
        # nova.access builds every refusal with a reason; if one ever arrived without,
        # the audit trail must not invent or drop it.
        assert _message(lambda: gate_event(Decision(kind))) == (
            "audit field 'reason' rejected"
        )
