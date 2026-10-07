import importlib.util
import io
import os
import re
import time
import uuid
import wave
from collections.abc import Callable, Iterable
from concurrent.futures import as_completed
from typing import Any

import streamlit as st
from deepgram import DeepgramClient
from dotenv import dotenv_values
from streamlit.errors import StreamlitSecretNotFoundError

from nova import audit
from nova.access import (
    Decision,
    SecretsSnapshot,
    anonymous_opt_out,
    decide_access,
    report_problem,
)
from nova.config import (
    ALLOW_ANONYMOUS_ENV,
    AUDIO_EXTENSIONS as _AUDIO_EXTENSIONS,
    DEFAULT_DIARIZE,
    DEFAULT_DICTATION,
    DEFAULT_LANGUAGE,
    DEFAULT_MEASUREMENTS,
    DEFAULT_MODEL,
    DEFAULT_SMART_FORMAT,
    LANGUAGES as _LANGUAGES,
    LOW_CONFIDENCE_THRESHOLD,
    MAX_FILE_SIZE,
    MAX_KEYTERMS,
    MAX_UPLOADS,
    MODELS as _MODELS,
    REDACT_GROUPS as _REDACT_GROUPS,
)
from nova.results import (
    Token,
    diarized_segments as _diarized_segments,
    first_alternative as _first_alternative,
    flagged_runs as _flagged_runs,
    low_confidence_count as _low_confidence_count,
    speaker_label as _speaker_label,
    transcript_text as _transcript_text,
)
from nova.transcribe import build_options, option_warnings, transcribe_batch


def _names_opt_out(keys: Iterable[object]) -> bool:
    """Whether any key names the anonymous opt-out, compared case-insensitively.

    Case-insensitive because Windows' `os.environ` is: a lowercase
    `nova_allow_anonymous` in `.env` or `secrets.toml` would land there as the real
    variable, so each file's detector must see it too.
    """
    return any(isinstance(k, str) and k.upper() == ALLOW_ANONYMOUS_ENV for k in keys)


def _load_dotenv() -> None:
    """Copy `.env` into `os.environ` like `load_dotenv()` (never overriding a set
    variable), except the anonymous opt-out, which is never copied.

    Filtered because `os.environ` outlives the script run: had `.env` ever put the
    opt-out there, deleting the line (as the blocked app's operator message says to)
    would leave it set until a restart, and the next run would open anonymously.
    The opt-out in `.env` is only ever detected (`_dotenv_has_opt_out`), which
    blocks the app.
    """
    for key, value in dotenv_values().items():
        if value is not None and not _names_opt_out([key]) and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

# 30 minutes — covers a standard clinic encounter. At st.audio_input's default 16 kHz
# 16-bit mono WAV (32 kB/s) that is ~58 MB: well under the 200 MB upload cap, but
# past MAX_PLAYBACK_BYTES from ~13.6 min, so longer recordings skip the output-panel
# player (the Record tab's own widget still plays them back).
MAX_RECORDING_SECONDS = 30 * 60
MAX_PLAYBACK_BYTES = 25 * 1024 * 1024  # larger uploads skip inline playback (memory)
# Fixed height (px) of the transcript output panel. Sized so a single result's panel
# — below the title, output header, download row, and pinned player — ends above the
# fold of a ~840px-tall viewport (1080p display minus browser chrome), avoiding a page
# scroll nested around the panel's own scroll. Measured when a tab strip sat above the
# panel; the caption header that replaced it is shorter, so the panel still clears the
# fold. Streamlit has no viewport-relative height, so a fixed value is the only option.
# The review editor and Reviewed checkbox render inside the panel's scroll container,
# so they add nothing above it (a long transcript then scrolls through twice: the
# highlighted view, then the editor). Signed in, the account block sits in the
# sidebar, so it adds nothing above the panel either; only the local-development
# anonymous-mode banner pushes the main area down (one callout, ~60px), exactly as
# the "API key required" warning does.
OUTPUT_HEIGHT = 480
INPUT_OUTPUT_RATIO = (2, 3)  # main-area column widths: audio inputs | output panel

# Material Symbol icons reused across status callouts (single-use icons stay inline).
_ICON_ERROR = ":material/error:"
_ICON_WARNING = ":material/warning:"

# Per-speaker label colors for diarized transcripts, cycled by speaker index.
# These are native Markdown color directives (:blue-background[...]) — no raw
# HTML/CSS — so they stay test-safe plain strings. 'rainbow' is excluded: it is
# not a valid background-highlight color.
_SPEAKER_COLORS = ("blue", "green", "violet", "orange", "red", "gray")

_AUDIO_TYPES = [ext.lstrip(".") for ext in _AUDIO_EXTENSIONS]
_AUDIO_MIME = {
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
}

# Inline Markdown metacharacters, escaped so transcript text renders literally. `$` is
# included because Streamlit's Markdown always enables single-dollar math: unescaped,
# smart-formatted currency such as "$20-$30" could render as a formula.
_MARKDOWN_SPECIAL = re.compile(r"([\\`*_\[\]~$])")


def _escape_markdown(text: str) -> str:
    """Backslash-escape inline Markdown metacharacters so text renders verbatim."""
    return _MARKDOWN_SPECIAL.sub(r"\\\1", text)


def _playback_source(data: bytes) -> bytes | None:
    """Keep small audio for inline playback; drop large upload/recording bytes (memory)."""
    return data if len(data) <= MAX_PLAYBACK_BYTES else None


def _secret_api_key() -> str:
    """Read `DEEPGRAM_API_KEY` from `st.secrets`, tolerating the no-secrets-file case.

    Deployment fallback for hosts that have no `.env` — Streamlit Community Cloud
    writes dashboard secrets to a `secrets.toml` instead. Guarded rather than
    probed because *every* access to `st.secrets` raises when no secrets file
    exists (`.get()` and `in` included), and that is the normal local setup here.
    Touching `st.secrets` at all is also what mirrors top-level secrets into
    `os.environ`; nothing else in the app does, so on Cloud the environment read
    alone comes up empty.
    """
    try:
        value = st.secrets.get("DEEPGRAM_API_KEY", "")
    except StreamlitSecretNotFoundError:
        return ""
    return value if isinstance(value, str) else ""


# Sign-in / access control. The policy is decided in nova.access; these helpers read
# its inputs (secrets, claims, environment) and render the outcome.
def _secrets_snapshot() -> SecretsSnapshot:
    """Read [auth] / [access] / the opt-out's presence from `st.secrets`, guarded.

    Read through the public `st.secrets` (so AppTest's `at.secrets` drives it). No
    secrets file at all is "missing"; any other read failure (a TOML syntax error, a
    bad path) is "malformed", which the gate treats as fail-closed. The opt-out
    counts as present under any value and any letter case, since Streamlit copies a
    top-level secret into `os.environ` as `str(value)` (`= 1` becomes "1").
    """
    try:
        return SecretsSnapshot(
            "ok",
            auth=st.secrets.get("auth"),
            access=st.secrets.get("access"),
            anonymous_opt_out_present=_names_opt_out(st.secrets),
        )
    except StreamlitSecretNotFoundError as exc:
        missing = getattr(exc, "error_id", None) == "no-secrets-found"
        return SecretsSnapshot("missing" if missing else "malformed")


def _authlib_installed() -> bool:
    """Whether Authlib (the streamlit[auth] extra) is importable; a test seam."""
    return importlib.util.find_spec("authlib") is not None


def _dotenv_has_opt_out() -> bool:
    """Whether `.env` names the anonymous opt-out, in any case (a test seam).

    `_load_dotenv` never copies it, so a dev `.env` copied onto a host cannot switch
    sign-in off; the gate blocks instead, to say so. The opt-out must come from the
    real process environment. Re-read on every run, so deleting the line unblocks
    the app without a restart — and into anonymous mode only if the process
    environment itself sets the opt-out.
    """
    return _names_opt_out(dotenv_values())


def _sign_in_label(provider: str | None) -> str:
    """The button label: "Sign in" (default provider) or "Sign in with <Name>"."""
    if provider is None:
        return "Sign in"
    return f"Sign in with {_escape_markdown(provider.capitalize())}"


def _deny_message(decision: Decision) -> str:
    """The denied visitor's error: why, naming their (escaped) email, then what to do."""
    email = _escape_markdown(decision.email) if decision.email else "This account"
    fallback = DENY_MESSAGES["domain_not_allowed"]
    template = DENY_MESSAGES.get(decision.reason or "", fallback)
    return template.format(email=email) + DENY_SUFFIX


def _access_gate() -> Decision | None:
    """Decide who may use the app, render the outcome, and return the Decision.

    Returns the Decision when the app may continue (signed in and allowed, or
    anonymous local development), else None after rendering the refusal — the
    caller then calls `st.stop()`. `access_ok` is rewritten on every full run (False
    before any stop, and until the gate's audit record is written), together with
    `access_expires_at` (when an allowed sign-in turns stale; None otherwise). The
    output fragment and the review callbacks check both via `_access_ok`, and the
    download callable its own captured expiry, since they run without passing
    through here. `st.login` is only ever a button callback, never called on render.
    """
    decision = decide_access(
        _secrets_snapshot(),
        claims=st.user.to_dict(),
        allow_anonymous=anonymous_opt_out(os.environ),
        opt_out_in_dotenv=_dotenv_has_opt_out(),
        trusted_headers=bool(st.get_option("server.trustedUserHeaders")),
        authlib_installed=_authlib_installed(),
        now=time.time(),
    )
    allowed = decision.kind in ("allow", "anonymous")
    # Closed until the gate's audit record is written: if it cannot be, nothing opens.
    st.session_state["access_ok"] = False
    st.session_state["access_expires_at"] = decision.expires_at
    _audit_gate(decision)
    st.session_state["access_ok"] = allowed

    if decision.kind == "anonymous":
        st.warning(ANONYMOUS_MODE, icon=":material/no_accounts:")
    elif decision.kind == "blocked":
        # Visitors see no configuration detail; the operator gets the problem (key
        # names only) on stderr, once per session rather than on every rerun.
        problem = decision.problem or ""
        if st.session_state.get("access_problem_logged") != problem:
            report_problem(problem)
            st.session_state["access_problem_logged"] = problem
        st.error(SIGN_IN_UNAVAILABLE, icon=_ICON_ERROR)
    elif decision.kind == "login":
        st.info(
            SESSION_EXPIRED if decision.reauth else SIGN_IN_PROMPT,
            icon=":material/lock:",
        )
        for provider in decision.providers:
            st.button(
                _sign_in_label(provider),
                key=f"sign_in_{provider or 'default'}",
                icon=":material/login:",
                on_click=st.login,
                args=(provider,),
            )
    elif decision.kind == "deny":
        st.error(_deny_message(decision), icon=_ICON_ERROR)
        _sign_out_button()
    return decision if allowed else None


def _audit_gate(decision: Decision) -> None:
    """Record this session's audit identity, and log the gate's outcome once per kind.

    The actor is rewritten on every full run (which also refreshes its class after a
    hot reload); the event — session_start, or access_denied with its reason — is
    logged only when the decision's kind differs from the last one logged, so reruns
    add nothing. The sign-in screen is not logged: `st.login` redirects into a new
    session, whose session_start is the login record.
    """
    session = st.session_state.setdefault("audit_session", uuid.uuid4().hex)
    actor = audit.actor_for(decision, session)
    st.session_state["audit_actor"] = actor
    if st.session_state.get("audit_gate_state") != decision.kind:
        event = audit.gate_event(decision)
        if event is not None:
            audit.emit(actor, event)
        st.session_state["audit_gate_state"] = decision.kind


def _audit_actor() -> audit.Actor:
    """This session's audit identity, as the access gate last recorded it.

    Raises rather than falling back to anonymous: a missing actor means the gate never
    ran for this session, and a silent fallback would hide exactly that. Not
    isinstance-checked, since a hot reload redefines the class (`audit.emit` reads it
    by attribute and re-validates it).
    """
    actor = st.session_state.get("audit_actor")
    if actor is None:
        raise RuntimeError("No audit actor for this session")
    return actor


# Session-state keys holding PHI (or keyed to it) that sign-out removes, plus the
# signed-in audit identity and review-audit record, which must not outlive the sign-in.
_PURGED_KEYS = frozenset(
    {
        "responses",
        "audio_sources",
        "run_id",
        "run_model",
        "keyterms",
        "audit_actor",
        "audit_review",
    }
)
_PURGED_PREFIXES = ("transcript_", "reviewed_", "uploads_", "recording_")


def _purge_session() -> None:
    """Drop this session's PHI (results, review state, inputs, keyterms) and its
    audit identity.

    Deletes the keys outright — the sign-out rerun stops at the gate, and Streamlit
    purges an unrendered widget's state only after a run that completes — then
    rotates `input_nonce` so the uploader and recorder come back as new, empty
    widgets, and clears `access_ok` so nothing renders before the gate runs again.
    """
    for key in list(st.session_state):
        if isinstance(key, str) and (
            key in _PURGED_KEYS or key.startswith(_PURGED_PREFIXES)
        ):
            del st.session_state[key]
    st.session_state["input_nonce"] = st.session_state.get("input_nonce", 0) + 1
    st.session_state["access_ok"] = False


def _sign_out() -> None:
    """Sign-out button callback: log it, purge this session's PHI, end the sign-in.

    The purge and `st.logout` run even if the audit line or the purge fails —
    signing out must never depend on anything else. `st.logout` clears `st.user` and
    redirects through the identity provider's logout (when it has one) into a new
    session.
    """
    try:
        audit.emit(_audit_actor(), audit.logout())
    finally:
        try:
            _purge_session()
        finally:
            st.logout()


def _sign_out_button() -> None:
    """The Sign out button, with the shared-workstation caveat under it."""
    st.button("Sign out", key="sign_out", icon=":material/logout:", on_click=_sign_out)
    st.caption(SIGN_OUT_NOTE)


def _account_panel(email: str) -> None:
    """Sidebar account block: who is signed in, and Sign out.

    The email is `st.text`, not part of the caption: Markdown would autolink it
    (a `mailto:` in link color) and the caption's 60% opacity then dims that link
    below WCAG AA. Plain text renders it verbatim, unlinked, as full-opacity body
    text — so it needs no Markdown escaping either.
    """
    st.caption(":material/account_circle: Signed in as")
    st.text(email)
    _sign_out_button()


def _completion_toast(n_ok: int, total: int) -> None:
    """Fire a one-shot toast summarizing the finished batch."""
    if n_ok == total:
        st.toast(f"Transcribed {n_ok} item(s)", icon=":material/check_circle:")
    elif n_ok:
        st.toast(
            f"Transcribed {n_ok}/{total}; {total - n_ok} failed",
            icon=_ICON_WARNING,
        )
    else:
        st.toast(f"All {total} item(s) failed", icon=_ICON_ERROR)


def _transcribe_batch(
    api_key: str,
    items: list[tuple[str, dict[str, Any]]],
    method: str,
    **opts: Any,
) -> int:
    """Transcribe a batch via the shared core, owning the Streamlit-side concerns.

    Thin UI adapter over `nova.transcribe.transcribe_batch`: it builds the playback
    sources up front, drives a live `st.status` progress region (label + bar), renders
    one `st.error` per failed item, writes results to session state under a fresh
    `run_id`, and fires a completion `st.toast`. Returns the number of successful
    items. `**opts` is the `_feature_opts` dict (its keys match `build_options`
    exactly); the module-global `DeepgramClient`/`as_completed` are passed as seams so
    the existing test patch points keep intercepting them.
    """
    options = build_options(**opts)
    total = len(items)
    sources = {
        i: _playback_source(kwargs["request"]) for i, (_, kwargs) in enumerate(items)
    }

    with st.status(f"Transcribing 0/{total}...", expanded=True) as status:
        progress = st.progress(0.0)

        def _on_progress(done: int, t: int) -> None:
            progress.progress(done / t)
            status.update(label=f"Transcribing {done}/{t}...")

        results = transcribe_batch(
            api_key,
            items,
            method,
            options=options,
            client_cls=DeepgramClient,
            as_completed_fn=as_completed,
            on_progress=_on_progress,
        )
        ok = [r for r in results if r.error is None]
        status.update(
            label=f"Transcribed {len(ok)}/{total}", state="complete", expanded=False
        )

    for r in results:
        if r.error is not None:
            st.error(
                f"Transcription failed for {r.label}: {r.error}",
                icon=_ICON_ERROR,
            )

    # Always overwrite (even when empty) so a fully-failed run clears stale results.
    st.session_state["responses"] = [(r.label, r.response) for r in ok]
    st.session_state["audio_sources"] = [sources[r.index] for r in ok]
    # A fresh id per Run — also unconditional, so a fully-failed run rotates it too.
    # The review widgets key on it, so a new batch starts with fresh editors and
    # unchecked Reviewed boxes (Streamlit purges the old run's keys at the end of
    # this full rerun, since nothing renders them again).
    st.session_state["run_id"] = uuid.uuid4().hex
    # The model this batch was sent to, so the panel and the export can name it even
    # after the sidebar selection changes.
    st.session_state["run_model"] = options["model"]

    _completion_toast(len(ok), total)
    return len(ok)


def _process_inputs(api_key: str, files: list[tuple[str, bytes]], **opts) -> int:
    """Transcribe files with a shared client, store results; return the success count."""
    items = [(name, {"request": data}) for name, data in files]
    return _transcribe_batch(api_key, items, "transcribe_file", **opts)


def _feature_opts() -> dict[str, Any]:
    """Read the sidebar Features form's control values from session state."""
    return {
        "model": st.session_state.get("model", DEFAULT_MODEL),
        "keyterms": st.session_state.get("keyterms", []),
        "language": st.session_state.get("language", DEFAULT_LANGUAGE),
        "smart_format": st.session_state.get("smart_format", DEFAULT_SMART_FORMAT),
        "dictation": st.session_state.get("dictation", DEFAULT_DICTATION),
        "measurements": st.session_state.get("measurements", DEFAULT_MEASUREMENTS),
        "diarize": st.session_state.get("diarize", DEFAULT_DIARIZE),
        "redact": st.session_state.get("redact", []),
    }


def _run(api_key: str, uploaded_files: list, recording: Any) -> None:
    """Validate and transcribe whichever input is provided (priority: upload, record).

    Audited: one `transcription_run` line per batch sent, or one
    `transcription_rejected` line when validation refuses it. The actor and the
    options' audit projection are resolved first, so a missing actor or an unknown
    option fails before any audio leaves the app. Only counts, flags and option
    names reach the audit builders — never a filename, error text, keyterm or
    transcript.
    """
    actor = _audit_actor()
    present = [
        name
        for name, ok in (
            ("Upload", bool(uploaded_files)),
            ("Record", recording is not None),
        )
        if ok
    ]
    if len(present) > 1:
        chosen, *ignored = present
        st.info(
            f"Multiple inputs detected; transcribing {chosen} and ignoring "
            f"{', '.join(ignored)} (priority: Upload > Record).",
            icon=":material/info:",
        )
    opts = _feature_opts()
    run_opts = audit.run_options(**opts)
    for message in option_warnings(**opts):
        st.warning(message, icon=_ICON_WARNING)
    if uploaded_files:
        if len(uploaded_files) > MAX_UPLOADS:
            st.error(
                f"Too many files. Maximum is {MAX_UPLOADS} per batch.",
                icon=_ICON_ERROR,
            )
            audit.emit(
                actor,
                audit.transcription_rejected(
                    input_kind="upload",
                    reason="too_many_files",
                    n_items=len(uploaded_files),
                ),
            )
            return
        # Backstop: Streamlit already 413s uploads over server.maxUploadSize
        # (== MAX_FILE_SIZE) before they reach _run, so this fires only if those
        # two limits ever drift apart.
        oversized = [f.name for f in uploaded_files if f.size > MAX_FILE_SIZE]
        if oversized:
            st.error(
                f"Skipped (exceeds {MAX_FILE_SIZE // (1024 * 1024)} MB): "
                f"{', '.join(oversized)}",
                icon=_ICON_ERROR,
            )
        valid = [
            (f.name, f.getvalue()) for f in uploaded_files if f.size <= MAX_FILE_SIZE
        ]
        if valid:
            n_ok = _process_inputs(api_key, valid, **opts)
            audit.emit(
                actor,
                audit.transcription_run(
                    run=st.session_state["run_id"],
                    input_kind="upload",
                    n_items=len(valid),
                    n_ok=n_ok,
                    n_skipped=len(oversized),
                    options=run_opts,
                ),
            )
        else:
            audit.emit(
                actor,
                audit.transcription_rejected(
                    input_kind="upload",
                    reason="all_oversize",
                    n_items=len(uploaded_files),
                ),
            )
    elif recording is not None:
        audio_bytes = recording.getvalue()
        try:
            with wave.open(io.BytesIO(audio_bytes)) as wf:
                framerate = wf.getframerate()
                if not framerate:
                    raise wave.Error("zero framerate")
                duration = wf.getnframes() / framerate
        except (wave.Error, EOFError):
            st.error("Could not read the recording.", icon=_ICON_ERROR)
            audit.emit(
                actor,
                audit.transcription_rejected(
                    input_kind="record", reason="recording_unreadable", n_items=1
                ),
            )
            return
        if duration > MAX_RECORDING_SECONDS:
            st.error(
                f"Recording exceeds the {MAX_RECORDING_SECONDS // 60}-minute limit.",
                icon=_ICON_ERROR,
            )
            audit.emit(
                actor,
                audit.transcription_rejected(
                    input_kind="record", reason="recording_too_long", n_items=1
                ),
            )
        else:
            n_ok = _process_inputs(api_key, [("Recording", audio_bytes)], **opts)
            audit.emit(
                actor,
                audit.transcription_run(
                    run=st.session_state["run_id"],
                    input_kind="record",
                    n_items=1,
                    n_ok=n_ok,
                    n_skipped=0,
                    options=run_opts,
                ),
            )


def _display_audio(name: str, source: bytes) -> None:
    """Render an audio player for a transcribed upload or recording (MIME from its name)."""
    mime = _AUDIO_MIME.get(os.path.splitext(name)[1].lower(), "audio/wav")
    st.audio(source, format=mime)


def _result_metrics(response: Any) -> tuple[float | None, float | None]:
    """Extract (duration_seconds, confidence) for a response; either may be None.

    Confidence is the alternative-level value Deepgram reports for the transcript.
    Both reads are getattr/type-guarded so a results-less response yields (None, None).
    """
    duration = getattr(getattr(response, "metadata", None), "duration", None)
    alt = _first_alternative(response)
    confidence = getattr(alt, "confidence", None) if alt is not None else None
    return (
        duration if isinstance(duration, (int, float)) else None,
        confidence if isinstance(confidence, (int, float)) else None,
    )


def _display_metrics(response: Any) -> None:
    """Render Duration / Confidence / Low-confidence words cards, when available.

    The low-confidence count is shown only when highlighting is available (so a
    missing card never reads as "0 words to check").
    """
    duration, confidence = _result_metrics(response)
    flagged = _low_confidence_count(response)
    if duration is None and confidence is None and flagged is None:
        return
    with st.container(horizontal=True):
        if duration is not None:
            st.metric("Duration", f"{duration:.1f} s", border=True)
        if confidence is not None:
            st.metric("Confidence", f"{confidence * 100:.1f}%", border=True)
        if flagged is not None:
            st.metric("Low-confidence words", str(flagged), border=True)


def _flagged_markdown(tokens: list[Token]) -> str:
    """Join tokens into escaped Markdown, low-confidence ones as `:orange[**…**]`.

    Bold is the non-color cue (WCAG 1.4.1), and the flag carries no background, so it
    cannot be mistaken for a `:{color}-background[**Speaker N:**]` label.
    """
    return " ".join(
        f":orange[**{_escape_markdown(t.text)}**]"
        if t.low_confidence
        else _escape_markdown(t.text)
        for t in tokens
    )


def _display_transcript(response: Any) -> None:
    """Render one result's metrics then transcript (Markdown-escaped so it shows verbatim).

    The transcript is rebuilt from Deepgram's words so low-confidence ones can be
    flagged in bold orange, under a caption that says whether any were. With
    diarization, one color-highlighted labeled line per speaker run (1-based, so the
    first speaker reads "Speaker 1", colored by speaker index); otherwise one line per
    paragraph. When the words cannot reproduce the transcript (or are missing), the
    plain transcript renders unhighlighted under a caption saying so; a response with
    no results gets a notice instead.
    """
    _display_metrics(response)
    runs = _flagged_runs(response)
    if runs is None:
        transcript = _transcript_text(response)
        if transcript is None:
            st.caption(NO_TRANSCRIPT)
            return
        if transcript.strip():
            st.caption(NO_CONFIDENCE)
            st.markdown(_escape_markdown(transcript))
        return
    flagged = any(t.low_confidence for _, tokens in runs for t in tokens)
    st.caption(LOW_CONFIDENCE_LEGEND if flagged else NO_FLAGS)
    for speaker, tokens in runs:
        body = _flagged_markdown(tokens)
        if speaker is None:
            st.markdown(body)
            continue
        color = _SPEAKER_COLORS[speaker % len(_SPEAKER_COLORS)]
        st.markdown(
            f":{color}-background[**Speaker {_speaker_label(speaker)}:**] {body}"
        )


def _plain_transcript(response: Any) -> str:
    """Plain-text transcript for export: diarized 'Speaker N: ...' lines, else flat text."""
    segments = _diarized_segments(response)
    if segments:
        lines = []
        for speaker, text in segments:
            lines.append(f"Speaker {_speaker_label(speaker)}: {text}")
        return "\n".join(lines)
    return _transcript_text(response) or ""


# Review / sign-off. Widget keys carry only the run id and the result's position —
# never a filename, since a key becomes an `st-key-*` class in the page's DOM.
def _text_key(run_id: str, index: int) -> str:
    """Session-state key of result `index`'s export editor (a `st.text_area`)."""
    return f"transcript_{run_id}_{index}"


def _reviewed_key(run_id: str, index: int) -> str:
    """Session-state key of result `index`'s "Reviewed" checkbox."""
    return f"reviewed_{run_id}_{index}"


def _is_reviewed(run_id: str, index: int) -> bool:
    """True only when the checkbox value is exactly True (identity, not truthiness)."""
    return st.session_state.get(_reviewed_key(run_id, index)) is True


def _model_name(model: object) -> str | None:
    """The display name of a run's model, or None when unknown (no run yet)."""
    return _MODELS.get(model) if isinstance(model, str) else None


def _export_blob(named_texts: list[tuple[str, str]], model_name: str | None) -> str:
    """The downloaded file: a `Model:` line (when known), then one `name` line and its
    text per result, blank-line separated."""
    blocks = [f"{name}\n{text}" for name, text in named_texts]
    if model_name is not None:
        blocks.insert(0, f"Model: Deepgram {model_name}")
    return "\n\n".join(blocks)


def _deferred_export(
    named_texts: list[tuple[str, str]],
    actor: audit.Actor,
    event: audit.AuditEvent,
    expires_at: float | None,
    model_name: str | None,
) -> Callable[[], str]:
    """Zero-arg `data` callable for the download button, over values captured now.

    Streamlit runs it on every click, on a worker thread with no ScriptRunContext, so
    it must never touch `st.*` (session state included): it closes over plain
    strings (the model's display name included), the audit actor, the
    already-validated `transcript_downloaded` event, and the sign-in's expiry, all
    captured at render. A click after that expiry is
    refused — the button can outlive it on an idle page, and a click reruns
    nothing. Otherwise it logs the event, then returns the file — one audit line per
    file generated. It raises only on expiry or if the line cannot be written, both
    failing the download closed; Streamlit logs a failing callable with its
    traceback, so every exception on this path must be PHI-free (`DOWNLOAD_EXPIRED`
    is fixed text, and an `AuditSchemaError` names fields, never values).
    """
    captured = [(str(name), str(text)) for name, text in named_texts]

    def _build() -> str:
        if _expired(expires_at):
            raise PermissionError(DOWNLOAD_EXPIRED)
        audit.emit(actor, event)
        return _export_blob(captured, model_name)

    return _build


def _expired(expires_at: object) -> bool:
    """Whether a sign-in expiry (epoch seconds) has passed; None never expires (an
    anonymous session). Anything but a number — or NaN — counts as expired."""
    if expires_at is None:
        return False
    return not (
        isinstance(expires_at, int | float)
        and not isinstance(expires_at, bool)
        and time.time() <= expires_at
    )


def _access_ok() -> bool:
    """Whether this session passed the gate on its last full run (identity, not
    truthiness) and that sign-in has not expired since. Fragment reruns and widget
    callbacks skip the gate — an open page can go on using them for hours — so they
    check this instead."""
    return st.session_state.get("access_ok") is True and not _expired(
        st.session_state.get("access_expires_at")
    )


def _audit_review(run_id: str, index: int) -> None:
    """Log a change in result `index`'s Reviewed state — once per actual change.

    Compares the flag with the last state this run's audit trail recorded
    (`audit_review`: the run id and the indexes logged as signed off), so the log
    follows the state, not the callbacks. That matters when an edit and a check land
    in the same rerun: Streamlit does not guarantee the two callbacks' order (1.64
    runs the editor's first), and `_on_edit` then clears the check. Checkbox-first
    logs signed-off then reopened; editor-first leaves the flag where it started and
    logs nothing — never a "reopened" for a sign-off that was never recorded.
    `edited` compares the editor's text with Deepgram's plain transcript.
    """
    responses = st.session_state.get("responses", [])
    if not 0 <= index < len(responses):
        return  # not a result of this session (unreachable through the UI)
    recorded = st.session_state.get("audit_review")
    signed: frozenset[int] = (
        recorded[1]
        if isinstance(recorded, tuple) and recorded[0] == run_id
        else frozenset()
    )
    reviewed = _is_reviewed(run_id, index)
    if reviewed == (index in signed):
        return
    response = responses[index][1]
    if reviewed:
        text = st.session_state.get(_text_key(run_id, index))
        event = audit.review_signed_off(
            run=run_id,
            result_index=index,
            n_results=len(responses),
            edited=isinstance(text, str) and text != _plain_transcript(response),
            n_flagged=_low_confidence_count(response),
        )
    else:
        event = audit.review_reopened(
            run=run_id, result_index=index, n_results=len(responses)
        )
    audit.emit(_audit_actor(), event)
    st.session_state["audit_review"] = (
        run_id,
        signed | {index} if reviewed else signed - {index},
    )


def _on_edit(run_id: str, index: int) -> None:
    """Editor `on_change`: an applied edit clears that result's Reviewed flag, and
    says so.

    While Reviewed is checked the editor renders disabled, and Streamlit then
    discards any incoming value and skips this callback — so it fires with the flag
    set only when an edit and a check land in the same rerun. In a browser that is
    the ordinary "type a correction, then click Reviewed" path: a text area commits
    on blur, and the click is the blur. Clearing is what keeps the edit. Left set,
    the flag would render the editor disabled in this very run, and Streamlit drops
    a disabled widget's incoming value when it registers — the box would stay
    checked over the pre-edit text, the correction silently gone. So the check is
    undone (audited if the sign-off already was) and a toast asks for it again. A
    no-op for a session that has not passed the gate.
    """
    if not _access_ok():
        return
    key = _reviewed_key(run_id, index)
    if st.session_state.get(key) is True:
        st.session_state[key] = False
        _audit_review(run_id, index)
        st.toast(REVIEW_CLEARED, icon=":material/edit_note:")


def _on_review(run_id: str, index: int) -> None:
    """Reviewed-checkbox `on_change`: audit the sign-off, or its reopening. Like
    `_on_edit`, it does nothing for a session that has not passed the gate."""
    if not _access_ok():
        return
    _audit_review(run_id, index)


def _review_controls(
    run_id: str, index: int, response: Any, total: int
) -> tuple[str, bool]:
    """The export editor and its Reviewed checkbox for one result; returns both values.

    The editor is seeded (via `value=`, never session state) with the plain
    transcript and holds what Download saves. It renders disabled while Reviewed is
    checked, so no edit can be typed after sign-off and then miss the download
    (a text area commits only on blur or Ctrl/⌘+Enter); a disabled widget stays
    registered, so it keeps its value. Both always render — an unrendered keyed
    widget loses its value. With several results, labels carry the position (never
    the filename).

    Returns `(text, reviewed)` from the widgets themselves, which the download gate
    must use rather than reading the keys from session state before they render:
    Streamlit drops an incoming value for a disabled widget (a stale UI's late edit,
    or a forged message) only when the widget registers, so a read taken above it
    would still see that unreviewed text.
    """
    position = f" ({index + 1} of {total})" if total > 1 else ""
    text = st.text_area(
        f"{EDIT_LABEL}{position}",
        value=_plain_transcript(response),
        key=_text_key(run_id, index),
        height="content",
        help=EDIT_HELP,
        on_change=_on_edit,
        args=(run_id, index),
        disabled=_is_reviewed(run_id, index),
    )
    reviewed = st.checkbox(
        f"{REVIEWED_LABEL}{position}",
        key=_reviewed_key(run_id, index),
        on_change=_on_review,
        args=(run_id, index),
    )
    return text, reviewed is True


def _transcript_download(
    responses: list[tuple[str, Any]], reviews: list[tuple[str, bool]], run_id: str
) -> None:
    """Plain-text download of the batch's edited transcripts, gated on review.

    `reviews` holds each result's `(editor text, reviewed)` as `_review_controls`
    returned them this run. Locked until every result is reviewed, with the label
    showing progress. While locked the button carries `data=""`, not the
    transcripts: `disabled` is enforced only in the browser on the deferred-file
    path, so this is the server-side half of the gate. Unlocked, the blob is a
    deferred callable over the editors' text captured at this render (assembled only
    on click) that also logs `transcript_downloaded` — its actor and event (the run,
    and how many results were edited) and the sign-in's expiry are captured here,
    since the callable runs without session context. `on_click="ignore"` skips the
    pointless rerun a click would otherwise trigger.
    """
    if not responses:
        return
    n = len(responses)
    done = sum(1 for _, reviewed in reviews if reviewed is True)
    unlocked = len(reviews) == n and done == n
    data: Callable[[], str] | str = ""
    if unlocked:
        names = [name for name, _ in responses]
        texts = [text for text, _ in reviews]
        n_edited = sum(
            1
            for (_, response), text in zip(responses, texts, strict=True)
            if text != _plain_transcript(response)
        )
        event = audit.transcript_downloaded(run=run_id, n_results=n, n_edited=n_edited)
        data = _deferred_export(
            list(zip(names, texts, strict=True)),
            _audit_actor(),
            event,
            st.session_state.get("access_expires_at"),
            _model_name(st.session_state.get("run_model")),
        )
    st.download_button(
        DOWNLOAD_LABEL if unlocked else f"Download locked — {done}/{n} reviewed",
        data,
        file_name="transcripts.txt",
        mime="text/plain",
        icon=":material/download:",
        key="download_transcripts",
        disabled=not unlocked,
        help=None if unlocked else DOWNLOAD_LOCKED_HELP,
        on_click="ignore",
    )


def _output_panel(
    responses: list[tuple[str, Any]],
    audio_sources: list[bytes | None],
    run_id: str,
) -> list[tuple[str, bool]]:
    """Render results in a fixed-height panel; return each result's review state.

    Empty -> placeholder. Single result -> player pinned above the scroll container.
    Multiple -> one labeled, divided block per result inside the container. A source
    of None (a large upload dropped from playback) renders a caption in place of the
    player. Each result's highlighted view is followed by its review controls, whose
    `(text, reviewed)` values are returned in result order (empty when there are no
    results).
    """
    if not responses:
        with st.container(height=OUTPUT_HEIGHT, border=True):
            st.caption(PLACEHOLDER)
        return []

    total = len(responses)
    if total == 1:
        (name, response), source = responses[0], audio_sources[0]
        if source is not None:
            _display_audio(name, source)
        else:
            st.caption(PLAYBACK_TOO_LARGE)
        with st.container(height=OUTPUT_HEIGHT, border=True):
            _display_transcript(response)
            return [_review_controls(run_id, 0, response, total)]

    reviews = []
    with st.container(height=OUTPUT_HEIGHT, border=True):
        for i, ((name, response), source) in enumerate(
            zip(responses, audio_sources, strict=True)
        ):
            if i:
                st.divider()
            st.markdown(f"**{_escape_markdown(name)}**")
            if source is not None:
                _display_audio(name, source)
            else:
                st.caption(PLAYBACK_TOO_LARGE)
            _display_transcript(response)
            reviews.append(_review_controls(run_id, i, response, total))
    return reviews


PLACEHOLDER = ":material/graphic_eq: Select audio, then click Run in the sidebar to see the response here."
NO_TRANSCRIPT = "No transcript in this response."
# Low-confidence captions. Plain caption text on purpose, no color directive: at
# caption opacity (60%) a colored sample would fall to ~2.8:1 in light mode.
LOW_CONFIDENCE_LEGEND = (
    ":material/flag: Words in **bold orange** scored below "
    f"{LOW_CONFIDENCE_THRESHOLD:.0%} model confidence — check them against the audio. "
    "Unmarked words can still be wrong."
)
NO_FLAGS = (
    f"No words scored below {LOW_CONFIDENCE_THRESHOLD:.0%} model confidence. "
    "Unmarked words can still be wrong — check against the audio."
)
NO_CONFIDENCE = (
    "Low-confidence highlighting isn't available for this result — "
    "review every word against the audio."
)
PLAYBACK_TOO_LARGE = "Inline playback unavailable for files over 25 MB."
# Review / sign-off copy. The editor (not the highlighted view) is what Download saves.
EDIT_LABEL = "Transcript to export"
EDIT_HELP = (
    "Correct errors here — this text, not the highlighted view above, is what "
    "Download saves. Press Ctrl/⌘+Enter (or click elsewhere) to apply an edit "
    "before checking Reviewed; an edit applied by the Reviewed click itself clears "
    "the check. Checking Reviewed locks this text; uncheck Reviewed to edit again."
)
# Toasted when an applied edit clears a Reviewed check (see _on_edit).
REVIEW_CLEARED = "Edit applied, so Reviewed was cleared — check it again to sign off."
REVIEWED_LABEL = "Reviewed against the audio"
DOWNLOAD_LABEL = "Download transcript"
DOWNLOAD_LOCKED_HELP = (
    'Mark every transcript "Reviewed against the audio" to enable download.'
)
# Sign-in copy. A blocked deploy shows visitors no configuration detail (the operator
# gets the problem on stderr); the anonymous banner is detailed because only an
# operator who deliberately opted out of sign-in ever sees it.
SIGN_IN_PROMPT = "Sign in with your work account to use this app."
SESSION_EXPIRED = "Your sign-in has expired. Sign in again to continue."
# Raised by a download clicked after the sign-in expired. Streamlit logs it with a
# traceback, so it is fixed text — never a filename or transcript.
DOWNLOAD_EXPIRED = "Download refused: the sign-in has expired"
SIGN_IN_UNAVAILABLE = "Sign-in is unavailable. Contact your administrator."
ANONYMOUS_MODE = (
    f"**Anonymous mode** — sign-in is off because `{ALLOW_ANONYMOUS_ENV}=1` is set in "
    "the environment. For local development only: never use it with real patient audio."
)
# Why a signed-in account was refused, by nova.access DenyReason; `{email}` is the
# account's normalized, Markdown-escaped email.
DENY_MESSAGES = {
    "email_missing": (
        "Your sign-in didn't include a usable email address, so access can't be "
        "checked."
    ),
    "email_unverified": (
        "{email} isn't a verified email address at your identity provider."
    ),
    "domain_not_allowed": "{email} is not authorized to use this app.",
}
DENY_SUFFIX = (
    " Sign out and sign in with an authorized work account, or contact your "
    "administrator."
)
SIGN_OUT_NOTE = (
    "Signing out here may not sign you out of your identity provider — on a shared "
    "computer, sign out there too."
)


@st.fragment
def _render_output() -> None:
    """Render the transcript output (header, download button, panel) as a fragment.

    Wrapped in `st.fragment` so a review interaction (an applied edit, a Reviewed
    check) reruns only this panel — not the whole script (the input tabs and the
    Features form). Results are read from session state, so each fragment rerun
    reflects the latest batch; a Run (in the Features form, outside this fragment)
    triggers a full rerun that refreshes it under a new `run_id`.

    Renders nothing unless this session passed the access gate and its sign-in has
    not expired since: a fragment rerun skips the gate, and a stopped run leaves the
    fragment registered. Once an admitted sign-in expires, the next interaction here
    reruns the whole app instead, whose gate then asks for a fresh sign-in.
    """
    if not _access_ok():
        if st.session_state.get("access_ok") is True:  # admitted, but now expired
            st.rerun(scope="app")
        return
    responses = st.session_state.get("responses", [])
    audio_sources = st.session_state.get("audio_sources", [])
    run_id = st.session_state.get("run_id", "")
    model_name = _model_name(st.session_state.get("run_model")) if responses else None
    st.caption(
        ":material/description: Transcript"
        + (f" · {_escape_markdown(model_name)}" if model_name else "")
    )
    # The download row sits above the panel but is filled after it: the gate reads
    # the review widgets' own return values, which exist only once they render.
    download_row = st.empty()
    reviews = _output_panel(responses, audio_sources, run_id)
    with download_row:
        _transcript_download(responses, reviews, run_id)


st.set_page_config(
    page_title="Deepgram Clinical Transcription",
    page_icon="🩺",
    layout="wide",
)

st.title("Deepgram Clinical Transcription")
st.caption(
    "Transcribe clinical audio with Deepgram's Nova-3 Medical or Pharma model — "
    "speaker labels, measurement formatting, and PII/PHI redaction."
)

# Access gate — before anything that reads audio, the API key, or results. It renders
# its own refusal (sign-in buttons, a denial, or "unavailable"); st.stop() then ends
# the run. In a bare-mode import (pytest) st.stop() is a no-op, so the code below must
# tolerate access_decision being None.
access_decision = _access_gate()
if access_decision is None:
    st.stop()

# Environment first (`.env` via _load_dotenv), then `st.secrets` for deployed hosts.
api_key = os.environ.get("DEEPGRAM_API_KEY", "") or _secret_api_key()
if not api_key:
    st.warning(
        "Deepgram API key required. Get a free key at https://deepgram.com.",
        icon=":material/key:",
    )
    api_key = st.text_input(
        "Deepgram API key",
        type="password",
        placeholder="Paste your Deepgram API key to continue",
        label_visibility="collapsed",
    )

# Inputs and output sit side by side so a wide display shows the audio source and its
# transcript at once — no scrolling past a full-width uploader to reach the result — and
# transcript lines keep a readable length. Columns stack (inputs first) on narrow viewports.
input_col, output_col = st.columns(INPUT_OUTPUT_RATIO, gap="medium")

# The input widgets are keyed by a nonce that sign-out rotates, so they come back
# empty (the uploader drops its files in the browser too).
input_nonce = st.session_state.get("input_nonce", 0)

with input_col:
    tab_upload, tab_record = st.tabs(
        [":material/upload: Upload", ":material/mic: Record"]
    )

    with tab_upload:
        uploaded_files = st.file_uploader(
            "Upload audio files",
            type=_AUDIO_TYPES,
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=f"uploads_{input_nonce}",
        )

    with tab_record:
        recording = st.audio_input(
            "Record audio",
            label_visibility="collapsed",
            key=f"recording_{input_nonce}",
        )


# Features live in the sidebar — the canonical home for app-level settings — so the
# main area is left to the audio inputs and the transcript output. The form batches
# feature edits so they don't rerun until Run is clicked.
with st.sidebar:
    # Who is signed in, and Sign out — only for a signed-in (not anonymous) session.
    if (
        access_decision is not None
        and access_decision.kind == "allow"
        and access_decision.email
    ):
        _account_panel(access_decision.email)
    st.caption(":material/tune: Transcription settings")
    with st.form("features", border=False):
        st.selectbox(
            "Model",
            options=list(_MODELS),
            format_func=lambda model: _MODELS[model],
            help="Medical suits clinician–patient encounters and dictation. Pharma is tuned for drug names — pharmacy calls and refill requests. Both accept the same languages.",
            key="model",
        )
        st.selectbox(
            "Language",
            options=list(_LANGUAGES),
            format_func=lambda code: _LANGUAGES[code],
            key="language",
        )
        st.multiselect(
            "Keyterm Prompting",
            options=[],
            accept_new_options=True,
            max_selections=MAX_KEYTERMS,
            placeholder="Add keyterms...",
            help="Boosts recognition of important words or phrases, like names, product terms, or jargon. The model pays extra attention to these; you can include up to 100 keyterms per request.",
            key="keyterms",
        )
        st.toggle(
            "Smart Format",
            value=DEFAULT_SMART_FORMAT,
            help="Smart Format improves readability by applying additional formatting. When enabled, punctuation and paragraph breaks will be applied as well as formatting of other entities, such as dates, times, and numbers.",
            key="smart_format",
        )
        st.toggle(
            "Diarize",
            value=DEFAULT_DIARIZE,
            help="Detects speaker changes and labels turns as Speaker 1, Speaker 2, … in the transcript. Speakers are numbered, not named by role. Use for clinician–patient encounters, with Dictation off.",
            key="diarize",
        )
        st.toggle(
            "Dictation",
            value=DEFAULT_DICTATION,
            help='Converts spoken formatting commands into characters (e.g. "period" becomes ".", "new paragraph" starts a new line). Automatically enables punctuation. For a single clinician dictating — turn off for encounters, where patient speech could be converted too.',
            key="dictation",
        )
        st.toggle(
            "Measurements",
            value=DEFAULT_MEASUREMENTS,
            help='Converts spoken measurements into abbreviated units (e.g. "five milligrams" becomes "5 mg"). Note: volumes come out as lowercase "ml" and "l", which ISMP lists as error-prone (use mL and L) — review volumes before clinical use.',
            key="measurements",
        )
        st.multiselect(
            "Redact",
            options=list(_REDACT_GROUPS),
            format_func=lambda group: _REDACT_GROUPS[group],
            placeholder="Select information to redact...",
            help='Replaces the selected information with redaction tags in the transcript. For de-identification, use PII (names, locations, IDs). Note: PHI redaction strips clinical content itself (conditions, drugs, injuries), and Numbers redaction removes any run of 3+ digits plus number-like entities (e.g. dates, times, ages, medical statistics, locations), so it redacts clinical values unpredictably ("500 mg" always, shorter values sometimes) — usually the opposite of what a medical transcript should keep.',
            key="redact",
        )
        has_input = bool(uploaded_files or recording is not None)
        run_clicked = st.form_submit_button(
            "Run",
            type="primary",
            icon=":material/graphic_eq:",
            disabled=not api_key or not has_input,
            help=None
            if (api_key and has_input)
            else "Add an API key and an upload or recording to enable transcription.",
            width="stretch",
        )

# Run feedback (validation callouts, the progress status, per-item errors) lands under
# the inputs that produced it, keeping the output column's top edge fixed.
if run_clicked:
    with input_col:
        _run(api_key, uploaded_files, recording)

with output_col:
    _render_output()
