"""Transcription constants — the single source of truth for the Streamlit UI."""

# Deepgram `model` values -> display names (also shown on each result and in the export).
# Medical is the default: Pharma is tuned for drug names (pharmacy calls, refills), so
# general clinical encounters stay on Medical.
MODELS = {
    "nova-3-medical": "Nova-3 Medical",
    "nova-3-pharma": "Nova-3 Pharma",
}
DEFAULT_MODEL = next(iter(MODELS))

# Both models accept exactly these English variants (Pharma is English-only; Medical's
# `multi` code-switching is not offered), so the list does not depend on the model.
LANGUAGES = {
    "en": "English",
    "en-US": "English (US)",
    "en-AU": "English (Australia)",
    "en-CA": "English (Canada)",
    "en-GB": "English (UK)",
    "en-IE": "English (Ireland)",
    "en-IN": "English (India)",
    "en-NZ": "English (New Zealand)",
}

# Feature defaults — shared by the UI widgets and option builders so they cannot drift.
DEFAULT_LANGUAGE = next(iter(LANGUAGES))
DEFAULT_SMART_FORMAT = True
DEFAULT_DICTATION = False
# Off: Deepgram abbreviates volumes as lowercase "ml" / "l", which ISMP lists as
# error-prone ("2 l" reads as 21; use mL / L), so abbreviation stays opt-in.
DEFAULT_MEASUREMENTS = False
DEFAULT_DIARIZE = False

# Redaction groups (Deepgram `redact` values) -> display labels.
# PII (de-identification) is listed first. Live output (2026-10-07) showed PII also
# tags occupations ("cardiologist" -> [OCCUPATION_1]) and time spans ("last month" ->
# [DURATION_1]), which carry clinical context, so its label says so; PHI and Numbers strip clinical content
# itself, so both are labeled to flag that trade-off in a medical workflow. Numbers is
# Deepgram's 3+-consecutive-digit rule plus its number-like entities (dates, times,
# ages, medical statistics, locations, ...), so "500 mg" is always redacted and
# shorter clinical values only sometimes: an unpredictable, partly redacted note.
REDACT_GROUPS = {
    "pii": "PII — de-identify (names, locations, IDs, dates; also occupations, time spans)",
    "phi": "PHI — removes clinical content (conditions, drugs, injuries)",
    "pci": "PCI (card numbers)",
    "numbers": "Numbers — 3+ digits, dates, ages, medical statistics (hits clinical values)",
}

MAX_KEYTERMS = 100  # client-side cap; Deepgram's real limit is 500 tokens/request
MAX_UPLOADS = 100
MAX_CONCURRENCY = 5
# 200 MiB — the effective per-file cap. Streamlit's uploader enforces
# server.maxUploadSize (.streamlit/config.toml, also 200); keep the two in sync
# so this guard is reachable rather than shadowed by that smaller default.
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MiB

AUDIO_EXTENSIONS = (".mp3", ".m4a", ".wav", ".flac", ".ogg")

# Words whose Deepgram per-word confidence is strictly BELOW this are flagged for review.
# Deepgram calls word confidence a calibrated probability; its 0.65 example is the
# high-precision/low-recall point. 0.90 flags ~6-7% of words on clean audio and catches
# most model-estimated errors. Re-evaluate on a de-identified local sample; 0.85 is the
# fallback if pilots flag >20% of words. Display-only: never sent to Deepgram, never a widget.
LOW_CONFIDENCE_THRESHOLD = 0.90

# Sign-in / access control (nova/access.py). The policy itself — providers, allowed
# email domains — lives in .streamlit/secrets.toml ([auth] / [access]), not here.
# Local-development opt-out: with no [auth] configured, the app runs without sign-in
# only when the PROCESS environment sets this to exactly "1". Set in .env or
# secrets.toml instead, it blocks the app rather than enabling anything.
ALLOW_ANONYMOUS_ENV = "NOVA_ALLOW_ANONYMOUS"
# [auth] cookie_secret signs the login cookie; anyone who knows it can forge a
# sign-in, so a short, placeholder, or low-variety value is refused.
MIN_COOKIE_SECRET_LENGTH = 32
# A sign-in older than this (by the ID token's `iat`) must sign in again. Streamlit's
# identity cookie lasts 30 days and never re-checks the token, so this is what makes
# deprovisioning at the identity provider take effect within a shift.
MAX_SESSION_AGE_SECONDS = 12 * 60 * 60
# An `iat` further in the future than this is refused too (clock skew allowance).
MAX_CLOCK_SKEW_SECONDS = 5 * 60
