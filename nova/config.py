"""Transcription constants — the single source of truth for the Streamlit UI."""

MODEL = "nova-3-medical"

# Nova-3 Medical supports English variants only.
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
# PII (de-identification) is listed first; PHI and Numbers strip clinical content
# itself, so both are labeled to flag that trade-off in a medical workflow. Numbers is
# Deepgram's 3+-consecutive-digit rule plus its number-like entities (dates, times,
# ages, medical statistics, locations, ...), so "500 mg" is always redacted and
# shorter clinical values only sometimes: an unpredictable, partly redacted note.
REDACT_GROUPS = {
    "pii": "PII — de-identify (names, locations, IDs)",
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
