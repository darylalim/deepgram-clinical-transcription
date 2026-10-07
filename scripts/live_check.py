"""Live check: does Deepgram accept each app option with each model?

Sends Deepgram's public sample (a NASA spacewalk clip, no patient data) by URL with
exactly the kwargs the app builds (`nova.transcribe.build_options`), once per
`MODELS` entry and option case. Use it when adding a model or an option: CI mocks
Deepgram, so this is the only check that the real API accepts the combination.

Accepted is not the same as applied: the sample has one speaker, no drug names and
nothing to redact, so this shows each option is accepted, not that it changes output.

Makes real, billed requests (a few seconds of audio each) with DEEPGRAM_API_KEY from
the environment or `.env`. Prints only status, Deepgram's error message, word counts
and the share of words below LOW_CONFIDENCE_THRESHOLD — never the key, headers, or
transcript text.

    uv run python scripts/live_check.py
"""

import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepgram import DeepgramClient  # noqa: E402
from dotenv import dotenv_values  # noqa: E402

from nova.config import LOW_CONFIDENCE_THRESHOLD, MODELS  # noqa: E402
from nova.results import first_alternative, is_low_confidence  # noqa: E402
from nova.transcribe import build_options  # noqa: E402

SAMPLE = "https://dpgr.am/spacewalk.wav"
# Each case is build_options kwargs (the model is added per run).
CASES: dict[str, dict[str, Any]] = {
    "defaults": {},
    "language=en": {"language": "en"},
    "language=en-GB": {"language": "en-GB"},
    "smart_format off": {"smart_format": False},
    "keyterms": {"keyterms": ["spacewalk", "metformin"]},
    "diarize": {"diarize": True},
    "dictation": {"dictation": True},
    "measurements": {"measurements": True},
    "redact all groups": {"redact": ["pii", "phi", "pci", "numbers"]},
    "everything on": {
        "keyterms": ["metformin"],
        "language": "en-US",
        "diarize": True,
        "dictation": True,
        "measurements": True,
        "redact": ["pii", "phi", "pci", "numbers"],
    },
}


def summarize(response: Any) -> str:
    words = list(getattr(first_alternative(response), "words", None) or [])
    low = sum(1 for w in words if is_low_confidence(w))
    speakers = {w.speaker for w in words if getattr(w, "speaker", None) is not None}
    pct = f"{100 * low / len(words):.0f}%" if words else "n/a"
    return (
        f"OK  words={len(words):3d} "
        f"below{LOW_CONFIDENCE_THRESHOLD:.2f}={pct:>4} speakers={len(speakers)}"
    )


def error_text(exc: Exception) -> str:
    # ApiError's str() includes headers; print only the status and Deepgram's message.
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    msg = (
        (body.get("err_msg") or body.get("err_code")) if isinstance(body, dict) else ""
    )
    return f"ERR {status} {type(exc).__name__}: {str(msg)[:140]}"


def main() -> None:
    key = os.environ.get("DEEPGRAM_API_KEY") or dotenv_values(".env").get(
        "DEEPGRAM_API_KEY"
    )
    if not key:
        sys.exit("DEEPGRAM_API_KEY not found in the environment or .env")
    media = DeepgramClient(api_key=key).listen.v1.media
    for model in MODELS:
        print(f"\n== {model}")
        for name, opts in CASES.items():
            try:
                result = summarize(
                    media.transcribe_url(
                        url=SAMPLE, **build_options(model=model, **opts)
                    )
                )
            except Exception as exc:  # report and keep going
                result = error_text(exc)
            print(f"  {name:18s} {result}")


if __name__ == "__main__":
    main()
