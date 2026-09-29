"""Deepgram response walkers for the Streamlit UI.

Pure getattr-guarded reads with no streamlit imports, kept separate from the renderer
so they can be unit-tested directly. The walkers keep Deepgram's native 0-based speaker
integers; `speaker_label` applies the 1-based display offset in one place, shared by the
Streamlit renderer and the plain-text transcript export.

`flagged_runs` / `low_confidence_count` mark words whose per-word confidence falls below
`LOW_CONFIDENCE_THRESHOLD` so a reviewer knows where to check the audio first. A missing
or non-numeric score fails toward review (flagged); redaction tags are never flagged.
"""

import math
import re
from dataclasses import dataclass
from typing import Any

from nova.config import LOW_CONFIDENCE_THRESHOLD


@dataclass(frozen=True)
class Token:
    """One display token of a transcript and whether it scored below the threshold."""

    text: str  # whitespace-normalized display token (" ".join(word_token(w).split()))
    low_confidence: bool


# A redaction tag as Deepgram emits it — [SSN_1], [CREDIT_CARD_1], the generic
# [REDACTED] — optionally followed by sentence punctuation. Used with fullmatch.
_REDACTION_TAG = re.compile(r"\[[A-Z][A-Z0-9_]*\][.,:;?!]*", re.IGNORECASE)

# Blank-line paragraph breaks in a smart-formatted transcript.
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


def word_token(word: Any) -> str:
    """The word's display token: `punctuated_word` when present, else `word`."""
    return getattr(word, "punctuated_word", None) or getattr(word, "word", "")


def speaker_label(speaker: Any) -> int | None:
    """1-based display label for an integer speaker, or None for a non-integer.

    The core keeps speakers as Deepgram's 0-based ints; this is the single place the
    `+1` display offset is applied (used by the Streamlit renderer and the plain-text
    transcript export).
    """
    return speaker + 1 if isinstance(speaker, int) else None


def is_redaction_tag(token: Any) -> bool:
    """True for a redaction tag such as `[SSN_1]`, `[REDACTED]` or `[redacted].`.

    Tags are never flagged: the spoken content is gone, so a reviewer cannot verify it
    against the audio, and the confidence Deepgram assigns a tag is undocumented.
    """
    return (
        isinstance(token, str) and _REDACTION_TAG.fullmatch(token.strip()) is not None
    )


def is_low_confidence(word: Any, threshold: float = LOW_CONFIDENCE_THRESHOLD) -> bool:
    """True when the word's confidence is strictly below `threshold`.

    Fails toward review: a missing, None, bool, non-numeric, NaN or infinite
    confidence counts as low. A redaction tag is never low. Only `confidence` is read —
    `speaker_confidence` measures who spoke, not whether the word is right.
    """
    if is_redaction_tag(word_token(word)):
        return False
    confidence = getattr(word, "confidence", None)
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
    ):
        return True
    return confidence < threshold


def first_alternative(response: Any) -> Any | None:
    """Return the first channel's first alternative, or None if the response lacks one.

    Pre-recorded calls return a `ListenV1Response` (with `results`); a callback/async
    call would instead yield a `ListenV1AcceptedResponse` that has only `request_id`
    and no `results`. Guard that path plus empty channels/alternatives so callers
    degrade gracefully instead of raising.
    """
    results = getattr(response, "results", None)
    channels = getattr(results, "channels", None) or []
    if not channels:
        return None
    alternatives = getattr(channels[0], "alternatives", None) or []
    return alternatives[0] if alternatives else None


def transcript_text(response: Any) -> str | None:
    """Pull the transcript, or None if the response carries no usable results."""
    alternative = first_alternative(response)
    if alternative is None:
        return None
    return getattr(alternative, "transcript", None)


def _speaker_runs(words: list[Any]) -> list[tuple[Any, list[Any]]]:
    """Group words into consecutive (speaker, words) runs.

    A new run opens only on an integer speaker that differs from the current run's. A
    word whose speaker is missing/non-int (rare mid-stream) continues the current run
    instead of opening a bogus "Speaker None" segment; callers gate on an integer
    `words[0].speaker`, so a run already exists by then.
    """
    runs: list[tuple[Any, list[Any]]] = []
    for word in words:
        speaker = getattr(word, "speaker", None)
        new_run = not runs or (isinstance(speaker, int) and speaker != runs[-1][0])
        if new_run:
            runs.append((speaker, [word]))
        else:
            runs[-1][1].append(word)
    return runs


def diarized_segments(response: Any) -> list[tuple[Any, str]] | None:
    """Group words into consecutive (speaker, text) runs when diarization labeled them.

    Returns None when the response has no per-word integer speaker labels (diarize off,
    or no usable results), so the caller falls back to the flat transcript.
    """
    alternative = first_alternative(response)
    if alternative is None:
        return None
    words = getattr(alternative, "words", None) or []
    if not words or not isinstance(getattr(words[0], "speaker", None), int):
        return None
    return [
        (speaker, " ".join(word_token(word) for word in run))
        for speaker, run in _speaker_runs(words)
    ]


def _tokens(words: list[Any], threshold: float) -> list[Token]:
    """Whitespace-normalized display tokens, skipping non-str or empty ones."""
    tokens = []
    for word in words:
        raw = word_token(word)
        if not isinstance(raw, str):
            continue
        text = " ".join(raw.split())
        if text:
            tokens.append(Token(text, is_low_confidence(word, threshold)))
    return tokens


def _paragraph_runs(
    tokens: list[Token], transcript: str
) -> list[tuple[int | None, list[Token]]] | None:
    """Split flat tokens into one run per transcript paragraph, or None on any mismatch.

    Tokens are consumed paragraph by paragraph, each compared piece-by-piece (split on
    whitespace — one smart-formatted token such as "(555) 123-4567" spans several) with
    that paragraph's text. The highlighted view is built from the words, so this is the
    fidelity guard: if the words do not reproduce the transcript exactly (modulo
    whitespace), the caller shows the plain transcript instead of a view that differs.
    """
    runs: list[tuple[int | None, list[Token]]] = []
    position = 0
    for paragraph in _PARAGRAPH_BREAK.split(transcript):
        expected = paragraph.split()
        consumed = 0
        run: list[Token] = []
        while consumed < len(expected):
            if position == len(tokens):
                return None
            token = tokens[position]
            pieces = token.text.split()
            if pieces != expected[consumed : consumed + len(pieces)]:
                return None
            run.append(token)
            consumed += len(pieces)
            position += 1
        if run:
            runs.append((None, run))
    if position != len(tokens) or not runs:
        return None
    return runs


def flagged_runs(
    response: Any, threshold: float = LOW_CONFIDENCE_THRESHOLD
) -> list[tuple[int | None, list[Token]]] | None:
    """Per-word low-confidence tokens, grouped into display runs.

    Diarized (an integer `words[0].speaker`): one `(speaker, tokens)` run per speaker
    turn, with Deepgram's 0-based speaker. Flat: one `(None, tokens)` run per transcript
    paragraph. Returns None — the caller then shows the plain transcript, unhighlighted —
    when there is no alternative, `words` is not a non-empty list, or (flat only) the
    words do not reproduce the transcript, paragraph by paragraph.
    """
    alternative = first_alternative(response)
    if alternative is None:
        return None
    words = getattr(alternative, "words", None)
    if not isinstance(words, list) or not words:
        return None
    if isinstance(getattr(words[0], "speaker", None), int):
        return [
            (speaker, _tokens(run, threshold)) for speaker, run in _speaker_runs(words)
        ]
    transcript = getattr(alternative, "transcript", None)
    if not isinstance(transcript, str):
        return None
    return _paragraph_runs(_tokens(words, threshold), transcript)


def low_confidence_count(
    response: Any, threshold: float = LOW_CONFIDENCE_THRESHOLD
) -> int | None:
    """Number of flagged tokens, or None when highlighting is unavailable."""
    runs = flagged_runs(response, threshold)
    if runs is None:
        return None
    return sum(token.low_confidence for _, tokens in runs for token in tokens)
