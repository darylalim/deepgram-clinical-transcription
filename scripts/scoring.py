"""Transcript scoring shared by scripts/synthetic_eval.py and scripts/calibrate.py.

Aligns a Deepgram response's words with a reference transcript and counts errors,
**dropped** words (missing speech, which no flag can show), and how many wrong words
the app's own low-confidence flag catches at each threshold. Counts, not
percentages, so files can be summed. Numbers, dates and units are normalized away:
Smart Format writes "fifty" or "50", "03/14/1961" for "March 14, 1961", so
formatting would otherwise be scored as recognition — which also means a misheard
number is not scored here (review doses against the audio regardless).
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from nova.results import first_alternative, is_low_confidence, word_token

NUMBER_WORDS = frozenset({
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
})  # fmt: skip
UNITS = {"mg": "milligrams", "milligram": "milligrams"}


def norm(text: str) -> list[str]:
    """Lowercase words without punctuation, numbers or dates, units unified — applied
    identically to the reference and the transcript."""
    text = re.sub(r"[^a-z0-9' ]", " ", text.lower().replace("-", " "))
    words = [w.strip("'") for w in text.split() if w.strip("'")]
    return [UNITS.get(w, w) for w in words if not w.isdigit() and w not in NUMBER_WORDS]


@dataclass(frozen=True)
class Counts:
    """One or more scored transcripts. `flagged`/`caught` are keyed by threshold."""

    n_ref: int = 0  # scored reference words
    n_hyp: int = 0  # scored transcript words
    errors: int = 0  # substitutions + insertions + deletions
    dropped: int = 0  # reference words with no transcript word at all
    wrong: int = 0  # transcript words outside an exact alignment
    flagged: dict[float, int] = field(default_factory=dict)
    caught: dict[float, int] = field(default_factory=dict)  # wrong AND flagged
    words: frozenset[str] = frozenset()  # transcript words, for term checks only

    def __add__(self, other: "Counts") -> "Counts":
        keys = set(self.flagged) | set(other.flagged)
        return Counts(
            n_ref=self.n_ref + other.n_ref,
            n_hyp=self.n_hyp + other.n_hyp,
            errors=self.errors + other.errors,
            dropped=self.dropped + other.dropped,
            wrong=self.wrong + other.wrong,
            flagged={k: self.flagged.get(k, 0) + other.flagged.get(k, 0) for k in keys},
            caught={k: self.caught.get(k, 0) + other.caught.get(k, 0) for k in keys},
            words=self.words | other.words,
        )

    @property
    def wer(self) -> float:
        return 100 * self.errors / self.n_ref if self.n_ref else 0.0

    def flagged_pct(self, threshold: float) -> float:
        return 100 * self.flagged[threshold] / self.n_hyp if self.n_hyp else 0.0


def score(
    response: Any, reference: Sequence[str], thresholds: Iterable[float]
) -> Counts:
    """Score one transcript against a normalized reference (see `norm`).

    A transcript word is "wrong" when it is outside an exact alignment with the
    reference. Flags use the app's own `is_low_confidence` per Deepgram word; a
    smart-formatted word that normalizes to several pieces shares its flag.
    """
    pieces: list[tuple[str, Any]] = []
    for word in getattr(first_alternative(response), "words", None) or []:
        pieces.extend((p, word) for p in norm(str(word_token(word))))
    hyp = [p for p, _ in pieces]
    correct = [False] * len(hyp)
    errors = dropped = 0
    for tag, i1, i2, j1, j2 in SequenceMatcher(
        a=reference, b=hyp, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            correct[j1:j2] = [True] * (j2 - j1)
            continue
        errors += max(i2 - i1, j2 - j1)
        dropped += max(0, (i2 - i1) - (j2 - j1))
    wrong = [not ok for ok in correct]
    flagged: dict[float, int] = {}
    caught: dict[float, int] = {}
    for threshold in thresholds:
        flags = [is_low_confidence(w, threshold) for _, w in pieces]
        flagged[threshold] = sum(flags)
        caught[threshold] = sum(f and x for f, x in zip(flags, wrong, strict=True))
    return Counts(
        n_ref=len(reference),
        n_hyp=len(hyp),
        errors=errors,
        dropped=dropped,
        wrong=sum(wrong),
        flagged=flagged,
        caught=caught,
        words=frozenset(hyp),
    )
