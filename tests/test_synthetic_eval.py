"""Guard for scripts/synthetic_eval.py, which CI never runs (billed, needs a key).

Its scoring is pure, so it is tested here with fake responses — the first draft of
it counted Smart Format's "fifty" vs "50" as recognition errors, the kind of bug that
silently inflates a word error rate. Also keeps its requests in step with the core."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from nova.config import MODELS
from nova.transcribe import build_options

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "synthetic_eval.py"


@pytest.fixture(scope="module")
def ev():
    spec = importlib.util.spec_from_file_location("synthetic_eval", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # import only: main() is behind __main__
    return module


def _response(tokens, confidence=0.99, low=()):
    words = [
        SimpleNamespace(
            punctuated_word=t, word=t, confidence=0.5 if i in low else confidence
        )
        for i, t in enumerate(tokens)
    ]
    alt = SimpleNamespace(words=words, transcript=" ".join(tokens))
    return SimpleNamespace(
        results=SimpleNamespace(channels=[SimpleNamespace(alternatives=[alt])])
    )


class TestRequests:
    def test_every_config_is_a_valid_request_and_covers_every_model(self, ev):
        configs = ev.configs()
        for opts in configs.values():
            build_options(**opts)
        assert {opts["model"] for opts in configs.values()} == set(MODELS)

    def test_every_scored_term_is_in_the_script(self, ev):
        reference = set(ev.norm(" ".join(line for _, line in ev.SCRIPT)))
        assert set(ev.TERMS) <= reference


class TestNorm:
    def test_numbers_and_dates_are_not_scored(self, ev):
        assert ev.norm("March 14, 1961: fifty 50 milligrams") == ["milligrams"]

    def test_units_and_punctuation_are_unified(self, ev):
        assert ev.norm("Take 10 mg. Metoprolol-succinate!") == [
            "take",
            "milligrams",
            "metoprolol",
            "succinate",
        ]


class TestScore:
    REFERENCE = ["take", "apixaban", "and", "tamsulosin", "today"]

    def test_perfect_transcript(self, ev):
        s = ev.score(
            _response(["Take", "apixaban", "and", "tamsulosin", "today."]),
            self.REFERENCE,
        )

        assert (s.wer, s.dropped) == (0.0, 0)
        assert s.caught[0.90] == (0, 0)
        assert "apixaban" not in s.terms_missed and "tamsulosin" not in s.terms_missed

    def test_missing_speech_counts_as_dropped_and_cannot_be_flagged(self, ev):
        s = ev.score(_response(["Take", "today."]), self.REFERENCE)

        assert s.dropped == 3
        assert s.wer == pytest.approx(60.0)
        assert s.caught[0.90] == (0, 0)  # nothing on screen to flag
        assert {"apixaban", "tamsulosin"} <= set(s.terms_missed)

    def test_flagged_and_unflagged_mistakes(self, ev):
        flagged = ev.score(
            _response(["Take", "apixaban", "and", "tamsulin", "today."], low={3}),
            self.REFERENCE,
        )
        confident = ev.score(
            _response(["Take", "apixaban", "and", "tamsulin", "today."]), self.REFERENCE
        )

        assert flagged.caught[0.90] == (1, 1)
        assert confident.caught[0.90] == (0, 1)  # confidently wrong: missed by flags
        assert flagged.dropped == confident.dropped == 0
        assert flagged.flagged[0.90] == pytest.approx(20.0)
