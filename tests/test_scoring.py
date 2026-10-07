"""scripts/scoring.py: the transcript scoring behind synthetic_eval and calibrate.

Pure, so tested with fake responses. Its first draft counted Smart Format's "fifty"
vs "50" as recognition errors — the kind of bug that silently inflates a word error
rate — so normalization and alignment are pinned here."""

from types import SimpleNamespace

import pytest

from scripts.scoring import Counts, norm, score

REFERENCE = ["take", "apixaban", "and", "tamsulosin", "today"]


def response(tokens, low=()):
    """A Deepgram-shaped response; words at indexes in `low` score 0.5, else 0.99."""
    words = [
        SimpleNamespace(punctuated_word=t, word=t, confidence=0.5 if i in low else 0.99)
        for i, t in enumerate(tokens)
    ]
    alt = SimpleNamespace(words=words, transcript=" ".join(tokens))
    return SimpleNamespace(
        results=SimpleNamespace(channels=[SimpleNamespace(alternatives=[alt])])
    )


class TestNorm:
    def test_numbers_and_dates_are_not_scored(self):
        assert norm("March 14, 1961: fifty 50 milligrams") == ["milligrams"]
        assert norm("seen on 03/14 in October") == ["seen", "on", "in"]

    def test_units_and_punctuation_are_unified(self):
        assert norm("Take 10 mg. Metoprolol-succinate!") == [
            "take",
            "milligrams",
            "metoprolol",
            "succinate",
        ]


class TestScore:
    def test_perfect_transcript(self):
        s = score(
            response(["Take", "apixaban", "and", "tamsulosin", "today."]),
            REFERENCE,
            [0.9],
        )

        assert (s.errors, s.dropped, s.wrong, s.caught[0.9]) == (0, 0, 0, 0)
        assert s.wer == 0.0
        assert {"apixaban", "tamsulosin"} <= s.words

    def test_missing_speech_counts_as_dropped_and_cannot_be_flagged(self):
        s = score(response(["Take", "today."]), REFERENCE, [0.9])

        assert s.dropped == 3
        assert s.wer == pytest.approx(60.0)
        assert (s.wrong, s.caught[0.9]) == (0, 0)  # nothing on screen to flag

    def test_flagged_and_unflagged_mistakes(self):
        tokens = ["Take", "apixaban", "and", "tamsulin", "today."]
        flagged = score(response(tokens, low={3}), REFERENCE, [0.9])
        confident = score(response(tokens), REFERENCE, [0.9])

        assert (flagged.caught[0.9], flagged.wrong) == (1, 1)
        assert (confident.caught[0.9], confident.wrong) == (0, 1)  # missed by flags
        assert flagged.flagged_pct(0.9) == pytest.approx(20.0)

    def test_each_threshold_is_scored(self):
        s = score(
            response(["Take", "apixaban"], low={1}), ["take", "apixaban"], [0.95, 0.4]
        )

        # 0.5 is below 0.95 but not 0.4; 0.99 is below neither (see response()).
        assert s.flagged == {0.95: 1, 0.4: 0}

    def test_counts_add_across_files(self):
        a = score(response(["Take", "today."]), REFERENCE, [0.9])
        b = score(
            response(["Take", "apixaban", "and", "tamsulin", "today."], low={3}),
            REFERENCE,
            [0.9],
        )
        total = a + b

        assert (total.n_ref, total.dropped, total.wrong, total.caught[0.9]) == (
            10,
            3,
            1,
            1,
        )
        assert total.wer == pytest.approx(40.0)
        assert Counts() + a == a
