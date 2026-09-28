import math
from types import SimpleNamespace

import pytest
from deepgram.core.unchecked_base_model import construct_type
from deepgram.types import ListenV1ResponseResultsChannelsItemAlternativesItemWordsItem

from nova.config import LOW_CONFIDENCE_THRESHOLD
from nova.results import (
    Token,
    diarized_segments,
    first_alternative,
    flagged_runs,
    is_low_confidence,
    is_redaction_tag,
    low_confidence_count,
    transcript_text,
    word_token,
)
from tests.helpers import mock_word


def _resp(words=None, transcript=None, *, has_results=True):
    """A ListenV1Response-shaped object; `has_results=False` mimics ListenV1AcceptedResponse."""
    if not has_results:
        return SimpleNamespace(request_id="req-1")
    alt = SimpleNamespace(transcript=transcript, words=[] if words is None else words)
    return SimpleNamespace(
        results=SimpleNamespace(channels=[SimpleNamespace(alternatives=[alt])])
    )


class TestFirstAlternative:
    def test_returns_first_alternative(self):
        alt = SimpleNamespace(transcript="hi")
        resp = SimpleNamespace(
            results=SimpleNamespace(channels=[SimpleNamespace(alternatives=[alt])])
        )
        assert first_alternative(resp) is alt

    def test_none_when_no_results(self):
        assert first_alternative(_resp(has_results=False)) is None

    def test_none_when_empty_channels(self):
        assert (
            first_alternative(SimpleNamespace(results=SimpleNamespace(channels=[])))
            is None
        )

    def test_none_when_empty_alternatives(self):
        resp = SimpleNamespace(
            results=SimpleNamespace(channels=[SimpleNamespace(alternatives=[])])
        )
        assert first_alternative(resp) is None


class TestTranscriptText:
    def test_returns_transcript(self):
        assert (
            transcript_text(_resp(transcript="life moves pretty fast"))
            == "life moves pretty fast"
        )

    def test_none_when_no_results(self):
        assert transcript_text(_resp(has_results=False)) is None


class TestDiarizedSegments:
    def test_groups_consecutive_runs_with_zero_based_speakers(self):
        words = [
            mock_word("Hello", 0.9, speaker=0),
            mock_word("doctor.", 0.9, speaker=0),
            mock_word("Hi", 0.9, speaker=1),
            mock_word("there.", 0.9, speaker=1),
            mock_word("Yes?", 0.9, speaker=0),
        ]
        assert diarized_segments(_resp(words)) == [
            (0, "Hello doctor."),
            (1, "Hi there."),
            (0, "Yes?"),
        ]

    def test_single_speaker_one_run(self):
        words = [mock_word("Note.", 0.9, speaker=0), mock_word("Done.", 0.9, speaker=0)]
        assert diarized_segments(_resp(words)) == [(0, "Note. Done.")]

    def test_punctuated_word_falls_back_to_word(self):
        w = SimpleNamespace(punctuated_word=None, word="stat", speaker=0)
        assert diarized_segments(_resp([w])) == [(0, "stat")]

    def test_unlabeled_word_continues_current_run(self):
        words = [
            mock_word("Patient", 0.9, speaker=0),
            mock_word("reports", 0.9, speaker=None),
            mock_word("pain.", 0.9, speaker=0),
        ]
        assert diarized_segments(_resp(words)) == [(0, "Patient reports pain.")]

    def test_none_without_integer_speaker(self):
        assert diarized_segments(_resp([mock_word("hi", 0.9)])) is None

    def test_none_for_empty_words(self):
        assert diarized_segments(_resp([])) is None

    def test_none_for_empty_alternatives(self):
        resp = SimpleNamespace(
            results=SimpleNamespace(channels=[SimpleNamespace(alternatives=[])])
        )
        assert diarized_segments(resp) is None


def _word(text, confidence=0.99, speaker=None):
    """A plain word object (not a MagicMock), so a missing attribute stays missing."""
    return SimpleNamespace(punctuated_word=text, confidence=confidence, speaker=speaker)


class TestIsLowConfidence:
    """Strictly *below* the threshold is low; anything unreadable fails toward review."""

    def test_equal_to_threshold_is_not_low(self):
        assert not is_low_confidence(_word("dose", LOW_CONFIDENCE_THRESHOLD))

    def test_next_float_below_threshold_is_low(self):
        below = math.nextafter(LOW_CONFIDENCE_THRESHOLD, 0)
        assert is_low_confidence(_word("dose", below))

    def test_one_is_not_low(self):
        assert not is_low_confidence(_word("dose", 1.0))

    def test_zero_is_low(self):
        assert is_low_confidence(_word("dose", 0.0))

    def test_int_one_is_not_low(self):
        assert not is_low_confidence(_word("dose", 1))

    def test_none_is_low(self):
        assert is_low_confidence(_word("dose", None))

    def test_missing_attribute_is_low(self):
        assert is_low_confidence(SimpleNamespace(punctuated_word="dose"))

    def test_bool_is_low_not_treated_as_one(self):
        assert is_low_confidence(_word("dose", True))

    def test_non_number_is_low(self):
        assert is_low_confidence(_word("dose", "0.99"))

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_non_finite_is_low(self, value):
        assert is_low_confidence(_word("dose", value))

    def test_custom_threshold(self):
        assert is_low_confidence(_word("dose", 0.7), threshold=0.75)
        assert not is_low_confidence(_word("dose", 0.7), threshold=0.65)

    def test_redaction_tag_is_never_low(self):
        assert not is_low_confidence(_word("[SSN_1]", 0.1))

    def test_ignores_speaker_confidence(self):
        word = SimpleNamespace(
            punctuated_word="absolutely", confidence=0.99, speaker_confidence=0.01
        )
        assert not is_low_confidence(word)


class TestIsRedactionTag:
    @pytest.mark.parametrize(
        "token", ["[SSN_1]", "[CREDIT_CARD_1],", "[REDACTED]", "[redacted]."]
    )
    def test_tags(self, token):
        assert is_redaction_tag(token)

    @pytest.mark.parametrize("token", ["SSN_1", "[1]", "[SSN_1]x", "[", "", None])
    def test_not_tags(self, token):
        assert not is_redaction_tag(token)


class TestFlaggedRuns:
    def test_flat_single_run_with_flags(self):
        words = [_word("Take"), _word("50", 0.4), _word("mg.")]
        assert flagged_runs(_resp(words, "Take 50 mg.")) == [
            (
                None,
                [Token("Take", False), Token("50", True), Token("mg.", False)],
            )
        ]

    def test_diarized_splits_on_speaker_change(self):
        words = [
            _word("Hello", speaker=0),
            _word("doctor.", 0.5, speaker=0),
            _word("Hi.", speaker=1),
        ]
        assert flagged_runs(_resp(words)) == [
            (0, [Token("Hello", False), Token("doctor.", True)]),
            (1, [Token("Hi.", False)]),
        ]

    def test_diarized_unlabeled_word_continues_its_run(self):
        words = [
            _word("Patient", speaker=0),
            _word("reports", 0.3, speaker=None),
            _word("pain.", speaker=0),
        ]
        assert flagged_runs(_resp(words)) == [
            (
                0,
                [
                    Token("Patient", False),
                    Token("reports", True),
                    Token("pain.", False),
                ],
            )
        ]

    def test_none_without_results(self):
        assert flagged_runs(_resp(has_results=False)) is None

    def test_none_for_empty_words(self):
        assert flagged_runs(_resp([], "")) is None

    def test_none_for_non_list_words(self):
        alt = SimpleNamespace(transcript="hi", words=(_word("hi"),))
        resp = SimpleNamespace(
            results=SimpleNamespace(channels=[SimpleNamespace(alternatives=[alt])])
        )
        assert flagged_runs(resp) is None

    def test_none_when_flat_words_mismatch_transcript(self):
        # The words must reproduce the transcript, or the highlighted view would show
        # different text than Deepgram returned: fall back to the plain transcript.
        words = [_word("Take"), _word("50"), _word("mg.")]
        assert flagged_runs(_resp(words, "Take 500 mg.")) is None

    def test_none_when_flat_words_run_past_transcript(self):
        words = [_word("Take"), _word("50"), _word("mg."), _word("daily.")]
        assert flagged_runs(_resp(words, "Take 50 mg.")) is None

    def test_none_when_flat_transcript_is_not_a_string(self):
        assert flagged_runs(_resp([_word("hi")], None)) is None

    def test_whitespace_only_difference_still_returns_runs(self):
        words = [_word("Take"), _word("50"), _word("mg.")]
        runs = flagged_runs(_resp(words, "  Take   50\tmg. "))
        assert runs == [
            (None, [Token("Take", False), Token("50", False), Token("mg.", False)])
        ]

    def test_multi_piece_token_matches_transcript(self):
        # One smart-formatted entity (a phone number) is a single word entry whose
        # token spans several whitespace-separated pieces of the transcript.
        words = [_word("Call"), _word("(555) 123-4567", 0.6), _word("today.")]
        assert flagged_runs(_resp(words, "Call (555) 123-4567 today.")) == [
            (
                None,
                [
                    Token("Call", False),
                    Token("(555) 123-4567", True),
                    Token("today.", False),
                ],
            )
        ]

    def test_flat_paragraphs_become_separate_runs(self):
        words = [_word("First."), _word("Second", 0.2), _word("para.")]
        transcript = "First.\n\nSecond para."
        assert flagged_runs(_resp(words, transcript)) == [
            (None, [Token("First.", False)]),
            (None, [Token("Second", True), Token("para.", False)]),
        ]

    def test_mismatch_in_second_paragraph_is_none(self):
        words = [_word("First."), _word("Second"), _word("para.")]
        assert flagged_runs(_resp(words, "First.\n\nSecond part.")) is None

    def test_token_spanning_a_paragraph_break_is_none(self):
        words = [_word("First. Second"), _word("para.")]
        assert flagged_runs(_resp(words, "First.\n\nSecond para.")) is None

    def test_punctuated_word_falls_back_to_word(self):
        w = SimpleNamespace(punctuated_word=None, word="stat", confidence=0.5)
        assert flagged_runs(_resp([w], "stat")) == [(None, [Token("stat", True)])]

    def test_empty_token_is_skipped(self):
        words = [_word("Take"), _word(""), _word("  "), _word("it.")]
        assert flagged_runs(_resp(words, "Take it.")) == [
            (None, [Token("Take", False), Token("it.", False)])
        ]

    def test_custom_threshold(self):
        words = [_word("Take", 0.8)]
        assert flagged_runs(_resp(words, "Take"), threshold=0.75) == [
            (None, [Token("Take", False)])
        ]


class TestLowConfidenceCount:
    def test_counts_flagged_tokens(self):
        words = [_word("Take", 0.1), _word("50", 0.4), _word("mg.")]
        assert low_confidence_count(_resp(words, "Take 50 mg.")) == 2

    def test_zero_when_nothing_flagged(self):
        assert low_confidence_count(_resp([_word("Take")], "Take")) == 0

    def test_none_when_highlighting_unavailable(self):
        assert low_confidence_count(_resp(has_results=False)) is None
        assert low_confidence_count(_resp([_word("Take")], "Other")) is None


class TestRealSdkWord:
    """The REST word model as the SDK actually builds it (`construct_type`, no
    validation): `punctuated_word` arrives only as an extra field, and a missing
    `confidence` key becomes None."""

    @staticmethod
    def _word(**fields):
        return construct_type(
            type_=ListenV1ResponseResultsChannelsItemAlternativesItemWordsItem,
            object_={"word": "hello", "punctuated_word": "Hello,", **fields},
        )

    def test_low_confidence_word_is_flagged(self):
        word = self._word(confidence=0.5, start=0.0, end=0.4)
        assert word_token(word) == "Hello,"
        assert is_low_confidence(word)

    def test_missing_confidence_is_flagged(self):
        word = self._word()
        assert word.confidence is None
        assert is_low_confidence(word)

    def test_confident_word_is_not_flagged(self):
        assert not is_low_confidence(self._word(confidence=0.99))
