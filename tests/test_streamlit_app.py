from unittest.mock import MagicMock, patch

import pytest

import streamlit_app
from tests.helpers import mock_upload, mock_word, wav_bytes

FAKE_AUDIO = b"fake-audio-data"

# Option building (build_options) and the batch runner (transcribe_batch) are tested
# directly in tests/test_transcribe.py; the response walkers in tests/test_results.py.
# This module covers only the Streamlit-side behavior: the _transcribe_batch wrapper's
# progress / error / session-state handling, _run validation, _feature_opts, and the
# renderers.


class TestProcessInputs:
    """The _transcribe_batch wrapper (via _process_inputs): session state, playback
    sources, progress, and per-item error rendering. Option pass-through and batch
    mechanics live in test_transcribe.py."""

    def test_stores_responses_in_session_state(self, mock_deepgram_cls, mock_st):
        streamlit_app._process_inputs("test-key", [("test.wav", FAKE_AUDIO)])

        responses = mock_st.session_state["responses"]
        assert len(responses) == 1
        assert responses[0][0] == "test.wav"

    def test_stores_audio_sources_in_session_state(self, mock_deepgram_cls, mock_st):
        streamlit_app._process_inputs("test-key", [("a.wav", b"a"), ("b.wav", b"b")])

        assert mock_st.session_state["audio_sources"] == [b"a", b"b"]

    def test_forwards_features_to_sdk_call(self, mock_deepgram_cls, mock_st):
        # The wrapper forwards each feature kwarg by name through build_options to the SDK
        # call; a forwarding typo (e.g. swapping diarize/measurements) would slip past the
        # build_options/transcribe_batch unit tests but is caught here.
        streamlit_app._process_inputs(
            "test-key",
            [("test.wav", FAKE_AUDIO)],
            keyterms=["metformin"],
            language="en-GB",
            diarize=True,
            # Opposite of diarize, so a swapped forward is visible whatever the defaults.
            measurements=False,
            redact=["pii"],
        )

        kwargs = mock_deepgram_cls.return_value.listen.v1.media.transcribe_file.call_args.kwargs
        assert kwargs["keyterm"] == ["metformin"]
        assert kwargs["language"] == "en-GB"
        assert kwargs["diarize"] is True
        assert "measurements" not in kwargs
        assert kwargs["request_options"] == {
            "additional_query_parameters": {"redact": ["pii"]}
        }

    def test_large_file_dropped_from_playback(self, mock_deepgram_cls, mock_st):
        with patch.object(streamlit_app, "MAX_PLAYBACK_BYTES", 2):
            streamlit_app._process_inputs(
                "test-key", [("big.wav", b"big"), ("small.wav", b"a")]
            )

        assert mock_st.session_state["audio_sources"] == [None, b"a"]

    def test_continues_after_single_file_failure(self, mock_deepgram_cls, mock_st):
        mock_client = mock_deepgram_cls.return_value
        good_response = MagicMock()

        def fake_transcribe(request, **_):
            if request == b"bad":
                raise Exception("API error")
            return good_response

        mock_client.listen.v1.media.transcribe_file.side_effect = fake_transcribe

        streamlit_app._process_inputs(
            "test-key", [("bad.wav", b"bad"), ("good.wav", b"good")]
        )

        mock_st.error.assert_called_once_with(
            "Transcription failed for bad.wav: API error", icon=":material/error:"
        )
        assert mock_st.session_state["responses"] == [("good.wav", good_response)]
        assert mock_st.session_state["audio_sources"] == [b"good"]

    def test_middle_file_failure_keeps_alignment(self, mock_deepgram_cls, mock_st):
        mock_client = mock_deepgram_cls.return_value
        resp_a, resp_c = MagicMock(), MagicMock()

        def fake_transcribe(request, **_):
            if request == b"b":
                raise Exception("boom")
            return resp_a if request == b"a" else resp_c

        mock_client.listen.v1.media.transcribe_file.side_effect = fake_transcribe

        streamlit_app._process_inputs(
            "test-key", [("a.wav", b"a"), ("b.wav", b"b"), ("c.wav", b"c")]
        )

        assert [n for n, _ in mock_st.session_state["responses"]] == ["a.wav", "c.wav"]
        assert mock_st.session_state["audio_sources"] == [b"a", b"c"]

    def test_all_files_failing_clears_session_state(self, mock_deepgram_cls, mock_st):
        mock_client = mock_deepgram_cls.return_value
        mock_client.listen.v1.media.transcribe_file.side_effect = Exception("fail")

        streamlit_app._process_inputs("test-key", [("a.wav", b"a"), ("b.wav", b"b")])

        assert mock_st.error.call_count == 2
        assert mock_st.session_state["responses"] == []
        assert mock_st.session_state["audio_sources"] == []

    def test_clears_stale_results_on_total_failure(self, mock_deepgram_cls, mock_st):
        mock_st.session_state["responses"] = [("old.wav", MagicMock())]
        mock_st.session_state["audio_sources"] = [b"old"]
        mock_client = mock_deepgram_cls.return_value
        mock_client.listen.v1.media.transcribe_file.side_effect = Exception("fail")

        streamlit_app._process_inputs("test-key", [("a.wav", b"a")])

        assert mock_st.session_state["responses"] == []
        assert mock_st.session_state["audio_sources"] == []

    def test_stores_all_successful_responses(self, mock_deepgram_cls, mock_st):
        streamlit_app._process_inputs(
            "test-key", [("a.wav", b"a"), ("b.wav", b"b"), ("c.wav", b"c")]
        )

        responses = mock_st.session_state["responses"]
        assert len(responses) == 3
        assert [name for name, _ in responses] == ["a.wav", "b.wav", "c.wav"]

    def test_preserves_input_order_under_reversed_completion(
        self, mock_deepgram_cls, mock_st
    ):
        mock_client = mock_deepgram_cls.return_value

        def dispatch(request, **_):
            resp = MagicMock()
            resp.tag = request
            return resp

        mock_client.listen.v1.media.transcribe_file.side_effect = dispatch

        with patch("streamlit_app.as_completed", side_effect=lambda fs: list(fs)[::-1]):
            streamlit_app._process_inputs(
                "test-key", [("a.wav", b"a"), ("b.wav", b"b"), ("c.wav", b"c")]
            )

        responses = mock_st.session_state["responses"]
        assert [n for n, _ in responses] == ["a.wav", "b.wav", "c.wav"]
        assert [r.tag for _, r in responses] == [b"a", b"b", b"c"]
        assert mock_st.session_state["audio_sources"] == [b"a", b"b", b"c"]

    def test_error_message_includes_filename_and_exception(
        self, mock_deepgram_cls, mock_st
    ):
        mock_client = mock_deepgram_cls.return_value
        mock_client.listen.v1.media.transcribe_file.side_effect = Exception("timeout")

        streamlit_app._process_inputs("test-key", [("bad.wav", b"bad")])

        mock_st.error.assert_called_once_with(
            "Transcription failed for bad.wav: timeout", icon=":material/error:"
        )

    def test_uses_progress_bar_not_spinner(self, mock_deepgram_cls, mock_st):
        streamlit_app._process_inputs("test-key", [("a.wav", b"a"), ("b.wav", b"b")])

        mock_st.progress.assert_called()
        mock_st.spinner.assert_not_called()

    def test_drives_a_status_container(self, mock_deepgram_cls, mock_st):
        streamlit_app._process_inputs("test-key", [("a.wav", b"a"), ("b.wav", b"b")])

        mock_st.status.assert_called_once()
        status = mock_st.status.return_value.__enter__.return_value
        status.update.assert_any_call(
            label="Transcribed 2/2", state="complete", expanded=False
        )

    def test_toast_summarizes_full_success(self, mock_deepgram_cls, mock_st):
        streamlit_app._process_inputs("test-key", [("a.wav", b"a"), ("b.wav", b"b")])

        mock_st.toast.assert_called_once()
        msg = mock_st.toast.call_args.args[0]
        assert "2" in msg
        assert "failed" not in msg.lower()
        assert mock_st.toast.call_args.kwargs["icon"] == ":material/check_circle:"

    def test_toast_reports_total_failure(self, mock_deepgram_cls, mock_st):
        mock_client = mock_deepgram_cls.return_value
        mock_client.listen.v1.media.transcribe_file.side_effect = Exception("fail")

        streamlit_app._process_inputs("test-key", [("a.wav", b"a")])

        mock_st.toast.assert_called_once()
        assert "fail" in mock_st.toast.call_args.args[0].lower()
        assert mock_st.toast.call_args.kwargs["icon"] == ":material/error:"

    def test_toast_reports_partial_success(self, mock_deepgram_cls, mock_st):
        mock_client = mock_deepgram_cls.return_value

        def fake_transcribe(request, **_):
            if request == b"bad":
                raise Exception("boom")
            return MagicMock()

        mock_client.listen.v1.media.transcribe_file.side_effect = fake_transcribe

        streamlit_app._process_inputs(
            "test-key", [("ok.wav", b"ok"), ("bad.wav", b"bad")]
        )

        msg = mock_st.toast.call_args.args[0]
        assert "1/2" in msg
        assert "failed" in msg.lower()
        assert mock_st.toast.call_args.kwargs["icon"] == ":material/warning:"


class TestRun:
    def test_uploads_take_priority(self, mock_deepgram_cls, mock_st):
        rec = MagicMock()
        rec.getvalue.return_value = wav_bytes(1)
        streamlit_app._run("key", [mock_upload("a.wav", b"a")], rec)

        media = mock_deepgram_cls.return_value.listen.v1.media
        media.transcribe_file.assert_called_once()
        assert mock_st.session_state["responses"][0][0] == "a.wav"

    def test_recording_used_when_no_files(self, mock_deepgram_cls, mock_st):
        rec = MagicMock()
        rec.getvalue.return_value = wav_bytes(1)
        streamlit_app._run("key", [], rec)

        media = mock_deepgram_cls.return_value.listen.v1.media
        media.transcribe_file.assert_called_once()
        assert mock_st.session_state["responses"][0][0] == "Recording"

    @pytest.mark.parametrize("source", ["upload", "record"])
    def test_option_warning_shown_before_transcribing(
        self, mock_deepgram_cls, mock_st, source
    ):
        mock_st.session_state.update({"dictation": True, "diarize": True})
        rec = MagicMock()
        rec.getvalue.return_value = wav_bytes(1)
        args = {
            "upload": ([mock_upload("a.wav", b"a")], None),
            "record": ([], rec),
        }[source]
        streamlit_app._run("key", *args)

        (expected,) = streamlit_app.option_warnings(dictation=True, diarize=True)
        mock_st.warning.assert_called_once_with(expected, icon=":material/warning:")
        # Shown ahead of the batch's st.status region, whichever input runs.
        names = [call[0] for call in mock_st.mock_calls]
        assert names.index("warning") < names.index("status")
        # Advisory only: the run still goes ahead, with the options as set.
        method = mock_deepgram_cls.return_value.listen.v1.media.transcribe_file
        method.assert_called_once()
        assert method.call_args.kwargs["dictation"] is True

    def test_no_input_is_noop(self, mock_deepgram_cls, mock_st):
        streamlit_app._run("key", [], None)

        mock_st.error.assert_not_called()
        mock_st.warning.assert_not_called()
        mock_deepgram_cls.assert_not_called()
        assert "responses" not in mock_st.session_state
        assert "audio_sources" not in mock_st.session_state

    def test_too_many_files_errors_and_skips(self, mock_deepgram_cls, mock_st):
        files = [
            mock_upload(f"f{i}.wav", b"x") for i in range(streamlit_app.MAX_UPLOADS + 1)
        ]
        streamlit_app._run("key", files, None)

        mock_st.error.assert_called_once_with(
            "Too many files. Maximum is 100 per batch.", icon=":material/error:"
        )
        media = mock_deepgram_cls.return_value.listen.v1.media
        media.transcribe_file.assert_not_called()

    def test_oversized_files_skipped_but_others_run(self, mock_deepgram_cls, mock_st):
        big = mock_upload("big.wav", b"x", size=streamlit_app.MAX_FILE_SIZE + 1)
        ok = mock_upload("ok.wav", b"ok")
        streamlit_app._run("key", [big, ok], None)

        mock_st.error.assert_called_once_with(
            "Skipped (exceeds 200 MB): big.wav", icon=":material/error:"
        )
        media = mock_deepgram_cls.return_value.listen.v1.media
        media.transcribe_file.assert_called_once()
        assert mock_st.session_state["responses"][0][0] == "ok.wav"

    def test_recording_too_long_errors(self, mock_deepgram_cls, mock_st):
        rec = MagicMock()
        rec.getvalue.return_value = wav_bytes(streamlit_app.MAX_RECORDING_SECONDS + 100)
        streamlit_app._run("key", [], rec)

        mock_st.error.assert_called_once_with(
            "Recording exceeds the 30-minute limit.", icon=":material/error:"
        )
        media = mock_deepgram_cls.return_value.listen.v1.media
        media.transcribe_file.assert_not_called()

    def test_recording_at_exact_limit_is_accepted(self, mock_deepgram_cls, mock_st):
        rec = MagicMock()
        rec.getvalue.return_value = wav_bytes(streamlit_app.MAX_RECORDING_SECONDS)
        streamlit_app._run("key", [], rec)

        mock_st.error.assert_not_called()
        media = mock_deepgram_cls.return_value.listen.v1.media
        media.transcribe_file.assert_called_once()

    def test_unreadable_recording_errors(self, mock_deepgram_cls, mock_st):
        rec = MagicMock()
        rec.getvalue.return_value = b"not-a-wav"
        streamlit_app._run("key", [], rec)

        mock_st.error.assert_called_once_with(
            "Could not read the recording.", icon=":material/error:"
        )
        media = mock_deepgram_cls.return_value.listen.v1.media
        media.transcribe_file.assert_not_called()

    def test_multiple_inputs_notify_and_keep_priority(self, mock_deepgram_cls, mock_st):
        rec = MagicMock()
        rec.getvalue.return_value = wav_bytes(1)
        streamlit_app._run("key", [mock_upload("a.wav", b"a")], rec)

        info = mock_st.info.call_args.args[0]
        assert "(priority: Upload > Record)" in info
        assert mock_st.info.call_args.kwargs["icon"] == ":material/info:"
        media = mock_deepgram_cls.return_value.listen.v1.media
        media.transcribe_file.assert_called_once()
        assert mock_st.session_state["responses"][0][0] == "a.wav"

    def test_single_input_no_notice(self, mock_deepgram_cls, mock_st):
        streamlit_app._run("key", [mock_upload("a.wav", b"a")], None)

        mock_st.info.assert_not_called()


class TestDisplayAudio:
    def test_bytes_source_uses_mime_from_extension(self, mock_st):
        streamlit_app._display_audio("dictation.mp3", b"audio-bytes")

        mock_st.audio.assert_called_once_with(b"audio-bytes", format="audio/mpeg")

    def test_bytes_source_without_extension_defaults_to_wav(self, mock_st):
        streamlit_app._display_audio("Recording", b"wav-bytes")

        mock_st.audio.assert_called_once_with(b"wav-bytes", format="audio/wav")


class TestDisplayTranscript:
    # The conftest response's 0.85 ("moves") and 0.80 ("fast") words fall below the
    # 0.90 threshold, so they render flagged in bold orange.
    FLAGGED = "Life :orange[**moves**] pretty :orange[**fast**] really."

    def test_renders_plain_transcript(self, mock_deepgram_cls, mock_st):
        response = (
            mock_deepgram_cls.return_value.listen.v1.media.transcribe_file.return_value
        )

        streamlit_app._display_transcript(response)

        mock_st.markdown.assert_called_once_with(self.FLAGGED)

    def test_flat_transcript_has_no_raw_html(self, mock_deepgram_cls, mock_st):
        # The non-diarized path renders one Markdown string per paragraph: native
        # color directives for the flags, never raw HTML.
        response = (
            mock_deepgram_cls.return_value.listen.v1.media.transcribe_file.return_value
        )

        streamlit_app._display_transcript(response)

        (markdown_arg,), markdown_kwargs = mock_st.markdown.call_args
        assert markdown_arg == self.FLAGGED
        assert "unsafe_allow_html" not in markdown_kwargs
        assert "<" not in markdown_arg

    def test_escapes_markdown_metacharacters(self, mock_st):
        response = MagicMock()
        response.results.channels[0].alternatives[0].transcript = "take *2* `mg` of x_y"

        streamlit_app._display_transcript(response)

        mock_st.markdown.assert_called_once_with("take \\*2\\* \\`mg\\` of x\\_y")

    def test_missing_results_renders_no_transcript_notice(self, mock_st):
        # A callback/async ListenV1AcceptedResponse has only request_id, no results.
        response = MagicMock(spec=["request_id"])
        response.request_id = "req-123"

        streamlit_app._display_transcript(response)

        mock_st.markdown.assert_not_called()
        mock_st.caption.assert_called_once_with(streamlit_app.NO_TRANSCRIPT)

    def test_empty_channels_renders_no_transcript_notice(self, mock_st):
        response = MagicMock()
        response.results.channels = []

        streamlit_app._display_transcript(response)

        mock_st.markdown.assert_not_called()
        mock_st.caption.assert_called_once_with(streamlit_app.NO_TRANSCRIPT)


class TestDiarizedTranscript:
    """Diarized rendering via _display_transcript — speaker labels are 1-based for display
    (the core's diarized_segments stays 0-based; that is tested in test_results.py)."""

    @staticmethod
    def _response(words):
        response = MagicMock()
        response.results.channels = [MagicMock(alternatives=[MagicMock(words=words)])]
        return response

    def test_groups_consecutive_speaker_runs(self, mock_st):
        words = [
            mock_word("Hello", 0.9, speaker=0),
            mock_word("doctor.", 0.9, speaker=0),
            mock_word("Hi", 0.9, speaker=1),
            mock_word("there.", 0.9, speaker=1),
            mock_word("Yes?", 0.9, speaker=0),
        ]

        streamlit_app._display_transcript(self._response(words))

        rendered = [c.args[0] for c in mock_st.markdown.call_args_list]
        assert rendered == [
            ":blue-background[**Speaker 1:**] Hello doctor.",
            ":green-background[**Speaker 2:**] Hi there.",
            ":blue-background[**Speaker 1:**] Yes?",
        ]
        # Nothing scored below the threshold, and the caption says so.
        mock_st.caption.assert_called_once_with(streamlit_app.NO_FLAGS)

    def test_single_speaker_renders_one_labeled_line(self, mock_st):
        words = [mock_word("Note.", 0.9, speaker=0), mock_word("Done.", 0.9, speaker=0)]

        streamlit_app._display_transcript(self._response(words))

        mock_st.markdown.assert_called_once_with(
            ":blue-background[**Speaker 1:**] Note. Done."
        )

    def test_speaker_text_is_markdown_escaped(self, mock_st):
        words = [mock_word("take *2*", 0.9, speaker=0)]

        streamlit_app._display_transcript(self._response(words))

        mock_st.markdown.assert_called_once_with(
            ":blue-background[**Speaker 1:**] take \\*2\\*"
        )

    def test_unlabeled_word_continues_current_run(self, mock_st):
        # A mid-stream word missing an integer speaker is absorbed into the current
        # run rather than opening a bogus "Speaker None" segment.
        words = [
            mock_word("Patient", 0.9, speaker=0),
            mock_word("reports", 0.9, speaker=None),
            mock_word("pain.", 0.9, speaker=0),
        ]

        streamlit_app._display_transcript(self._response(words))

        mock_st.markdown.assert_called_once_with(
            ":blue-background[**Speaker 1:**] Patient reports pain."
        )

    def test_falls_back_to_word_when_no_punctuated_word(self, mock_st):
        word = MagicMock()
        word.punctuated_word = None
        word.word = "stat"
        word.speaker = 0
        word.confidence = 0.9  # a MagicMock confidence would be flagged

        streamlit_app._display_transcript(self._response([word]))

        mock_st.markdown.assert_called_once_with(
            ":blue-background[**Speaker 1:**] stat"
        )

    def test_no_speaker_labels_falls_back_to_flat_transcript(self, mock_st):
        # Words without integer speakers (diarize off) -> flat transcript path.
        alt = MagicMock(words=[mock_word("plain words", 0.9)])
        alt.transcript = "plain words"
        response = MagicMock()
        response.results.channels = [MagicMock(alternatives=[alt])]

        streamlit_app._display_transcript(response)

        mock_st.markdown.assert_called_once_with("plain words")

    def test_speaker_label_color_cycles_by_index(self, mock_st):
        # Every _SPEAKER_COLORS entry is exercised (speakers 0-5), and the cycle wraps
        # (speaker 6 -> blue again).
        words = [
            mock_word("a.", 0.9, speaker=0),
            mock_word("b.", 0.9, speaker=1),
            mock_word("c.", 0.9, speaker=2),
            mock_word("d.", 0.9, speaker=3),
            mock_word("e.", 0.9, speaker=4),
            mock_word("f.", 0.9, speaker=5),
            mock_word("g.", 0.9, speaker=6),
        ]

        streamlit_app._display_transcript(self._response(words))

        rendered = [c.args[0] for c in mock_st.markdown.call_args_list]
        assert rendered == [
            ":blue-background[**Speaker 1:**] a.",
            ":green-background[**Speaker 2:**] b.",
            ":violet-background[**Speaker 3:**] c.",
            ":orange-background[**Speaker 4:**] d.",
            ":red-background[**Speaker 5:**] e.",
            ":gray-background[**Speaker 6:**] f.",
            ":blue-background[**Speaker 7:**] g.",
        ]


class TestLowConfidenceFlags:
    """Low-confidence words render as `:orange[**…**]` under a caption; the flagging
    rule itself (threshold, redaction tags, fidelity guard) is tested in
    test_results.py."""

    @staticmethod
    def _response(words, transcript=None):
        alt = MagicMock(words=words)
        alt.transcript = transcript
        response = MagicMock()
        response.results.channels = [MagicMock(alternatives=[alt])]
        return response

    def test_flagged_token_inside_diarized_line(self, mock_st):
        words = [
            mock_word("Take", 0.95, speaker=0),
            mock_word("50", 0.42, speaker=0),
            mock_word("mg.", 0.97, speaker=0),
        ]

        streamlit_app._display_transcript(self._response(words))

        mock_st.markdown.assert_called_once_with(
            ":blue-background[**Speaker 1:**] Take :orange[**50**] mg."
        )

    def test_flagged_token_is_escaped(self, mock_st):
        words = [mock_word("x_y*", 0.5)]

        streamlit_app._display_transcript(self._response(words, "x_y*"))

        mock_st.markdown.assert_called_once_with(":orange[**x\\_y\\***]")

    def test_legend_caption_when_something_is_flagged(self, mock_st):
        words = [mock_word("Take", 0.95), mock_word("50", 0.42)]

        streamlit_app._display_transcript(self._response(words, "Take 50"))

        mock_st.caption.assert_called_once_with(streamlit_app.LOW_CONFIDENCE_LEGEND)

    def test_no_flags_caption_when_nothing_is_flagged(self, mock_st):
        words = [mock_word("Take", 0.95), mock_word("50", 0.99)]

        streamlit_app._display_transcript(self._response(words, "Take 50"))

        mock_st.caption.assert_called_once_with(streamlit_app.NO_FLAGS)
        mock_st.markdown.assert_called_once_with("Take 50")

    def test_captions_state_the_threshold_and_residual_risk(self):
        for caption in (streamlit_app.LOW_CONFIDENCE_LEGEND, streamlit_app.NO_FLAGS):
            assert "90%" in caption
            assert "can still be wrong" in caption
        # Plain caption text: a colored sample would fail contrast at caption opacity.
        assert ":orange[" not in streamlit_app.LOW_CONFIDENCE_LEGEND

    def test_missing_words_fall_back_to_plain_transcript(self, mock_st):
        response = self._response([], "Take 50 mg.")

        streamlit_app._display_transcript(response)

        mock_st.caption.assert_called_once_with(streamlit_app.NO_CONFIDENCE)
        mock_st.markdown.assert_called_once_with("Take 50 mg.")

    def test_transcript_mismatch_falls_back_to_plain_transcript(self, mock_st):
        # Words that do not reproduce the transcript must never be shown in its place.
        words = [mock_word("Take", 0.4), mock_word("5", 0.4)]
        response = self._response(words, "Take 50 mg.")

        streamlit_app._display_transcript(response)

        mock_st.caption.assert_called_once_with(streamlit_app.NO_CONFIDENCE)
        mock_st.markdown.assert_called_once_with("Take 50 mg.")

    def test_paragraphs_render_as_separate_markdown_calls(self, mock_st):
        words = [
            mock_word("First.", 0.99),
            mock_word("Second", 0.3),
            mock_word("para.", 0.99),
        ]

        streamlit_app._display_transcript(
            self._response(words, "First.\n\nSecond para.")
        )

        rendered = [c.args[0] for c in mock_st.markdown.call_args_list]
        assert rendered == ["First.", ":orange[**Second**] para."]

    def test_currency_dollar_signs_are_escaped(self, mock_st):
        # Streamlit's Markdown enables single-dollar math, so "$20-$30" would
        # otherwise render as a formula with its dollar signs dropped.
        words = [mock_word("Copay", 0.99), mock_word("$20-$30.", 0.99)]

        streamlit_app._display_transcript(self._response(words, "Copay $20-$30."))

        mock_st.markdown.assert_called_once_with("Copay \\$20-\\$30.")

    def test_plain_fallback_escapes_dollar_signs(self, mock_st):
        streamlit_app._display_transcript(self._response([], "$20-$30"))

        mock_st.markdown.assert_called_once_with("\\$20-\\$30")


class TestOutputPanel:
    """The panel's layout; `_display_transcript` is patched out (tested above)."""

    @pytest.fixture
    def render(self):
        with patch.object(streamlit_app, "_display_transcript") as render:
            yield render

    def test_shows_placeholder_when_empty(self, mock_st, render):
        streamlit_app._output_panel([], [])

        mock_st.caption.assert_called_once_with(streamlit_app.PLACEHOLDER)
        render.assert_not_called()

    def test_single_result_has_player_and_no_divider(self, mock_st, render):
        response = MagicMock()

        streamlit_app._output_panel([("a.mp3", response)], [b"a"])

        render.assert_called_once_with(response)
        mock_st.audio.assert_called_once()
        mock_st.divider.assert_not_called()
        mock_st.caption.assert_not_called()

    def test_multiple_results_labeled_with_dividers(self, mock_st, render):
        responses = [("a.mp3", MagicMock()), ("b.mp3", MagicMock())]

        streamlit_app._output_panel(responses, [b"a", b"b"])

        assert render.call_count == 2
        assert mock_st.audio.call_count == 2
        mock_st.divider.assert_called_once()
        labels = [c.args[0] for c in mock_st.markdown.call_args_list]
        assert any("a.mp3" in m for m in labels)
        assert any("b.mp3" in m for m in labels)

    def test_single_none_source_renders_no_player(self, mock_st, render):
        response = MagicMock()

        streamlit_app._output_panel([("big.wav", response)], [None])

        mock_st.audio.assert_not_called()
        mock_st.caption.assert_called_once_with(streamlit_app.PLAYBACK_TOO_LARGE)
        render.assert_called_once_with(response)

    def test_none_source_skipped_among_multiple(self, mock_st, render):
        responses = [("big.wav", MagicMock()), ("small.wav", MagicMock())]

        streamlit_app._output_panel(responses, [None, b"a"])

        mock_st.audio.assert_called_once_with(b"a", format="audio/wav")
        mock_st.caption.assert_called_once_with(streamlit_app.PLAYBACK_TOO_LARGE)
        assert render.call_count == 2


class TestFeatureOpts:
    def test_defaults_when_session_empty(self, mock_st):
        assert streamlit_app._feature_opts() == {
            "keyterms": [],
            "language": "en",
            "smart_format": True,
            "dictation": False,
            "measurements": False,
            "diarize": False,
            "redact": [],
        }

    def test_reads_values_from_session_state(self, mock_st):
        mock_st.session_state.update(
            {
                "keyterms": ["metformin"],
                "language": "en-GB",
                "smart_format": False,
                "dictation": True,
                "measurements": True,
                "diarize": True,
                "redact": ["phi", "pii"],
            }
        )

        assert streamlit_app._feature_opts() == {
            "keyterms": ["metformin"],
            "language": "en-GB",
            "smart_format": False,
            "dictation": True,
            "measurements": True,
            "diarize": True,
            "redact": ["phi", "pii"],
        }

    def test_partial_session_state_mixes_values_and_defaults(self, mock_st):
        mock_st.session_state.update({"language": "en-GB", "diarize": True})

        assert streamlit_app._feature_opts() == {
            "keyterms": [],
            "language": "en-GB",
            "smart_format": True,
            "dictation": False,
            "measurements": False,
            "diarize": True,
            "redact": [],
        }


class TestMetrics:
    """Per-result Duration / Confidence cards rendered above each transcript."""

    def test_renders_duration_and_confidence(self, mock_deepgram_cls, mock_st):
        response = (
            mock_deepgram_cls.return_value.listen.v1.media.transcribe_file.return_value
        )

        streamlit_app._display_metrics(response)

        # Assert the formatted value strings, not just the labels — the percent
        # conversion and unit/precision formatting is the logic under test.
        values = {c.args[0]: c.args[1] for c in mock_st.metric.call_args_list}
        assert values == {
            "Duration": "3.5 s",
            "Confidence": "98.0%",
            "Low-confidence words": "2",
        }

    def test_no_metrics_when_response_has_no_results(self, mock_st):
        # A results-less response (no metadata duration, no alternative confidence)
        # renders no metric cards rather than blank/garbage values.
        response = MagicMock(spec=["request_id"])

        streamlit_app._display_metrics(response)

        mock_st.metric.assert_not_called()

    def test_no_count_card_when_highlighting_is_unavailable(self, mock_st):
        # Words that cannot reproduce the transcript -> no highlighting, so no count
        # card (a missing count must never read as "0 words to check").
        alt = MagicMock(words=[mock_word("Other", 0.5)])
        alt.transcript = "Take 50 mg."
        alt.confidence = 0.9
        response = MagicMock()
        response.metadata.duration = 2.0
        response.results.channels = [MagicMock(alternatives=[alt])]

        streamlit_app._display_metrics(response)

        values = {c.args[0]: c.args[1] for c in mock_st.metric.call_args_list}
        assert values == {"Duration": "2.0 s", "Confidence": "90.0%"}

    def test_renders_only_the_available_metric(self, mock_st):
        # Duration present but no numeric confidence -> only the Duration card.
        alt = MagicMock()
        alt.confidence = "n/a"
        response = MagicMock()
        response.metadata.duration = 4.0
        response.results.channels = [MagicMock(alternatives=[alt])]

        streamlit_app._display_metrics(response)

        values = {c.args[0]: c.args[1] for c in mock_st.metric.call_args_list}
        assert values == {"Duration": "4.0 s"}


class TestTranscriptDownload:
    """The plain-text transcript download button above the output panel."""

    def test_no_button_when_no_responses(self, mock_st):
        streamlit_app._transcript_download([])

        mock_st.download_button.assert_not_called()

    def test_button_carries_transcript_text(self, mock_deepgram_cls, mock_st):
        response = (
            mock_deepgram_cls.return_value.listen.v1.media.transcribe_file.return_value
        )

        streamlit_app._transcript_download([("a.wav", response)])

        mock_st.download_button.assert_called_once()
        build_transcript = mock_st.download_button.call_args.args[1]
        assert callable(build_transcript)  # deferred: built on click, not per rerun
        text = build_transcript()
        assert "a.wav" in text
        assert "Life moves pretty fast really." in text

    def test_diarized_export_uses_plain_speaker_lines(self):
        # The export is plain text (no color directives): "Speaker N: ..." per turn.
        words = [
            mock_word("Hello.", 0.9, speaker=0),
            mock_word("Hi.", 0.9, speaker=1),
        ]
        response = MagicMock()
        response.results.channels = [MagicMock(alternatives=[MagicMock(words=words)])]

        assert (
            streamlit_app._plain_transcript(response)
            == "Speaker 1: Hello.\nSpeaker 2: Hi."
        )

    def test_multiple_results_combined_into_one_file(self, mock_st):
        # A multi-result batch exports one combined plain-text file (both present).
        def _diarized(token):
            words = [mock_word(token, 0.9, speaker=0)]
            response = MagicMock()
            response.results.channels = [
                MagicMock(alternatives=[MagicMock(words=words)])
            ]
            return response

        streamlit_app._transcript_download(
            [("a.wav", _diarized("Alpha.")), ("b.wav", _diarized("Beta."))]
        )

        mock_st.download_button.assert_called_once()
        combined = mock_st.download_button.call_args.args[1]()
        assert "a.wav" in combined and "b.wav" in combined
        assert "Alpha." in combined and "Beta." in combined


class TestSecretApiKey:
    """`_secret_api_key` guards a `st.secrets` read that raises when unconfigured."""

    def test_missing_secrets_file_returns_empty(self):
        # st.secrets raises on *every* access when no secrets.toml exists — the
        # normal local setup here, where the key comes from .env instead.
        secrets = MagicMock()
        secrets.get.side_effect = streamlit_app.StreamlitSecretNotFoundError("none")
        with patch.object(streamlit_app.st, "secrets", secrets):
            assert streamlit_app._secret_api_key() == ""

    def test_configured_secret_is_returned(self):
        secrets = MagicMock()
        secrets.get.return_value = "sk-from-secrets"
        with patch.object(streamlit_app.st, "secrets", secrets):
            assert streamlit_app._secret_api_key() == "sk-from-secrets"

    def test_non_string_secret_is_rejected(self):
        # A TOML table (or number) under the key must never reach DeepgramClient.
        secrets = MagicMock()
        secrets.get.return_value = {"nested": "table"}
        with patch.object(streamlit_app.st, "secrets", secrets):
            assert streamlit_app._secret_api_key() == ""


class TestAppSmoke:
    """Run the whole script under a real Streamlit runtime (not the mock).

    Catches the class of errors the whole-module ``mock_st`` MagicMock cannot.
    Four runs (a known key is seeded for the first three so they don't depend on a
    local ``.env``):

    - **empty state** — module load (``set_page_config`` ordering, the ``st.form``
      structure, the output fragment) plus the idle UI: the placeholder caption, a
      disabled Run button, the inputs | output column split, and the Features control
      order
      (``language, keyterms, smart_format, diarize, dictation, measurements, redact``).
    - **seeded diarized** — renders the transcript panel for a diarized result,
      asserting the exact 1-based color-directive speaker lines, the
      Duration/Confidence/Low-confidence words metric cards, the nothing-flagged
      caption, and the dropped-playback caption (the ``download_button`` icon also
      runs here).
    - **seeded flat** — the non-diarized render branch: an escaped transcript with
      no speaker labels, its one low-confidence word in bold orange under the legend
      caption.
    - **no-key state** — clears ``DEEPGRAM_API_KEY`` so the key-required warning and
      the API-Key input render.

    No Deepgram call happens — nothing clicks Run — so it never touches the network;
    the toast/per-item status icons fire only on a real batch and are not exercised.
    """

    def test_script_renders_clean_empty_and_seeded(self):
        # Run in a SEPARATE process: the function tests `import streamlit_app` at
        # module scope, which executes its module-level `st.form` once in bare mode
        # and leaves Streamlit's form-context state dirty in-process (a spurious
        # "forms cannot be nested" on a later in-process run). A clean subprocess
        # exercises the real script faithfully.
        import os
        import subprocess
        import sys

        root = os.path.dirname(os.path.dirname(__file__))
        app = os.path.join(root, "streamlit_app.py")
        code = """
import os
import sys
from unittest.mock import MagicMock
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import Block

app = sys.argv[1]

# Seed a key so the first three runs are deterministic regardless of a local .env
# (load_dotenv does not override an already-set var); run #4 clears it.
os.environ["DEEPGRAM_API_KEY"] = "test-key"


def _word(text, speaker, confidence=0.9):
    w = MagicMock()
    w.punctuated_word = text
    w.word = text
    w.speaker = speaker
    w.confidence = confidence
    return w


def _resp(transcript, words, duration, confidence):
    alt = MagicMock()
    alt.transcript = transcript
    alt.confidence = confidence
    alt.words = words
    r = MagicMock()
    r.metadata.duration = duration
    r.results.channels = [MagicMock(alternatives=[alt])]
    return r


def _widget_keys_in_order(node, acc):
    # Depth-first walk of the element tree, collecting widget keys in document order.
    # Containers are skipped: since Streamlit 1.64 the st.form Block carries its form
    # id ("features") as .key, which is a container, not a Features control.
    for child in getattr(node, "children", {}).values():
        if not isinstance(child, Block) and getattr(child, "key", None):
            acc.append(child.key)
        _widget_keys_in_order(child, acc)


# 1) Empty state — module load (set_page_config / form / output fragment) plus the
#    idle UI: the placeholder caption and a Run button disabled with no input selected.
at = AppTest.from_file(app, default_timeout=30).run()
assert not at.exception, at.exception
assert at.title[0].value == "Deepgram Medical Transcription"
assert any("Select audio, then click Run" in c.value for c in at.caption), [c.value for c in at.caption]
run = [b for b in at.button if b.label == "Run"]
assert run and run[0].disabled, "Run should be disabled with no audio input"
# Key was seeded, so no api-key warning fires — this pins the disable to the
# no-input branch (`not has_input`), not the no-key branch (`not api_key`).
assert not at.warning, [w.value for w in at.warning]

# Main area is a side-by-side split: the audio input tabs on the left; on the right the
# output panel under its Transcript caption header (no output tabs) with its
# empty-state placeholder — the output is not stacked under the inputs.
inputs, output = at.main.columns
assert [t.label for t in inputs.tabs] == [
    ":material/upload: Upload",
    ":material/mic: Record",
]
assert not output.tabs, [t.label for t in output.tabs]
captions = [c.value for c in output.caption]
assert captions[0] == ":material/description: Transcript", captions
assert any("Select audio, then click Run" in c for c in captions[1:]), captions

# Features live in the sidebar and render in the intended order: inputs (Language,
# Keyterm) first, the four toggles grouped, Redact deliberately last.
order = []
_widget_keys_in_order(at.sidebar, order)
assert [k for k in order if not k.startswith("FormSubmitter")] == [
    "language",
    "keyterms",
    "smart_format",
    "diarize",
    "dictation",
    "measurements",
    "redact",
], order

# 2) Seeded diarized result — asserts the real rendered output: 1-based,
#    color-highlighted speaker lines; Duration + Confidence + Low-confidence words
#    metric cards; the nothing-flagged caption (0.9 is not below the 0.90 threshold);
#    and the dropped-playback caption (audio_source is None).
diar = _resp("Hello. Hi.", [_word("Hello.", 0), _word("Hi.", 1)], 3.5, 0.95)
seeded = AppTest.from_file(app, default_timeout=30)
seeded.session_state["responses"] = [("sample.wav", diar)]
seeded.session_state["audio_sources"] = [None]
seeded.run()
assert not seeded.exception, seeded.exception
assert [m.value for m in seeded.markdown] == [
    ":blue-background[**Speaker 1:**] Hello.",
    ":green-background[**Speaker 2:**] Hi.",
], [m.value for m in seeded.markdown]
assert [m.label for m in seeded.metric] == ["Duration", "Confidence", "Low-confidence words"]
assert [m.value for m in seeded.metric] == ["3.5 s", "95.0%", "0"]
assert any("No words scored below 90%" in c.value for c in seeded.caption), [c.value for c in seeded.caption]
assert any("Inline playback unavailable" in c.value for c in seeded.caption)

# 3) Seeded flat (non-diarized) result — the other render branch: an escaped
#    transcript with no speaker labels, rebuilt from its words so the one
#    low-confidence word renders in bold orange under the legend caption.
flat_words = [_word("Patient", None), _word("is", None), _word("stable.", None, 0.5)]
flat = _resp("Patient is stable.", flat_words, 12.0, 0.88)
flat_at = AppTest.from_file(app, default_timeout=30)
flat_at.session_state["responses"] = [("note.wav", flat)]
flat_at.session_state["audio_sources"] = [None]
flat_at.run()
assert not flat_at.exception, flat_at.exception
assert [m.value for m in flat_at.markdown] == ["Patient is :orange[**stable.**]"]
assert not any("Speaker" in m.value for m in flat_at.markdown)
assert any("bold orange" in c.value for c in flat_at.caption), [c.value for c in flat_at.caption]
assert [m.value for m in flat_at.metric][-1] == "1"

# 4) No-key state — clear the key, no-op load_dotenv, and empty the secrets search
#    path, so neither a local .env nor a developer's ~/.streamlit/secrets.toml can
#    repopulate it; the key-required warning + API-key input must render.
import dotenv
from streamlit import config

dotenv.load_dotenv = lambda *a, **k: False
config.set_option("secrets.files", [])
os.environ.pop("DEEPGRAM_API_KEY", None)
nokey = AppTest.from_file(app, default_timeout=30).run()
assert not nokey.exception, nokey.exception
assert any("API key required" in w.value for w in nokey.warning), [w.value for w in nokey.warning]
assert nokey.text_input  # the API-Key password input renders when the key is missing
"""
        result = subprocess.run(
            [sys.executable, "-c", code, app],
            cwd=root,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
