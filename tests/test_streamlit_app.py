from unittest.mock import MagicMock, patch

import streamlit_app
from tests.helpers import mock_upload, mock_word, wav_bytes

FAKE_AUDIO = b"fake-audio-data"

# Option building (build_options) and the batch runner (transcribe_batch) are tested
# directly in tests/test_transcribe.py; the response walkers in tests/test_results.py.
# This module covers only the Streamlit-side behavior: the _transcribe_batch wrapper's
# progress / error / session-state handling, _run validation, _feature_opts, and the
# renderers.


class TestParseUrls:
    def test_empty_text_returns_no_urls(self):
        valid, invalid = streamlit_app._parse_urls("")
        assert valid == []
        assert invalid == []

    def test_blank_lines_are_skipped(self):
        valid, invalid = streamlit_app._parse_urls("  \n\n  \n")
        assert valid == []
        assert invalid == []

    def test_valid_http_url(self):
        valid, invalid = streamlit_app._parse_urls("http://example.com/audio.wav")
        assert valid == ["http://example.com/audio.wav"]
        assert invalid == []

    def test_valid_https_url(self):
        valid, invalid = streamlit_app._parse_urls("https://example.com/audio.wav")
        assert valid == ["https://example.com/audio.wav"]
        assert invalid == []

    def test_invalid_protocol_rejected(self):
        valid, invalid = streamlit_app._parse_urls("ftp://example.com/audio.wav")
        assert valid == []
        assert invalid == ["ftp://example.com/audio.wav"]

    def test_mixed_valid_and_invalid(self):
        text = (
            "https://example.com/a.wav\nftp://bad.com/b.wav\nhttp://example.com/c.mp3"
        )
        valid, invalid = streamlit_app._parse_urls(text)
        assert valid == ["https://example.com/a.wav", "http://example.com/c.mp3"]
        assert invalid == ["ftp://bad.com/b.wav"]


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
            redact=["pii"],
        )

        kwargs = mock_deepgram_cls.return_value.listen.v1.media.transcribe_file.call_args.kwargs
        assert kwargs["keyterm"] == ["metformin"]
        assert kwargs["language"] == "en-GB"
        assert kwargs["diarize"] is True
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


class TestProcessUrls:
    """URL-specific wrapper behavior: labels and playback sources are the URLs."""

    def test_stores_responses_in_session_state(self, mock_deepgram_cls, mock_st):
        streamlit_app._process_urls("test-key", ["https://example.com/test.wav"])

        responses = mock_st.session_state["responses"]
        assert len(responses) == 1
        assert responses[0][0] == "https://example.com/test.wav"

    def test_stores_audio_sources_as_urls(self, mock_deepgram_cls, mock_st):
        urls = ["https://example.com/a.wav", "https://example.com/b.wav"]
        streamlit_app._process_urls("test-key", urls)

        assert mock_st.session_state["audio_sources"] == urls

    def test_continues_after_single_url_failure(self, mock_deepgram_cls, mock_st):
        mock_client = mock_deepgram_cls.return_value
        good_response = MagicMock()

        def fake_transcribe(url, **_):
            if url == "https://example.com/bad.wav":
                raise Exception("API error")
            return good_response

        mock_client.listen.v1.media.transcribe_url.side_effect = fake_transcribe

        streamlit_app._process_urls(
            "test-key",
            ["https://example.com/bad.wav", "https://example.com/good.wav"],
        )

        mock_st.error.assert_called_once_with(
            "Transcription failed for https://example.com/bad.wav: API error",
            icon=":material/error:",
        )
        assert mock_st.session_state["responses"] == [
            ("https://example.com/good.wav", good_response)
        ]
        assert mock_st.session_state["audio_sources"] == [
            "https://example.com/good.wav"
        ]


class TestRun:
    def test_uploads_take_priority(self, mock_deepgram_cls, mock_st):
        rec = MagicMock()
        rec.getvalue.return_value = wav_bytes(1)
        streamlit_app._run(
            "key", [mock_upload("a.wav", b"a")], rec, "https://example.com/x.wav"
        )

        media = mock_deepgram_cls.return_value.listen.v1.media
        media.transcribe_file.assert_called_once()
        media.transcribe_url.assert_not_called()

    def test_recording_used_when_no_files(self, mock_deepgram_cls, mock_st):
        rec = MagicMock()
        rec.getvalue.return_value = wav_bytes(1)
        streamlit_app._run("key", [], rec, "")

        media = mock_deepgram_cls.return_value.listen.v1.media
        media.transcribe_file.assert_called_once()
        assert mock_st.session_state["responses"][0][0] == "Recording"

    def test_urls_used_when_no_files_or_recording(self, mock_deepgram_cls, mock_st):
        streamlit_app._run("key", [], None, "https://example.com/x.wav")

        media = mock_deepgram_cls.return_value.listen.v1.media
        media.transcribe_url.assert_called_once()
        media.transcribe_file.assert_not_called()

    def test_no_input_is_noop(self, mock_deepgram_cls, mock_st):
        streamlit_app._run("key", [], None, "   ")

        mock_st.error.assert_not_called()
        mock_st.warning.assert_not_called()
        mock_deepgram_cls.assert_not_called()
        assert "responses" not in mock_st.session_state
        assert "audio_sources" not in mock_st.session_state

    def test_too_many_files_errors_and_skips(self, mock_deepgram_cls, mock_st):
        files = [
            mock_upload(f"f{i}.wav", b"x") for i in range(streamlit_app.MAX_UPLOADS + 1)
        ]
        streamlit_app._run("key", files, None, "")

        mock_st.error.assert_called_once_with(
            "Too many files. Maximum is 100 per batch.", icon=":material/error:"
        )
        media = mock_deepgram_cls.return_value.listen.v1.media
        media.transcribe_file.assert_not_called()

    def test_oversized_files_skipped_but_others_run(self, mock_deepgram_cls, mock_st):
        big = mock_upload("big.wav", b"x", size=streamlit_app.MAX_FILE_SIZE + 1)
        ok = mock_upload("ok.wav", b"ok")
        streamlit_app._run("key", [big, ok], None, "")

        mock_st.error.assert_called_once_with(
            "Skipped (exceeds 200 MB): big.wav", icon=":material/error:"
        )
        media = mock_deepgram_cls.return_value.listen.v1.media
        media.transcribe_file.assert_called_once()
        assert mock_st.session_state["responses"][0][0] == "ok.wav"

    def test_recording_too_long_errors(self, mock_deepgram_cls, mock_st):
        rec = MagicMock()
        rec.getvalue.return_value = wav_bytes(streamlit_app.MAX_RECORDING_SECONDS + 100)
        streamlit_app._run("key", [], rec, "")

        mock_st.error.assert_called_once_with(
            "Recording exceeds the 10-minute limit.", icon=":material/error:"
        )
        media = mock_deepgram_cls.return_value.listen.v1.media
        media.transcribe_file.assert_not_called()

    def test_recording_at_exact_limit_is_accepted(self, mock_deepgram_cls, mock_st):
        rec = MagicMock()
        rec.getvalue.return_value = wav_bytes(streamlit_app.MAX_RECORDING_SECONDS)
        streamlit_app._run("key", [], rec, "")

        mock_st.error.assert_not_called()
        media = mock_deepgram_cls.return_value.listen.v1.media
        media.transcribe_file.assert_called_once()

    def test_unreadable_recording_errors(self, mock_deepgram_cls, mock_st):
        rec = MagicMock()
        rec.getvalue.return_value = b"not-a-wav"
        streamlit_app._run("key", [], rec, "")

        mock_st.error.assert_called_once_with(
            "Could not read the recording.", icon=":material/error:"
        )
        media = mock_deepgram_cls.return_value.listen.v1.media
        media.transcribe_file.assert_not_called()

    def test_invalid_urls_error(self, mock_deepgram_cls, mock_st):
        streamlit_app._run("key", [], None, "ftp://bad.com/a.wav")

        mock_st.error.assert_called_once_with(
            "Invalid URL(s): ftp://bad.com/a.wav", icon=":material/error:"
        )
        media = mock_deepgram_cls.return_value.listen.v1.media
        media.transcribe_url.assert_not_called()

    def test_url_without_audio_extension_warns_but_runs(
        self, mock_deepgram_cls, mock_st
    ):
        streamlit_app._run("key", [], None, "https://example.com/audio")

        warning = mock_st.warning.call_args.args[0]
        assert "https://example.com/audio" in warning
        assert "mp3" in warning
        assert mock_st.warning.call_args.kwargs["icon"] == ":material/warning:"
        media = mock_deepgram_cls.return_value.listen.v1.media
        media.transcribe_url.assert_called_once()

    def test_url_mixed_extension_warns_only_on_extensionless(
        self, mock_deepgram_cls, mock_st
    ):
        streamlit_app._run(
            "key", [], None, "https://example.com/a.mp3\nhttps://example.com/audio"
        )

        warning = mock_st.warning.call_args.args[0]
        assert "https://example.com/audio" in warning
        assert "https://example.com/a.mp3" not in warning
        assert mock_st.warning.call_args.kwargs["icon"] == ":material/warning:"
        media = mock_deepgram_cls.return_value.listen.v1.media
        assert media.transcribe_url.call_count == 2

    def test_url_with_query_string_extension_no_warning(
        self, mock_deepgram_cls, mock_st
    ):
        streamlit_app._run("key", [], None, "https://example.com/audio.mp3?token=x")

        mock_st.warning.assert_not_called()
        media = mock_deepgram_cls.return_value.listen.v1.media
        media.transcribe_url.assert_called_once()

    def test_multiple_inputs_notify_and_keep_priority(self, mock_deepgram_cls, mock_st):
        rec = MagicMock()
        rec.getvalue.return_value = wav_bytes(1)
        streamlit_app._run(
            "key", [mock_upload("a.wav", b"a")], rec, "https://example.com/x.wav"
        )

        info = mock_st.info.call_args.args[0]
        assert "Upload" in info and "Record" in info and "URL" in info
        assert mock_st.info.call_args.kwargs["icon"] == ":material/info:"
        media = mock_deepgram_cls.return_value.listen.v1.media
        media.transcribe_file.assert_called_once()
        media.transcribe_url.assert_not_called()

    def test_single_input_no_notice(self, mock_deepgram_cls, mock_st):
        streamlit_app._run("key", [], None, "https://example.com/x.wav")

        mock_st.info.assert_not_called()


class TestDisplayAudio:
    def test_bytes_source_uses_mime_from_extension(self, mock_st):
        streamlit_app._display_audio("dictation.mp3", b"audio-bytes")

        mock_st.audio.assert_called_once_with(b"audio-bytes", format="audio/mpeg")

    def test_bytes_source_without_extension_defaults_to_wav(self, mock_st):
        streamlit_app._display_audio("Recording", b"wav-bytes")

        mock_st.audio.assert_called_once_with(b"wav-bytes", format="audio/wav")

    def test_url_source_passed_through(self, mock_st):
        streamlit_app._display_audio(
            "https://example.com/a.mp3", "https://example.com/a.mp3"
        )

        mock_st.audio.assert_called_once_with("https://example.com/a.mp3")


class TestDisplayTranscript:
    def test_renders_plain_transcript(self, mock_deepgram_cls, mock_st):
        response = (
            mock_deepgram_cls.return_value.listen.v1.media.transcribe_file.return_value
        )

        streamlit_app._display_transcript(response)

        mock_st.markdown.assert_called_once_with("Life moves pretty fast really.")

    def test_flat_transcript_is_plain_no_raw_html(self, mock_deepgram_cls, mock_st):
        # The non-diarized path renders the transcript as a single plain Markdown
        # string: no color directives, no raw HTML, and no JSON in the transcript view.
        # (Per-speaker color highlighting applies only to the diarized path.)
        response = (
            mock_deepgram_cls.return_value.listen.v1.media.transcribe_file.return_value
        )

        streamlit_app._display_transcript(response)

        (markdown_arg,), markdown_kwargs = mock_st.markdown.call_args
        assert markdown_arg == "Life moves pretty fast really."
        assert "unsafe_allow_html" not in markdown_kwargs
        mock_st.json.assert_not_called()

    def test_escapes_markdown_metacharacters(self, mock_st):
        response = MagicMock()
        response.results.channels[0].alternatives[0].transcript = "take *2* `mg` of x_y"

        streamlit_app._display_transcript(response)

        mock_st.markdown.assert_called_once_with("take \\*2\\* \\`mg\\` of x\\_y")

    def test_missing_results_renders_no_transcript_notice(self, mock_st):
        # A callback/async ListenV1AcceptedResponse has only request_id, no results.
        response = MagicMock(spec=["request_id", "model_dump_json"])
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
        mock_st.caption.assert_not_called()

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


class TestDisplayJson:
    def test_renders_raw_json(self, mock_deepgram_cls, mock_st):
        response = (
            mock_deepgram_cls.return_value.listen.v1.media.transcribe_file.return_value
        )

        streamlit_app._display_json(response)

        mock_st.json.assert_called_once_with(response.model_dump_json())

    def test_minimal_no_markdown_expander_or_downloads(
        self, mock_deepgram_cls, mock_st
    ):
        response = (
            mock_deepgram_cls.return_value.listen.v1.media.transcribe_file.return_value
        )

        streamlit_app._display_json(response)

        mock_st.markdown.assert_not_called()
        mock_st.expander.assert_not_called()
        mock_st.download_button.assert_not_called()

    def test_results_less_response_still_serialized(self, mock_st):
        # An accepted/callback response (no results) still serializes via model_dump_json.
        response = MagicMock(spec=["model_dump_json"])
        response.model_dump_json.return_value = '{"request_id": "req-123"}'

        streamlit_app._display_json(response)

        mock_st.json.assert_called_once_with('{"request_id": "req-123"}')


class TestOutputPanel:
    def test_shows_placeholder_when_empty(self, mock_st):
        render = MagicMock()

        streamlit_app._output_panel([], [], render)

        mock_st.caption.assert_called_once_with(streamlit_app.PLACEHOLDER)
        render.assert_not_called()

    def test_single_result_has_player_and_no_divider(self, mock_st):
        response = MagicMock()
        render = MagicMock()

        streamlit_app._output_panel([("a.mp3", response)], [b"a"], render)

        render.assert_called_once_with(response)
        mock_st.audio.assert_called_once()
        mock_st.divider.assert_not_called()
        mock_st.caption.assert_not_called()

    def test_multiple_results_labeled_with_dividers(self, mock_st):
        render = MagicMock()
        responses = [("a.mp3", MagicMock()), ("b.mp3", MagicMock())]

        streamlit_app._output_panel(responses, [b"a", b"b"], render)

        assert render.call_count == 2
        assert mock_st.audio.call_count == 2
        mock_st.divider.assert_called_once()
        labels = [c.args[0] for c in mock_st.markdown.call_args_list]
        assert any("a.mp3" in m for m in labels)
        assert any("b.mp3" in m for m in labels)

    def test_single_none_source_renders_no_player(self, mock_st):
        response = MagicMock()
        render = MagicMock()

        streamlit_app._output_panel([("big.wav", response)], [None], render)

        mock_st.audio.assert_not_called()
        mock_st.caption.assert_called_once_with(streamlit_app.PLAYBACK_TOO_LARGE)
        render.assert_called_once_with(response)

    def test_none_source_skipped_among_multiple(self, mock_st):
        render = MagicMock()
        responses = [("big.wav", MagicMock()), ("small.wav", MagicMock())]

        streamlit_app._output_panel(responses, [None, b"a"], render)

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
        assert values == {"Duration": "3.5 s", "Confidence": "98.0%"}

    def test_no_metrics_when_response_has_no_results(self, mock_st):
        # A results-less response (no metadata duration, no alternative confidence)
        # renders no metric cards rather than blank/garbage values.
        response = MagicMock(spec=["request_id"])

        streamlit_app._display_metrics(response)

        mock_st.metric.assert_not_called()

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
    """The Transcript-tab download button (absent for the JSON renderer by design)."""

    def test_no_button_when_no_responses(self, mock_st):
        streamlit_app._transcript_download([])

        mock_st.download_button.assert_not_called()

    def test_button_carries_transcript_text(self, mock_deepgram_cls, mock_st):
        response = (
            mock_deepgram_cls.return_value.listen.v1.media.transcribe_file.return_value
        )

        streamlit_app._transcript_download([("a.wav", response)])

        # Focus: the plain-text transcript button and its lazy (zero-arg callable)
        # data. SRT-button behavior is covered by the dedicated tests below.
        assert mock_st.download_button.call_count >= 1
        build_transcript = mock_st.download_button.call_args_list[0].args[1]
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

    def test_single_diarized_result_offers_srt_subtitles(self, mock_st):
        # A single timed, diarized result adds the SRT button; its cues carry the
        # 1-based "Speaker N:" prefix and a timing line.
        words = [
            mock_word("Hello.", 0.9, speaker=0, start=0.0, end=1.0),
            mock_word("Hi.", 0.9, speaker=1, start=1.0, end=2.0),
        ]
        response = MagicMock()
        response.results.channels = [MagicMock(alternatives=[MagicMock(words=words)])]

        streamlit_app._transcript_download([("visit.wav", response)])

        assert mock_st.download_button.call_count == 2
        srt_call = mock_st.download_button.call_args_list[1]
        assert "Speaker 1: Hello." in srt_call.args[1]
        assert "-->" in srt_call.args[1]
        assert srt_call.kwargs["file_name"] == "subtitles.srt"
        assert srt_call.kwargs["mime"] == "application/x-subrip"
        assert srt_call.kwargs["icon"] == ":material/subtitles:"

    def test_no_srt_button_when_no_timed_cues(self, mock_st):
        # A transcript with no word timings yields no SRT cues, so only the
        # plain-text button renders (still deferred behind a callable).
        alt = MagicMock(words=[])
        alt.transcript = "Patient is stable."
        response = MagicMock()
        response.results.channels = [MagicMock(alternatives=[alt])]

        streamlit_app._transcript_download([("note.wav", response)])

        assert mock_st.download_button.call_count == 1
        build_transcript = mock_st.download_button.call_args_list[0].args[1]
        assert "Patient is stable." in build_transcript()

    def test_multiple_results_export_txt_only(self, mock_st):
        # SRT maps to a single media track, so a multi-result batch skips it and
        # exports only the combined plain-text transcript (both files present).
        def _timed(token):
            words = [mock_word(token, 0.9, speaker=0, start=0.0, end=1.0)]
            response = MagicMock()
            response.results.channels = [
                MagicMock(alternatives=[MagicMock(words=words)])
            ]
            return response

        streamlit_app._transcript_download(
            [("a.wav", _timed("Alpha.")), ("b.wav", _timed("Beta."))]
        )

        assert mock_st.download_button.call_count == 1
        combined = mock_st.download_button.call_args_list[0].args[1]()
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
      structure, the dynamic-tab ``.open`` access) plus the idle UI: the placeholder
      caption, a disabled Run button, the inputs | output column split, and the
      Features control order
      (``language, keyterms, smart_format, diarize, dictation, measurements, redact``).
    - **seeded diarized** — renders the Transcript tab for a diarized result, asserting
      the exact 1-based color-directive speaker lines, the Duration/Confidence metric
      cards, the dropped-playback caption, and that the hidden JSON tab's
      ``model_dump_json`` is skipped (the ``download_button`` icon also runs here).
    - **seeded flat** — the non-diarized render branch: a plain escaped transcript
      with no speaker labels.
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


def _word(text, speaker):
    w = MagicMock()
    w.punctuated_word = text
    w.word = text
    w.speaker = speaker
    w.confidence = 0.9
    w.start = 0.0
    w.end = 1.0
    return w


def _resp(transcript, words, duration, confidence):
    alt = MagicMock()
    alt.transcript = transcript
    alt.confidence = confidence
    alt.words = words
    r = MagicMock()
    r.metadata.duration = duration
    r.results.channels = [MagicMock(alternatives=[alt])]
    r.model_dump_json.return_value = '{"results": "ok"}'
    return r


def _widget_keys_in_order(node, acc):
    # Depth-first walk of the element tree, collecting widget keys in document order.
    # Containers are skipped: since Streamlit 1.64 the st.form Block carries its form
    # id ("features") as .key, which is a container, not a Features control.
    for child in getattr(node, "children", {}).values():
        if not isinstance(child, Block) and getattr(child, "key", None):
            acc.append(child.key)
        _widget_keys_in_order(child, acc)


# 1) Empty state — module load (set_page_config / form / dynamic-tab .open) plus the
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

# Main area is a side-by-side split: audio inputs on the left, output panel (with its
# empty-state placeholder) on the right — the output is not stacked under the inputs.
inputs, output = at.main.columns
assert [t.label for t in inputs.tabs] == [
    ":material/upload: Upload",
    ":material/mic: Record",
    ":material/link: URL",
]
assert [t.label for t in output.tabs] == [
    ":material/description: Transcript",
    ":material/data_object: JSON",
]
assert any("Select audio, then click Run" in c.value for c in output.caption)

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
#    color-highlighted speaker lines; Duration + Confidence metric cards; and the
#    dropped-playback caption (audio_source is None).
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
assert [m.label for m in seeded.metric] == ["Duration", "Confidence"]
assert [m.value for m in seeded.metric] == ["3.5 s", "95.0%"]
assert any("Inline playback unavailable" in c.value for c in seeded.caption)
# The hidden JSON tab's serialization is skipped while Transcript shows: no st.json
# element is emitted and model_dump_json is never called (the .open guard fires).
assert not seeded.json, [j.value for j in seeded.json]
assert not diar.model_dump_json.called

# 3) Seeded flat (non-diarized) result — the other render branch: a plain escaped
#    transcript with no speaker labels.
flat = _resp("Patient is stable.", [_word("Patient", None)], 12.0, 0.88)
flat_at = AppTest.from_file(app, default_timeout=30)
flat_at.session_state["responses"] = [("note.wav", flat)]
flat_at.session_state["audio_sources"] = [None]
flat_at.run()
assert not flat_at.exception, flat_at.exception
assert [m.value for m in flat_at.markdown] == ["Patient is stable."]
assert not any("Speaker" in m.value for m in flat_at.markdown)

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
