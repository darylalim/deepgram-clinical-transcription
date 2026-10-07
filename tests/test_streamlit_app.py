import inspect
import json
import logging
import os
import re
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

import pytest

import streamlit_app
from nova.access import (
    PROBLEM_AUTHLIB,
    PROBLEM_NOT_CONFIGURED,
    PROBLEM_OPT_OUT_IN_DOTENV,
    PROBLEM_OPT_OUT_IN_SECRETS,
    PROBLEM_SECRETS_UNPARSEABLE,
    PROBLEM_TRUSTED_HEADERS,
    Decision,
)
from nova.audit import Actor, AuditSchemaError
from nova.config import ALLOW_ANONYMOUS_ENV, MAX_SESSION_AGE_SECONDS
from tests.helpers import RUN_ID, audit_lines, mock_upload, mock_word, wav_bytes

FAKE_AUDIO = b"fake-audio-data"
# Stand-ins for PHI that must never reach a log line: a filename carrying an MRN, the
# transcript's words, a keyterm, and a URL from an error message.
PHI_MARKERS = (
    "Jane_Doe_MRN12345",
    "MRN12345",
    "hydroxyzine",
    "Jane Doe",
    "rash on left forearm",
    "https://",
)


def _assert_no_phi(text: str) -> None:
    lowered = text.lower()
    for marker in PHI_MARKERS:
        assert marker.lower() not in lowered, marker


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

    @pytest.mark.parametrize(
        ("opts", "model"),
        [({}, "nova-3-medical"), ({"model": "nova-3-pharma"}, "nova-3-pharma")],
    )
    def test_records_the_runs_model(self, mock_deepgram_cls, mock_st, opts, model):
        # Kept per run, so the panel and export name the model the audio was sent to
        # even after the sidebar selection changes.
        streamlit_app._process_inputs("test-key", [("a.wav", b"a")], **opts)

        assert mock_st.session_state["run_model"] == model

    def test_forwards_features_to_sdk_call(self, mock_deepgram_cls, mock_st):
        # The wrapper forwards each feature kwarg by name through build_options to the SDK
        # call; a forwarding typo (e.g. swapping diarize/measurements) would slip past the
        # build_options/transcribe_batch unit tests but is caught here.
        streamlit_app._process_inputs(
            "test-key",
            [("test.wav", FAKE_AUDIO)],
            model="nova-3-pharma",
            keyterms=["metformin"],
            language="en-GB",
            diarize=True,
            # Opposite of diarize, so a swapped forward is visible whatever the defaults.
            measurements=False,
            redact=["pii"],
        )

        kwargs = mock_deepgram_cls.return_value.listen.v1.media.transcribe_file.call_args.kwargs
        assert kwargs["model"] == "nova-3-pharma"
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

    def test_writes_a_fresh_run_id_per_run(self, mock_deepgram_cls, mock_st):
        # The review widgets key on it, so each Run starts with fresh editors and
        # unchecked Reviewed boxes.
        streamlit_app._process_inputs("test-key", [("a.wav", b"a")])
        first = mock_st.session_state["run_id"]
        streamlit_app._process_inputs("test-key", [("a.wav", b"a")])
        second = mock_st.session_state["run_id"]

        assert re.fullmatch(r"[0-9a-f]{32}", first)
        assert re.fullmatch(r"[0-9a-f]{32}", second)
        assert first != second

    def test_fully_failed_run_still_rotates_run_id(self, mock_deepgram_cls, mock_st):
        mock_st.session_state["run_id"] = RUN_ID
        mock_client = mock_deepgram_cls.return_value
        mock_client.listen.v1.media.transcribe_file.side_effect = Exception("fail")

        streamlit_app._process_inputs("test-key", [("a.wav", b"a")])

        assert re.fullmatch(r"[0-9a-f]{32}", mock_st.session_state["run_id"])
        assert mock_st.session_state["run_id"] != RUN_ID

    def test_returns_the_success_count(self, mock_deepgram_cls, mock_st):
        mock_client = mock_deepgram_cls.return_value

        def fake_transcribe(request, **_):
            if request == b"bad":
                raise Exception("boom")
            return MagicMock()

        mock_client.listen.v1.media.transcribe_file.side_effect = fake_transcribe

        n_ok = streamlit_app._process_inputs(
            "test-key", [("ok.wav", b"ok"), ("bad.wav", b"bad")]
        )

        assert n_ok == 1

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


def _recording(data: bytes) -> MagicMock:
    rec = MagicMock()
    rec.getvalue.return_value = data
    return rec


class TestRunAudit:
    """`_run` logs one line per Run — counts and options only, never PHI."""

    def test_run_logs_counts_and_options_never_phi(
        self, mock_deepgram_cls, mock_st, capsys
    ):
        mock_st.session_state["keyterms"] = ["hydroxyzine"]
        good = MagicMock()
        good.results.channels[0].alternatives[
            0
        ].transcript = (
            "Patient Jane Doe reports rash on left forearm; start hydroxyzine"
        )

        def fake_transcribe(request, **_):
            if request == b"b":
                raise Exception(
                    "400 bad audio Jane_Doe_MRN12345.wav https://x/?mrn=MRN12345"
                )
            return good

        media = mock_deepgram_cls.return_value.listen.v1.media
        media.transcribe_file.side_effect = fake_transcribe

        streamlit_app._run(
            "key",
            [
                mock_upload("Jane_Doe_MRN12345.wav", b"a"),
                mock_upload("Jane_Doe_MRN12345_b.wav", b"b"),
            ],
            None,
        )

        out, err = capsys.readouterr()
        (line,) = audit_lines(out)
        assert {
            k: line[k]
            for k in (
                "event",
                "outcome",
                "user",
                "input_kind",
                "n_items",
                "n_ok",
                "n_failed",
                "n_skipped",
                "n_keyterms",
                "model",
                "language",
                "redact",
            )
        } == {
            "event": "transcription_run",
            "outcome": "partial",
            "user": "clinician@example.org",
            "input_kind": "upload",
            "n_items": 2,
            "n_ok": 1,
            "n_failed": 1,
            "n_skipped": 0,
            "n_keyterms": 1,
            "model": "nova-3-medical",
            "language": "en",
            "redact": [],
        }
        # The run id ties this line to the review and download lines that follow.
        assert line["run"] == mock_st.session_state["run_id"]
        _assert_no_phi(out + err)
        # The failure is still reported to the user — on screen only, never logged.
        assert "Jane_Doe_MRN12345_b.wav" in mock_st.error.call_args.args[0]

    def test_recording_run(self, mock_deepgram_cls, mock_st, capsys):
        streamlit_app._run("key", [], _recording(wav_bytes(1)))

        (line,) = audit_lines(capsys.readouterr().out)
        assert (line["event"], line["outcome"], line["input_kind"]) == (
            "transcription_run",
            "success",
            "record",
        )
        assert (line["n_items"], line["n_ok"], line["n_skipped"]) == (1, 1, 0)

    def test_oversize_uploads_are_counted_as_skipped(
        self, mock_deepgram_cls, mock_st, capsys
    ):
        big = mock_upload("big.wav", b"x", size=streamlit_app.MAX_FILE_SIZE + 1)
        streamlit_app._run("key", [big, mock_upload("ok.wav", b"ok")], None)

        (line,) = audit_lines(capsys.readouterr().out)
        assert (line["n_items"], line["n_ok"], line["n_skipped"]) == (1, 1, 1)

    @pytest.mark.parametrize(
        ("files", "recording", "kind", "reason", "n_items"),
        [
            (
                [mock_upload(f"f{i}.wav", b"x") for i in range(101)],
                None,
                "upload",
                "too_many_files",
                101,
            ),
            (
                [
                    mock_upload("Jane_Doe_MRN12345.wav", b"x", size=2**40),
                    mock_upload("b.wav", b"x", size=2**40),
                ],
                None,
                "upload",
                "all_oversize",
                2,
            ),
            ([], _recording(b"not-a-wav"), "record", "recording_unreadable", 1),
            ([], _recording(wav_bytes(30 * 60 + 1)), "record", "recording_too_long", 1),
        ],
        ids=["too-many-files", "all-oversize", "unreadable", "too-long"],
    )
    def test_rejections_log_one_line_with_their_reason(
        self,
        mock_deepgram_cls,
        mock_st,
        capsys,
        files,
        recording,
        kind,
        reason,
        n_items,
    ):
        streamlit_app._run("key", files, recording)

        out, err = capsys.readouterr()
        (line,) = audit_lines(out)
        assert {
            k: line[k] for k in ("event", "outcome", "input_kind", "reason", "n_items")
        } == {
            "event": "transcription_rejected",
            "outcome": "rejected",
            "input_kind": kind,
            "reason": reason,
            "n_items": n_items,
        }
        mock_deepgram_cls.assert_not_called()
        _assert_no_phi(out + err)

    def test_no_input_logs_nothing(self, mock_deepgram_cls, mock_st, capsys):
        streamlit_app._run("key", [], None)

        assert capsys.readouterr().out == ""

    def test_missing_actor_fails_before_any_audio_is_sent(
        self, mock_deepgram_cls, mock_st, capsys
    ):
        # Never a silent fallback to "anonymous": that would hide a broken gate.
        mock_st.session_state.pop("audit_actor")

        with pytest.raises(RuntimeError, match="No audit actor"):
            streamlit_app._run("key", [mock_upload("a.wav", b"a")], None)

        mock_deepgram_cls.assert_not_called()
        assert capsys.readouterr().out == ""

    @pytest.mark.parametrize("key", ["model", "language"])
    def test_unknown_option_fails_before_any_audio_is_sent(
        self, key, mock_deepgram_cls, mock_st, capsys
    ):
        mock_st.session_state[key] = "Jane_Doe_MRN12345"

        with pytest.raises(AuditSchemaError) as exc:
            streamlit_app._run("key", [mock_upload("a.wav", b"a")], None)

        mock_deepgram_cls.assert_not_called()
        _assert_no_phi(str(exc.value))
        assert capsys.readouterr().out == ""


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
    """The panel's layout; `_display_transcript` and `_review_controls` are patched
    out (tested above and below)."""

    @pytest.fixture
    def render(self):
        with patch.object(streamlit_app, "_display_transcript") as render:
            yield render

    @pytest.fixture
    def controls(self):
        with patch.object(
            streamlit_app,
            "_review_controls",
            side_effect=lambda *a: (f"text{a[1]}", True),
        ) as controls:
            yield controls

    def test_shows_placeholder_when_empty(self, mock_st, render, controls):
        reviews = streamlit_app._output_panel([], [], RUN_ID)

        mock_st.caption.assert_called_once_with(streamlit_app.PLACEHOLDER)
        render.assert_not_called()
        controls.assert_not_called()
        assert reviews == []

    def test_single_result_has_player_and_no_divider(self, mock_st, render, controls):
        response = MagicMock()

        reviews = streamlit_app._output_panel([("a.mp3", response)], [b"a"], RUN_ID)

        render.assert_called_once_with(response)
        controls.assert_called_once_with(RUN_ID, 0, response, 1)
        assert reviews == [("text0", True)]
        mock_st.audio.assert_called_once()
        mock_st.divider.assert_not_called()
        mock_st.caption.assert_not_called()

    def test_multiple_results_labeled_with_dividers(self, mock_st, render, controls):
        responses = [("a.mp3", MagicMock()), ("b.mp3", MagicMock())]

        reviews = streamlit_app._output_panel(responses, [b"a", b"b"], RUN_ID)

        assert render.call_count == 2
        assert mock_st.audio.call_count == 2
        mock_st.divider.assert_called_once()
        labels = [c.args[0] for c in mock_st.markdown.call_args_list]
        assert any("a.mp3" in m for m in labels)
        assert any("b.mp3" in m for m in labels)
        # One set of review controls per result, by position (never by filename),
        # and their values come back in result order for the download gate.
        assert [c.args for c in controls.call_args_list] == [
            (RUN_ID, 0, responses[0][1], 2),
            (RUN_ID, 1, responses[1][1], 2),
        ]
        assert reviews == [("text0", True), ("text1", True)]

    def test_single_none_source_renders_no_player(self, mock_st, render, controls):
        response = MagicMock()

        streamlit_app._output_panel([("big.wav", response)], [None], RUN_ID)

        mock_st.audio.assert_not_called()
        mock_st.caption.assert_called_once_with(streamlit_app.PLAYBACK_TOO_LARGE)
        render.assert_called_once_with(response)
        controls.assert_called_once_with(RUN_ID, 0, response, 1)

    def test_none_source_skipped_among_multiple(self, mock_st, render, controls):
        responses = [("big.wav", MagicMock()), ("small.wav", MagicMock())]

        streamlit_app._output_panel(responses, [None, b"a"], RUN_ID)

        mock_st.audio.assert_called_once_with(b"a", format="audio/wav")
        mock_st.caption.assert_called_once_with(streamlit_app.PLAYBACK_TOO_LARGE)
        assert render.call_count == 2
        assert controls.call_count == 2


class TestReviewControls:
    """The per-result export editor and Reviewed checkbox, and their callbacks."""

    def test_editor_is_seeded_with_the_plain_transcript(
        self, mock_deepgram_cls, mock_st
    ):
        response = (
            mock_deepgram_cls.return_value.listen.v1.media.transcribe_file.return_value
        )

        streamlit_app._review_controls(RUN_ID, 0, response, 1)

        mock_st.text_area.assert_called_once_with(
            streamlit_app.EDIT_LABEL,
            value=streamlit_app._plain_transcript(response),
            key=f"transcript_{RUN_ID}_0",
            height="content",
            help=streamlit_app.EDIT_HELP,
            on_change=streamlit_app._on_edit,
            args=(RUN_ID, 0),
            disabled=False,
        )

    def test_checkbox_is_keyed_by_run_and_position(self, mock_st):
        streamlit_app._review_controls(RUN_ID, 0, MagicMock(), 1)

        # No value= — the checkbox's state lives only under its key.
        mock_st.checkbox.assert_called_once_with(
            streamlit_app.REVIEWED_LABEL,
            key=f"reviewed_{RUN_ID}_0",
            on_change=streamlit_app._on_review,
            args=(RUN_ID, 0),
        )

    def test_editor_is_disabled_once_reviewed(self, mock_st):
        # Signed-off text is frozen: unchecking Reviewed is the only way to edit it.
        mock_st.session_state[f"reviewed_{RUN_ID}_0"] = True

        streamlit_app._review_controls(RUN_ID, 0, MagicMock(), 1)

        assert mock_st.text_area.call_args.kwargs["disabled"] is True

    def test_truthy_non_true_flag_does_not_count_as_reviewed(self, mock_st):
        mock_st.session_state[f"reviewed_{RUN_ID}_0"] = "yes"
        mock_st.checkbox.return_value = 1

        _, reviewed = streamlit_app._review_controls(RUN_ID, 0, MagicMock(), 1)

        assert mock_st.text_area.call_args.kwargs["disabled"] is False
        assert reviewed is False

    def test_returns_the_widget_values(self, mock_st):
        mock_st.text_area.return_value = "Edited text."
        mock_st.checkbox.return_value = True

        result = streamlit_app._review_controls(RUN_ID, 0, MagicMock(), 1)

        assert result == ("Edited text.", True)

    def test_multi_result_labels_carry_position_not_filename(self, mock_st):
        streamlit_app._review_controls(RUN_ID, 1, MagicMock(), 3)

        assert (
            mock_st.text_area.call_args.args[0]
            == f"{streamlit_app.EDIT_LABEL} (2 of 3)"
        )
        assert (
            mock_st.checkbox.call_args.args[0]
            == f"{streamlit_app.REVIEWED_LABEL} (2 of 3)"
        )
        assert mock_st.text_area.call_args.kwargs["key"] == f"transcript_{RUN_ID}_1"
        assert mock_st.checkbox.call_args.kwargs["key"] == f"reviewed_{RUN_ID}_1"

    def test_on_edit_clears_a_set_flag_and_says_so(self, mock_st):
        # The ordinary "type a correction, then click Reviewed" path in a browser:
        # the click is the blur that applies the edit, so both land in one rerun.
        mock_st.session_state[f"reviewed_{RUN_ID}_0"] = True

        streamlit_app._on_edit(RUN_ID, 0)

        assert mock_st.session_state[f"reviewed_{RUN_ID}_0"] is False
        mock_st.toast.assert_called_once_with(
            streamlit_app.REVIEW_CLEARED, icon=":material/edit_note:"
        )

    def test_on_edit_leaves_an_unset_flag_untouched(self, mock_st):
        mock_st.session_state[f"reviewed_{RUN_ID}_1"] = False

        streamlit_app._on_edit(RUN_ID, 0)
        streamlit_app._on_edit(RUN_ID, 1)

        assert f"reviewed_{RUN_ID}_0" not in mock_st.session_state
        assert mock_st.session_state[f"reviewed_{RUN_ID}_1"] is False
        mock_st.toast.assert_not_called()  # nothing was cleared, so nothing to say

    def test_edit_help_explains_export_apply_and_unlock(self):
        assert "Download saves" in streamlit_app.EDIT_HELP
        assert "Ctrl/⌘+Enter" in streamlit_app.EDIT_HELP
        assert "before checking Reviewed" in streamlit_app.EDIT_HELP
        assert "uncheck Reviewed to edit again" in streamlit_app.EDIT_HELP


REVIEWED = f"reviewed_{RUN_ID}_0"
TEXT = f"transcript_{RUN_ID}_0"


class TestReviewAudit:
    """The Reviewed checkbox's and editor's callbacks log each change in a result's
    review state exactly once, whatever order Streamlit runs them in."""

    @pytest.fixture
    def session(self, mock_deepgram_cls, mock_st):
        # One conftest result (two words below 0.90), its editor still at the seed.
        response = (
            mock_deepgram_cls.return_value.listen.v1.media.transcribe_file.return_value
        )
        mock_st.session_state.update(
            {
                "responses": [("Jane_Doe_MRN12345.wav", response)],
                "run_id": RUN_ID,
                TEXT: streamlit_app._plain_transcript(response),
            }
        )
        return mock_st.session_state

    @staticmethod
    def _events(capsys):
        return [
            (line["event"], line.get("edited"))
            for line in audit_lines(capsys.readouterr().out)
        ]

    def test_check_logs_a_sign_off(self, session, capsys):
        session[REVIEWED] = True

        streamlit_app._on_review(RUN_ID, 0)

        out, err = capsys.readouterr()
        (line,) = audit_lines(out)
        assert {k: line[k] for k in ("event", "user", "auth")} == {
            "event": "review_signed_off",
            "user": "clinician@example.org",
            "auth": "oidc",
        }
        assert (line["run"], line["result_index"], line["n_results"]) == (RUN_ID, 0, 1)
        assert line["edited"] is False
        assert line["n_flagged"] == 2  # the conftest response's 0.85 / 0.80 words
        assert "Jane_Doe" not in out + err
        assert "Life moves" not in out + err

    def test_uncheck_reopens_and_an_edited_sign_off_says_so(self, session, capsys):
        session[REVIEWED] = True
        streamlit_app._on_review(RUN_ID, 0)
        session[REVIEWED] = False
        streamlit_app._on_review(RUN_ID, 0)
        session[TEXT] = "Life moves pretty fast, really."
        session[REVIEWED] = True
        streamlit_app._on_review(RUN_ID, 0)

        assert self._events(capsys) == [
            ("review_signed_off", False),
            ("review_reopened", None),
            ("review_signed_off", True),
        ]

    def test_a_repeated_callback_logs_nothing_new(self, session, capsys):
        session[REVIEWED] = True

        streamlit_app._on_review(RUN_ID, 0)
        streamlit_app._on_review(RUN_ID, 0)

        assert self._events(capsys) == [("review_signed_off", False)]

    def test_an_edit_that_undoes_a_logged_check_logs_reopened(self, session, capsys):
        # An edit and a check in one rerun, checkbox callback first: the sign-off is
        # logged, then the edit clears the check, which is logged too.
        session[REVIEWED] = True
        streamlit_app._on_review(RUN_ID, 0)

        streamlit_app._on_edit(RUN_ID, 0)

        assert session[REVIEWED] is False
        assert self._events(capsys) == [
            ("review_signed_off", False),
            ("review_reopened", None),
        ]

    def test_an_edit_first_in_the_same_rerun_logs_nothing(self, session, capsys):
        # Editor callback first (what Streamlit 1.64 does): the check arrived, the
        # edit clears it before its own callback runs, so the state never changed —
        # no sign-off, and no "reopened" for a sign-off that was never logged.
        session[REVIEWED] = True

        streamlit_app._on_edit(RUN_ID, 0)
        streamlit_app._on_review(RUN_ID, 0)

        assert session[REVIEWED] is False
        assert capsys.readouterr().out == ""

    def test_an_edit_with_no_check_logs_nothing(self, session, capsys):
        streamlit_app._on_edit(RUN_ID, 0)

        assert capsys.readouterr().out == ""

    def test_a_new_run_starts_a_fresh_record(self, session, capsys):
        session[REVIEWED] = True
        streamlit_app._on_review(RUN_ID, 0)
        other = "1" * 32
        session.update({"run_id": other, f"reviewed_{other}_0": True})

        streamlit_app._on_review(other, 0)

        lines = audit_lines(capsys.readouterr().out)
        assert [(line["event"], line["run"]) for line in lines] == [
            ("review_signed_off", RUN_ID),
            ("review_signed_off", other),
        ]

    def test_a_result_no_longer_in_session_logs_nothing(self, session, capsys):
        session[f"reviewed_{RUN_ID}_3"] = True

        streamlit_app._on_review(RUN_ID, 3)

        assert capsys.readouterr().out == ""


class TestFeatureOpts:
    def test_defaults_when_session_empty(self, mock_st):
        assert streamlit_app._feature_opts() == {
            "model": "nova-3-medical",
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
                "model": "nova-3-pharma",
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
            "model": "nova-3-pharma",
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
            "model": "nova-3-medical",
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
    """The review-gated plain-text download button above the output panel.

    `_transcript_download` takes each result's `(editor text, reviewed)` pair as the
    review widgets returned them this run (see TestReviewControls)."""

    @staticmethod
    def _call_kwargs(mock_st):
        mock_st.download_button.assert_called_once()
        return mock_st.download_button.call_args

    def test_no_button_when_no_responses(self, mock_st):
        streamlit_app._transcript_download([], [], RUN_ID)

        mock_st.download_button.assert_not_called()

    def test_locked_until_reviewed_and_carries_no_transcript(
        self, mock_deepgram_cls, mock_st
    ):
        response = (
            mock_deepgram_cls.return_value.listen.v1.media.transcribe_file.return_value
        )
        text = streamlit_app._plain_transcript(response)

        streamlit_app._transcript_download(
            [("a.wav", response)], [(text, False)], RUN_ID
        )

        call = self._call_kwargs(mock_st)
        assert call.args == ("Download locked — 0/1 reviewed", "")
        assert call.kwargs["disabled"] is True
        assert call.kwargs["help"] == streamlit_app.DOWNLOAD_LOCKED_HELP
        assert call.kwargs["key"] == "download_transcripts"
        # The server-side half of the gate: no transcript text (and no callable that
        # could produce one) is registered while any result is unreviewed.
        assert text not in repr(call)
        assert not any(callable(a) for a in call.args)

    def test_one_unreviewed_result_of_two_keeps_it_locked(self, mock_st):
        responses = [("a.wav", MagicMock()), ("b.wav", MagicMock())]

        streamlit_app._transcript_download(
            responses, [("Alpha.", True), ("Beta.", False)], RUN_ID
        )

        call = self._call_kwargs(mock_st)
        assert call.args == ("Download locked — 1/2 reviewed", "")
        assert call.kwargs["disabled"] is True
        assert "Alpha." not in repr(call)

    def test_missing_review_state_keeps_it_locked(self, mock_st):
        # Defensive: fewer review pairs than results never unlocks.
        responses = [("a.wav", MagicMock()), ("b.wav", MagicMock())]

        streamlit_app._transcript_download(responses, [("Alpha.", True)], RUN_ID)

        call = self._call_kwargs(mock_st)
        assert call.args[1] == ""
        assert call.kwargs["disabled"] is True

    def test_unlocked_exports_the_edited_text(self, mock_deepgram_cls, mock_st):
        response = (
            mock_deepgram_cls.return_value.listen.v1.media.transcribe_file.return_value
        )

        streamlit_app._transcript_download(
            [("a.wav", response)], [("Life moves pretty fast, really.", True)], RUN_ID
        )

        call = self._call_kwargs(mock_st)
        label, build = call.args
        assert label == streamlit_app.DOWNLOAD_LABEL
        assert call.kwargs == {
            "file_name": "transcripts.txt",
            "mime": "text/plain",
            "icon": ":material/download:",
            "key": "download_transcripts",
            "disabled": False,
            "help": None,
            "on_click": "ignore",
        }
        assert callable(build)  # deferred: built on click, not per rerun
        # The editor's text — not Deepgram's original — is what gets saved.
        assert build() == "a.wav\nLife moves pretty fast, really."

    def test_unedited_result_exports_the_plain_transcript(
        self, mock_deepgram_cls, mock_st
    ):
        # End to end through the review controls: an untouched editor returns its
        # seed, so the export is the plain transcript.
        response = (
            mock_deepgram_cls.return_value.listen.v1.media.transcribe_file.return_value
        )
        mock_st.text_area.side_effect = lambda label, value, **_: value
        mock_st.checkbox.return_value = True

        review = streamlit_app._review_controls(RUN_ID, 0, response, 1)
        streamlit_app._transcript_download([("a.wav", response)], [review], RUN_ID)

        build = mock_st.download_button.call_args.args[1]
        assert build() == "a.wav\nLife moves pretty fast really."

    def test_multiple_results_combined_into_one_file(self, mock_st):
        responses = [("a.wav", MagicMock()), ("b.wav", MagicMock())]

        streamlit_app._transcript_download(
            responses, [("Alpha, edited.", True), ("Beta.", True)], RUN_ID
        )

        build = mock_st.download_button.call_args.args[1]
        assert build() == "a.wav\nAlpha, edited.\n\nb.wav\nBeta."

    def test_export_names_the_runs_model_as_captured_at_render(self, mock_st):
        mock_st.session_state["run_model"] = "nova-3-pharma"
        streamlit_app._transcript_download(
            [("a.wav", MagicMock())], [("Text.", True)], RUN_ID
        )
        build = mock_st.download_button.call_args.args[1]

        mock_st.session_state["run_model"] = "nova-3-medical"
        with patch.object(streamlit_app, "st", Mock(spec=[])):
            assert build() == "Model: Deepgram Nova-3 Pharma\n\na.wav\nText."

    @pytest.mark.parametrize("run_model", [None, "Jane_Doe_MRN12345"])
    def test_export_omits_an_unknown_model(self, mock_st, run_model):
        mock_st.session_state["run_model"] = run_model
        streamlit_app._transcript_download(
            [("a.wav", MagicMock())], [("Text.", True)], RUN_ID
        )

        build = mock_st.download_button.call_args.args[1]
        assert build() == "a.wav\nText."

    def test_deferred_export_uses_render_time_text_and_no_st(self, mock_st):
        # Streamlit runs the callable on click, on a worker thread with no
        # ScriptRunContext: it must return the text captured at render and never
        # touch `st.*` (a spec=[] Mock raises on any attribute access).
        reviews = [("Reviewed text.", True)]
        mock_st.session_state[f"transcript_{RUN_ID}_0"] = "Reviewed text."
        streamlit_app._transcript_download([("a.wav", MagicMock())], reviews, RUN_ID)
        build = mock_st.download_button.call_args.args[1]

        reviews[0] = ("Changed after render.", True)
        mock_st.session_state[f"transcript_{RUN_ID}_0"] = "Changed after render."
        with patch.object(streamlit_app, "st", Mock(spec=[])):
            assert build() == "a.wav\nReviewed text."

    def test_diarized_export_uses_plain_speaker_lines(self):
        # The editor's seed is plain text (no color directives): "Speaker N: ..."
        # per turn.
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


class TestDownloadAudit:
    """Each generated download file logs `transcript_downloaded` — from inside the
    deferred callable, so the line is 1:1 with files actually built."""

    @pytest.fixture
    def build(self, mock_deepgram_cls, mock_st):
        # Two reviewed results: the first edited in the editor, the second not.
        response = (
            mock_deepgram_cls.return_value.listen.v1.media.transcribe_file.return_value
        )
        plain = streamlit_app._plain_transcript(response)
        streamlit_app._transcript_download(
            [("Jane_Doe_MRN12345.wav", response), ("b.wav", response)],
            [("Patient Jane Doe reports rash on left forearm.", True), (plain, True)],
            RUN_ID,
        )
        return mock_st.download_button.call_args.args[1]

    def test_logs_each_file_without_its_content(self, build, capsys):
        assert capsys.readouterr().out == ""  # built at render, logged on click

        blob = build()

        assert "Jane_Doe_MRN12345.wav" in blob  # the file names its sources...
        out, err = capsys.readouterr()
        (line,) = audit_lines(out)
        assert {k: line[k] for k in ("event", "outcome", "user", "run")} == {
            "event": "transcript_downloaded",
            "outcome": "success",
            "user": "clinician@example.org",
            "run": RUN_ID,
        }
        assert (line["n_results"], line["n_edited"]) == (2, 1)
        _assert_no_phi(out + err)  # ...the audit line never does

    def test_every_click_is_logged(self, build, capsys):
        build()
        build()

        assert [line["event"] for line in audit_lines(capsys.readouterr().out)] == [
            "transcript_downloaded",
            "transcript_downloaded",
        ]

    def test_locked_download_logs_nothing(self, mock_st, capsys):
        streamlit_app._transcript_download(
            [("a.wav", MagicMock())], [("Text.", False)], RUN_ID
        )

        assert mock_st.download_button.call_args.args[1] == ""
        assert capsys.readouterr().out == ""

    def test_fails_closed_when_the_line_cannot_be_written(self, build, capsys):
        # No audit line, no file — and the error Streamlit would log is PHI-free.
        with (
            patch.object(
                streamlit_app.audit, "emit", side_effect=AuditSchemaError("run")
            ),
            pytest.raises(AuditSchemaError) as exc,
        ):
            build()

        _assert_no_phi(str(exc.value))
        assert capsys.readouterr().out == ""

    def test_missing_actor_keeps_download_from_rendering(self, mock_st):
        mock_st.session_state.pop("audit_actor")

        with pytest.raises(RuntimeError, match="No audit actor"):
            streamlit_app._transcript_download(
                [("a.wav", MagicMock())], [("Text.", True)], RUN_ID
            )

        mock_st.download_button.assert_not_called()


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


AUTH = {
    "redirect_uri": "http://localhost:8501/oauth2callback",
    "cookie_secret": "test-cookie-secret-0123456789abcdef",
    "client_id": "id",
    "client_secret": "secret",
    "server_metadata_url": "https://idp.example/.well-known/openid-configuration",
}
ACCESS = {"allowed_email_domains": ["hospital.org"]}


def _mock_secrets(mock_st, data):
    """Back the mocked `st.secrets` with `data`: `.get` and iteration, as the gate
    reads it (iteration is how the opt-out's presence is found)."""
    mock_st.secrets.get.side_effect = lambda key, default=None: data.get(key, default)
    mock_st.secrets.__iter__.side_effect = lambda: iter(data)


def _signed_in(email="dr@hospital.org", **overrides):
    """Claims as Streamlit stores them after a default-provider sign-in."""
    return {
        "is_logged_in": True,
        "email": email,
        "email_verified": True,
        "sub": "abc",
        "provider": "default",
        "iat": time.time() - 60,
        **overrides,
    }


class TestAccessGate:
    """`_access_gate` renders nova.access's decision (the policy itself is tested in
    test_access.py). Its Streamlit inputs — secrets, claims, config — are mocked, and
    the `.env` / Authlib probes are patched, so no developer file can steer it."""

    @pytest.fixture
    def secrets(self, mock_st, monkeypatch):
        # A developer's shell may set the opt-out (.env never reaches os.environ with it).
        monkeypatch.delenv(ALLOW_ANONYMOUS_ENV, raising=False)
        # Required: a MagicMock option value is truthy, so it would read as trusted
        # user headers being configured.
        mock_st.get_option.return_value = {}
        mock_st.user.to_dict.return_value = {}
        data: dict = {}
        _mock_secrets(mock_st, data)
        with (
            patch.object(streamlit_app, "_authlib_installed", return_value=True),
            patch.object(streamlit_app, "_dotenv_has_opt_out", return_value=False),
        ):
            yield data

    @pytest.fixture
    def configured(self, secrets):
        secrets.update({"auth": dict(AUTH), "access": dict(ACCESS)})
        return secrets

    @staticmethod
    def _buttons(mock_st):
        return [(c.args[0], c.kwargs) for c in mock_st.button.call_args_list]

    def test_not_configured_shows_only_a_generic_error(self, mock_st, secrets, caplog):
        with caplog.at_level(logging.ERROR, logger="nova.access"):
            decision = streamlit_app._access_gate()

        assert decision is None
        mock_st.error.assert_called_once_with(
            streamlit_app.SIGN_IN_UNAVAILABLE, icon=":material/error:"
        )
        mock_st.button.assert_not_called()
        assert mock_st.session_state["access_ok"] is False
        # The visitor sees no configuration detail; the operator gets it on stderr.
        assert PROBLEM_NOT_CONFIGURED not in repr(mock_st.mock_calls)
        assert [r.getMessage() for r in caplog.records] == [
            f"Sign-in is unavailable: {PROBLEM_NOT_CONFIGURED}"
        ]

    def test_problem_is_logged_once_per_session(self, mock_st, secrets, caplog):
        with caplog.at_level(logging.ERROR, logger="nova.access"):
            streamlit_app._access_gate()
            streamlit_app._access_gate()

        assert len(caplog.records) == 1
        assert mock_st.error.call_count == 2  # but the visitor sees it every run

    def test_anonymous_opt_out_shows_the_banner(self, mock_st, secrets, monkeypatch):
        monkeypatch.setenv(ALLOW_ANONYMOUS_ENV, "1")

        decision = streamlit_app._access_gate()

        assert decision == Decision("anonymous")
        mock_st.warning.assert_called_once_with(
            streamlit_app.ANONYMOUS_MODE, icon=":material/no_accounts:"
        )
        mock_st.error.assert_not_called()
        assert mock_st.session_state["access_ok"] is True
        assert mock_st.session_state["access_expires_at"] is None  # never expires
        assert ALLOW_ANONYMOUS_ENV in streamlit_app.ANONYMOUS_MODE

    def test_no_secrets_file_is_treated_as_missing(self, mock_st, secrets, monkeypatch):
        monkeypatch.setenv(ALLOW_ANONYMOUS_ENV, "1")
        mock_st.secrets.get.side_effect = streamlit_app.StreamlitSecretNotFoundError(
            "none", error_id="no-secrets-found"
        )

        assert streamlit_app._access_gate() == Decision("anonymous")

    def test_unreadable_secrets_block_even_with_the_opt_out(
        self, mock_st, secrets, monkeypatch, caplog
    ):
        monkeypatch.setenv(ALLOW_ANONYMOUS_ENV, "1")
        mock_st.secrets.get.side_effect = streamlit_app.StreamlitSecretNotFoundError(
            "bad", error_id="failed-parsing-secrets-file"
        )

        with caplog.at_level(logging.ERROR, logger="nova.access"):
            decision = streamlit_app._access_gate()

        assert decision is None
        mock_st.error.assert_called_once()
        mock_st.warning.assert_not_called()
        assert PROBLEM_SECRETS_UNPARSEABLE in caplog.text

    def test_opt_out_in_dotenv_blocks_even_with_sign_in_configured(
        self, mock_st, configured, monkeypatch, caplog
    ):
        monkeypatch.setenv(ALLOW_ANONYMOUS_ENV, "1")
        mock_st.user.to_dict.return_value = _signed_in()

        with (
            patch.object(streamlit_app, "_dotenv_has_opt_out", return_value=True),
            caplog.at_level(logging.ERROR, logger="nova.access"),
        ):
            decision = streamlit_app._access_gate()

        assert decision is None
        assert PROBLEM_OPT_OUT_IN_DOTENV in caplog.text
        assert mock_st.session_state["access_ok"] is False

    def test_missing_authlib_blocks(self, mock_st, configured, caplog):
        with (
            patch.object(streamlit_app, "_authlib_installed", return_value=False),
            caplog.at_level(logging.ERROR, logger="nova.access"),
        ):
            decision = streamlit_app._access_gate()

        assert decision is None
        mock_st.error.assert_called_once_with(
            streamlit_app.SIGN_IN_UNAVAILABLE, icon=":material/error:"
        )
        assert PROBLEM_AUTHLIB in caplog.text

    def test_trusted_user_headers_block(self, mock_st, configured, caplog):
        mock_st.get_option.return_value = {"X-Forwarded-Email": "email"}

        with caplog.at_level(logging.ERROR, logger="nova.access"):
            assert streamlit_app._access_gate() is None

        mock_st.get_option.assert_called_with("server.trustedUserHeaders")
        assert PROBLEM_TRUSTED_HEADERS in caplog.text

    @pytest.mark.parametrize(
        "claims",
        [
            {},
            {"is_logged_in": False},  # production's logged-out shape with [auth] set
            {"email": "test@example.com"},  # AppTest's injected email alone
        ],
    )
    def test_logged_out_offers_sign_in(self, mock_st, configured, claims):
        mock_st.user.to_dict.return_value = claims

        decision = streamlit_app._access_gate()

        assert decision is None
        mock_st.info.assert_called_once_with(
            streamlit_app.SIGN_IN_PROMPT, icon=":material/lock:"
        )
        ((label, kwargs),) = self._buttons(mock_st)
        assert label == "Sign in"
        # st.login runs only as the click callback, never on render.
        assert kwargs["on_click"] is mock_st.login
        assert kwargs["args"] == (None,)
        assert kwargs["key"] == "sign_in_default"
        assert kwargs["icon"] == ":material/login:"
        assert "type" not in kwargs  # secondary: Run stays the only primary button
        mock_st.login.assert_not_called()
        assert mock_st.session_state["access_ok"] is False

    def test_one_button_per_named_provider(self, mock_st, configured):
        auth = {k: v for k, v in AUTH.items() if k in ("redirect_uri", "cookie_secret")}
        provider = {
            k: AUTH[k] for k in ("client_id", "client_secret", "server_metadata_url")
        }
        configured["auth"] = {**auth, "google": provider, "microsoft": provider}

        streamlit_app._access_gate()

        buttons = self._buttons(mock_st)
        assert [label for label, _ in buttons] == [
            "Sign in with Google",
            "Sign in with Microsoft",
        ]
        assert [kw["args"] for _, kw in buttons] == [("google",), ("microsoft",)]
        assert [kw["key"] for _, kw in buttons] == [
            "sign_in_google",
            "sign_in_microsoft",
        ]

    def test_stale_sign_in_asks_to_sign_in_again(self, mock_st, configured):
        mock_st.user.to_dict.return_value = _signed_in(iat=time.time() - 13 * 3600)

        assert streamlit_app._access_gate() is None

        mock_st.info.assert_called_once_with(
            streamlit_app.SESSION_EXPIRED, icon=":material/lock:"
        )
        assert [label for label, _ in self._buttons(mock_st)] == ["Sign in"]

    def test_allowed_continues(self, mock_st, configured):
        claims = _signed_in("Dr@Hospital.org")
        mock_st.user.to_dict.return_value = claims

        decision = streamlit_app._access_gate()

        assert decision is not None
        assert (decision.kind, decision.email) == ("allow", "dr@hospital.org")
        mock_st.error.assert_not_called()
        mock_st.info.assert_not_called()
        mock_st.warning.assert_not_called()
        assert mock_st.session_state["access_ok"] is True
        # Recorded for the checks that run without the gate (see TestSignInExpiry).
        assert (
            mock_st.session_state["access_expires_at"]
            == claims["iat"] + MAX_SESSION_AGE_SECONDS
        )

    def test_opt_out_in_secrets_blocks(self, mock_st, secrets, caplog):
        secrets[ALLOW_ANONYMOUS_ENV.lower()] = 1  # any value, any case

        with caplog.at_level(logging.ERROR, logger="nova.access"):
            assert streamlit_app._access_gate() is None

        assert PROBLEM_OPT_OUT_IN_SECRETS in caplog.text

    def test_denied_names_the_escaped_email_and_offers_sign_out(
        self, mock_st, configured
    ):
        mock_st.user.to_dict.return_value = _signed_in("dr_x@evil-hospital.org")

        assert streamlit_app._access_gate() is None

        mock_st.error.assert_called_once_with(
            "dr\\_x@evil-hospital.org is not authorized to use this app."
            + streamlit_app.DENY_SUFFIX,
            icon=":material/error:",
        )
        ((label, kwargs),) = self._buttons(mock_st)
        assert label == "Sign out"
        assert kwargs["on_click"] is streamlit_app._sign_out
        assert mock_st.session_state["access_ok"] is False

    def test_unverified_email_message(self, mock_st, configured):
        mock_st.user.to_dict.return_value = _signed_in(email_verified=False)

        streamlit_app._access_gate()

        assert "isn't a verified email address" in mock_st.error.call_args.args[0]

    def test_every_deny_reason_has_a_message(self):
        for reason in ("email_missing", "email_unverified", "domain_not_allowed"):
            assert streamlit_app.DENY_MESSAGES[reason]

    # The gate's audit record: the session's actor, rewritten every full run, and one
    # line per kind of outcome — not one per rerun.

    def test_allowed_session_logs_one_session_start(self, mock_st, configured, capsys):
        mock_st.user.to_dict.return_value = _signed_in("Dr@Hospital.org")

        streamlit_app._access_gate()
        streamlit_app._access_gate()  # a rerun

        (line,) = audit_lines(capsys.readouterr().out)
        session = mock_st.session_state["audit_session"]
        assert re.fullmatch(r"[0-9a-f]{32}", session)
        assert {k: line[k] for k in ("event", "user", "auth", "session")} == {
            "event": "session_start",
            "user": "dr@hospital.org",
            "auth": "oidc",
            "session": session,
        }
        assert mock_st.session_state["audit_actor"] == Actor(
            "dr@hospital.org", "oidc", session
        )

    def test_anonymous_session_is_logged_as_anonymous(
        self, mock_st, secrets, monkeypatch, capsys
    ):
        monkeypatch.setenv(ALLOW_ANONYMOUS_ENV, "1")

        streamlit_app._access_gate()

        (line,) = audit_lines(capsys.readouterr().out)
        assert (line["event"], line["user"], line["auth"]) == (
            "session_start",
            "anonymous",
            "anonymous",
        )

    def test_blocked_deploy_logs_one_access_denied(self, mock_st, secrets, capsys):
        streamlit_app._access_gate()
        streamlit_app._access_gate()

        (line,) = audit_lines(capsys.readouterr().out)
        assert {k: line[k] for k in ("event", "outcome", "user", "auth", "reason")} == {
            "event": "access_denied",
            "outcome": "denied",
            "user": None,
            "auth": "none",
            "reason": "auth_not_configured",
        }

    def test_denied_account_is_logged_with_its_reason(
        self, mock_st, configured, capsys
    ):
        mock_st.user.to_dict.return_value = _signed_in("dr_x@evil-hospital.org")

        streamlit_app._access_gate()

        (line,) = audit_lines(capsys.readouterr().out)
        assert (line["event"], line["reason"], line["user"], line["auth"]) == (
            "access_denied",
            "domain_not_allowed",
            "dr_x@evil-hospital.org",
            "oidc",
        )

    def test_sign_in_screen_logs_nothing(self, mock_st, configured, capsys):
        streamlit_app._access_gate()

        assert capsys.readouterr().out == ""
        assert mock_st.session_state["audit_actor"].auth == "none"

    def test_session_id_is_kept_across_runs(self, mock_st, configured):
        streamlit_app._access_gate()
        session = mock_st.session_state["audit_session"]
        mock_st.user.to_dict.return_value = _signed_in()

        streamlit_app._access_gate()

        assert mock_st.session_state["audit_actor"].session == session

    def test_stays_closed_if_its_audit_line_cannot_be_written(
        self, mock_st, configured
    ):
        mock_st.user.to_dict.return_value = _signed_in()

        with (
            patch.object(streamlit_app.audit, "emit", side_effect=OSError("down")),
            pytest.raises(OSError),
        ):
            streamlit_app._access_gate()

        assert mock_st.session_state["access_ok"] is False


class TestAccessInputs:
    """The gate's probes of secrets, `.env`, and Authlib."""

    def test_snapshot_reads_auth_access_and_the_opt_out(self, mock_st):
        _mock_secrets(
            mock_st, {"auth": AUTH, "access": ACCESS, ALLOW_ANONYMOUS_ENV: "1"}
        )

        snapshot = streamlit_app._secrets_snapshot()

        assert snapshot.state == "ok"
        assert (snapshot.auth, snapshot.access) == (AUTH, ACCESS)
        assert snapshot.anonymous_opt_out_present is True

    @pytest.mark.parametrize(
        "key",
        [ALLOW_ANONYMOUS_ENV, ALLOW_ANONYMOUS_ENV.lower(), "Nova_Allow_Anonymous"],
    )
    @pytest.mark.parametrize("value", [1, "0", "true", ""])
    def test_opt_out_in_secrets_counts_under_any_value_and_case(
        self, mock_st, key, value
    ):
        # Presence, never the value: Streamlit copies a top-level secret into
        # os.environ as str(value), so a TOML `= 1` arrives as "1" — and on Windows
        # os.environ ignores case, so `nova_allow_anonymous` arrives as the real
        # variable.
        _mock_secrets(mock_st, {key: value})

        assert streamlit_app._secrets_snapshot().anonymous_opt_out_present is True

    def test_other_secrets_are_not_the_opt_out(self, mock_st):
        _mock_secrets(
            mock_st,
            {"auth": AUTH, "DEEPGRAM_API_KEY": "k", f"{ALLOW_ANONYMOUS_ENV}_X": "1"},
        )

        assert streamlit_app._secrets_snapshot().anonymous_opt_out_present is False

    @pytest.mark.parametrize(
        "line", ["NOVA_ALLOW_ANONYMOUS = 1", 'nova_allow_anonymous = "1"']
    )
    def test_real_secrets_file_opt_out_is_found(self, tmp_path, line):
        # Through Streamlit's own parser: a TOML integer, and a lowercase key.
        from streamlit import config
        from streamlit.runtime.secrets import Secrets

        path = tmp_path / "secrets.toml"
        path.write_text(line + "\n")
        saved = config.get_option("secrets.files")
        config.set_option("secrets.files", [str(path)])
        try:
            # patch.dict restores os.environ, which a parse mirrors top-level values into.
            with patch.dict(os.environ):
                loaded = Secrets()
                loaded._file_watchers_installed = True  # no watcher on a temp file
                with patch.object(streamlit_app.st, "secrets", loaded):
                    snapshot = streamlit_app._secrets_snapshot()
        finally:
            config.set_option("secrets.files", saved)

        assert (snapshot.state, snapshot.anonymous_opt_out_present) == ("ok", True)

    @pytest.mark.parametrize(
        ("error_id", "state"),
        [
            ("no-secrets-found", "missing"),
            ("failed-parsing-secrets-file", "malformed"),
            ("invalid-secrets-path", "malformed"),
            (None, "malformed"),
        ],
    )
    def test_snapshot_classifies_read_failures(self, mock_st, error_id, state):
        mock_st.secrets.get.side_effect = streamlit_app.StreamlitSecretNotFoundError(
            "x", error_id=error_id
        )

        assert streamlit_app._secrets_snapshot().state == state

    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            ({}, False),
            ({"DEEPGRAM_API_KEY": "k"}, False),
            ({ALLOW_ANONYMOUS_ENV: "1"}, True),
            ({ALLOW_ANONYMOUS_ENV: "0"}, True),  # any mention in .env blocks
            ({ALLOW_ANONYMOUS_ENV: None}, True),
            (
                {ALLOW_ANONYMOUS_ENV.lower(): "1"},
                True,
            ),  # Windows' os.environ ignores case
            ({f"{ALLOW_ANONYMOUS_ENV}_X": "1"}, False),
        ],
    )
    def test_dotenv_opt_out_probe(self, values, expected):
        with patch.object(streamlit_app, "dotenv_values", return_value=values):
            assert streamlit_app._dotenv_has_opt_out() is expected

    def test_authlib_probe(self):
        assert streamlit_app._authlib_installed() is True  # the streamlit[auth] extra
        with patch.object(streamlit_app.importlib.util, "find_spec", return_value=None):
            assert streamlit_app._authlib_installed() is False


class TestDotenvLoad:
    """`.env` reaches `os.environ` only through `_load_dotenv`, which never copies
    the anonymous opt-out. `os.environ` outlives a script run, so an opt-out ever
    copied there would outlast its deletion from `.env` until a restart."""

    @pytest.fixture
    def dotenv(self, monkeypatch):
        # A stand-in for the .env file, re-read on every call like the real one.
        values: dict[str, str | None] = {}
        monkeypatch.delenv(ALLOW_ANONYMOUS_ENV, raising=False)
        with (
            patch.object(
                streamlit_app, "dotenv_values", side_effect=lambda: dict(values)
            ),
            patch.dict(os.environ),  # restores whatever the load sets
        ):
            yield values

    def test_copies_new_values_and_never_overrides(self, dotenv):
        os.environ["NOVA_TEST_SET"] = "process"
        dotenv.update(
            {"NOVA_TEST_NEW": "a", "NOVA_TEST_SET": "dotenv", "NOVA_TEST_BARE": None}
        )

        streamlit_app._load_dotenv()

        assert os.environ["NOVA_TEST_NEW"] == "a"
        assert os.environ["NOVA_TEST_SET"] == "process"  # like load_dotenv()
        assert "NOVA_TEST_BARE" not in os.environ  # a bare `KEY` line has no value

    @pytest.mark.parametrize(
        "key",
        [ALLOW_ANONYMOUS_ENV, ALLOW_ANONYMOUS_ENV.lower(), "Nova_Allow_Anonymous"],
    )
    def test_never_copies_the_opt_out(self, dotenv, key):
        dotenv[key] = "1"

        streamlit_app._load_dotenv()

        assert not [k for k in os.environ if k.upper() == ALLOW_ANONYMOUS_ENV]

    def test_opt_out_deleted_from_dotenv_without_a_restart_stays_blocked(
        self, dotenv, mock_st
    ):
        # Two script runs in one process, as Streamlit reruns are, with the line
        # deleted from .env in between — what the blocked app's operator message
        # says to do. With no [auth], the second run must not open anonymously.
        mock_st.get_option.return_value = {}
        mock_st.user.to_dict.return_value = {}
        _mock_secrets(mock_st, {})
        dotenv.update({"NOVA_TEST_NEW": "a", ALLOW_ANONYMOUS_ENV: "1"})
        problems = []
        with patch.object(streamlit_app, "_authlib_installed", return_value=True):
            for _ in range(2):
                streamlit_app._load_dotenv()  # what each run does at module level
                assert streamlit_app._access_gate() is None
                assert mock_st.session_state["access_ok"] is False
                problems.append(mock_st.session_state["access_problem_logged"])
                dotenv.pop(ALLOW_ANONYMOUS_ENV, None)

        assert problems == [PROBLEM_OPT_OUT_IN_DOTENV, PROBLEM_NOT_CONFIGURED]
        mock_st.warning.assert_not_called()  # never the anonymous banner


class TestSignOut:
    """The sidebar account block, Sign out, and the PHI purge behind it."""

    def test_account_panel_shows_the_email_as_plain_text(self, mock_st):
        streamlit_app._account_panel("dr_smith@hospital.org")

        # Not inside the caption: Markdown would autolink it, and the caption's 60%
        # opacity dims that link below 4.5:1. st.text renders it verbatim (so no
        # escaping), unlinked, at full opacity.
        assert mock_st.caption.call_args_list[0].args == (
            ":material/account_circle: Signed in as",
        )
        mock_st.text.assert_called_once_with("dr_smith@hospital.org")
        mock_st.button.assert_called_once_with(
            "Sign out",
            key="sign_out",
            icon=":material/logout:",
            on_click=streamlit_app._sign_out,
        )
        # The shared-workstation caveat sits under the button.
        assert mock_st.caption.call_args_list[-1].args == (streamlit_app.SIGN_OUT_NOTE,)

    def test_sign_out_purges_phi_then_logs_out(self, mock_st):
        mock_st.session_state.update(
            {
                "responses": [("a.wav", MagicMock())],
                "audio_sources": [b"a"],
                "run_id": RUN_ID,
                "run_model": "nova-3-pharma",
                "keyterms": ["Jane Doe"],
                f"transcript_{RUN_ID}_0": "Patient text.",
                f"reviewed_{RUN_ID}_0": True,
                "uploads_0": [MagicMock()],
                "recording_0": MagicMock(),
                "audit_review": (RUN_ID, frozenset({0})),
                "language": "en-GB",  # a setting, not PHI: kept
                "input_nonce": 0,
            }
        )

        streamlit_app._sign_out()

        # The signed-in audit identity ("audit_actor", seeded by conftest) goes too.
        assert mock_st.session_state == {
            "language": "en-GB",
            "input_nonce": 1,
            "access_ok": False,
        }
        mock_st.logout.assert_called_once_with()

    def test_sign_out_is_logged_under_the_signed_in_account(self, mock_st, capsys):
        streamlit_app._sign_out()

        (line,) = audit_lines(capsys.readouterr().out)
        assert (line["event"], line["outcome"], line["user"], line["auth"]) == (
            "logout",
            "success",
            "clinician@example.org",
            "oidc",
        )

    def test_purge_and_logout_run_even_if_the_audit_line_fails(self, mock_st):
        mock_st.session_state.pop("audit_actor")  # so _audit_actor raises
        mock_st.session_state["responses"] = [("a.wav", MagicMock())]

        with pytest.raises(RuntimeError, match="No audit actor"):
            streamlit_app._sign_out()

        assert "responses" not in mock_st.session_state
        mock_st.logout.assert_called_once_with()

    def test_nonce_starts_from_zero(self, mock_st):
        streamlit_app._purge_session()

        assert mock_st.session_state["input_nonce"] == 1

    def test_logout_runs_even_if_the_purge_fails(self, mock_st):
        with (
            patch.object(
                streamlit_app, "_purge_session", side_effect=RuntimeError("boom")
            ),
            pytest.raises(RuntimeError),
        ):
            streamlit_app._sign_out()

        mock_st.logout.assert_called_once_with()


class TestAccessRecheck:
    """The output fragment and review callbacks rerun without passing through the
    gate, so they re-check `access_ok` themselves."""

    @staticmethod
    def _render_output():
        # Unwrap st.fragment, which returns without calling the body in bare mode.
        inspect.unwrap(streamlit_app._render_output)()

    def test_output_renders_nothing_without_access(self, mock_st):
        mock_st.session_state.update(
            {"access_ok": False, "responses": [("a.wav", MagicMock())]}
        )

        self._render_output()

        assert mock_st.mock_calls == []

    @pytest.mark.parametrize("flag", [None, "true", 1])
    def test_access_flag_is_checked_by_identity(self, mock_st, flag):
        mock_st.session_state["access_ok"] = flag

        self._render_output()

        assert mock_st.mock_calls == []

    def test_output_renders_with_access(self, mock_st):
        self._render_output()

        mock_st.caption.assert_any_call(":material/description: Transcript")

    def test_header_names_the_runs_model(self, mock_st):
        mock_st.session_state.update(
            {
                "responses": [("a.wav", MagicMock())],
                "audio_sources": [None],
                "run_id": RUN_ID,
                "run_model": "nova-3-pharma",
            }
        )

        # Only the header is under test here; the panel and download have their own.
        with (
            patch.object(streamlit_app, "_output_panel", return_value=[]),
            patch.object(streamlit_app, "_transcript_download"),
        ):
            self._render_output()

        mock_st.caption.assert_any_call(
            ":material/description: Transcript · Nova-3 Pharma"
        )

    def test_header_names_no_model_without_results(self, mock_st):
        # A stale run_model with nothing to show must not label the placeholder.
        mock_st.session_state["run_model"] = "nova-3-pharma"

        self._render_output()

        mock_st.caption.assert_any_call(":material/description: Transcript")

    def test_on_edit_does_nothing_without_access(self, mock_st):
        mock_st.session_state.update({"access_ok": False, f"reviewed_{RUN_ID}_0": True})

        streamlit_app._on_edit(RUN_ID, 0)

        assert mock_st.session_state[f"reviewed_{RUN_ID}_0"] is True

    def test_on_review_does_nothing_without_access(self, mock_st, capsys):
        mock_st.session_state.update(
            {
                "access_ok": False,
                "responses": [("a.wav", MagicMock())],
                f"reviewed_{RUN_ID}_0": True,
            }
        )

        streamlit_app._on_review(RUN_ID, 0)

        assert capsys.readouterr().out == ""
        assert "audit_review" not in mock_st.session_state


class TestSignInExpiry:
    """The 12-hour sign-in limit holds where the gate does not run: the output
    fragment, the review callbacks, and the download callable. A full run's gate
    admits a sign-in; the clock then passes its expiry with no full run between."""

    T0 = 1_800_000_000.0
    EXPIRY = T0 - 60 + MAX_SESSION_AGE_SECONDS  # signed in a minute before T0

    @pytest.fixture
    def clock(self, mock_st, monkeypatch):
        now = [self.T0]
        monkeypatch.delenv(ALLOW_ANONYMOUS_ENV, raising=False)
        mock_st.get_option.return_value = {}
        _mock_secrets(mock_st, {"auth": dict(AUTH), "access": dict(ACCESS)})
        mock_st.user.to_dict.return_value = _signed_in(iat=self.T0 - 60)
        with (
            patch.object(streamlit_app, "time", SimpleNamespace(time=lambda: now[0])),
            patch.object(streamlit_app, "_authlib_installed", return_value=True),
            patch.object(streamlit_app, "_dotenv_has_opt_out", return_value=False),
        ):
            assert streamlit_app._access_gate() is not None  # the full run
            assert mock_st.session_state["access_expires_at"] == self.EXPIRY
            mock_st.reset_mock()
            yield now

    @staticmethod
    def _render_output():
        inspect.unwrap(streamlit_app._render_output)()

    def test_output_renders_until_the_expiry(self, mock_st, clock):
        clock[0] = self.EXPIRY  # the gate's own boundary: still fresh

        self._render_output()

        mock_st.caption.assert_any_call(":material/description: Transcript")
        mock_st.rerun.assert_not_called()

    def test_output_reruns_the_whole_app_once_expired(self, mock_st, clock):
        mock_st.session_state["responses"] = [("a.wav", MagicMock())]
        clock[0] = self.EXPIRY + 1

        self._render_output()

        # Nothing renders; the full rerun's gate then asks for a new sign-in.
        assert mock_st.mock_calls == [call.rerun(scope="app")]

    def test_review_callbacks_are_inert_once_expired(self, mock_st, clock, capsys):
        mock_st.session_state.update(
            {"responses": [("a.wav", MagicMock())], f"reviewed_{RUN_ID}_0": True}
        )
        clock[0] = self.EXPIRY + 1

        streamlit_app._on_review(RUN_ID, 0)
        streamlit_app._on_edit(RUN_ID, 0)

        assert mock_st.session_state[f"reviewed_{RUN_ID}_0"] is True
        assert "audit_review" not in mock_st.session_state
        events = [line["event"] for line in audit_lines(capsys.readouterr().out)]
        assert "review_signed_off" not in events and "review_reopened" not in events

    def _build(self, mock_st):
        # Rendered, reviewed and unlocked, before the expiry.
        streamlit_app._transcript_download(
            [("Jane_Doe_MRN12345.wav", MagicMock())],
            [("Patient Jane Doe reports rash on left forearm.", True)],
            RUN_ID,
        )
        return mock_st.download_button.call_args.args[1]

    def test_download_works_until_the_expiry(self, mock_st, clock):
        build = self._build(mock_st)
        clock[0] = self.EXPIRY

        with patch.object(streamlit_app, "st", Mock(spec=[])):
            assert build().startswith("Jane_Doe_MRN12345.wav\n")

    def test_download_clicked_after_the_expiry_is_refused(self, mock_st, clock, capsys):
        # A click reruns nothing (on_click="ignore"), so the callable checks itself —
        # from values captured at render, never st.* (a spec=[] Mock).
        build = self._build(mock_st)
        capsys.readouterr()
        clock[0] = self.EXPIRY + 1

        with (
            patch.object(streamlit_app, "st", Mock(spec=[])),
            pytest.raises(PermissionError) as exc,
        ):
            build()

        assert str(exc.value) == streamlit_app.DOWNLOAD_EXPIRED
        _assert_no_phi(str(exc.value))  # Streamlit logs it with a traceback
        assert capsys.readouterr().out == ""  # no file, so no transcript_downloaded

    @pytest.mark.parametrize("expires_at", ["soon", True, float("nan")])
    def test_an_unreadable_expiry_counts_as_expired(self, mock_st, expires_at):
        mock_st.session_state["access_expires_at"] = expires_at

        assert streamlit_app._access_ok() is False

    def test_an_anonymous_session_never_expires(self, mock_st):
        mock_st.session_state["access_expires_at"] = None

        with patch.object(streamlit_app, "time", SimpleNamespace(time=lambda: 1e12)):
            assert streamlit_app._access_ok() is True


class TestBareImport:
    """`import streamlit_app` runs the module-level access gate in bare mode, where
    `st.stop()` is a no-op, so everything after the gate must tolerate either
    outcome. The root conftest pins this process to one branch (opt-out "0", no
    secrets); these subprocesses exercise both, so every machine covers both — the
    gate's audit line included, which is written at import on either branch."""

    @pytest.mark.parametrize(
        ("opt_out", "expected", "logged"),
        [
            ("1", "anonymous", ("session_start", "anonymous", None)),
            (None, "None", ("access_denied", "none", "auth_not_configured")),
        ],
    )
    def test_import_survives_either_gate_outcome(self, opt_out, expected, logged):
        import os
        import subprocess
        import sys

        root = os.path.dirname(os.path.dirname(__file__))
        code = """
import dotenv
from streamlit import config

dotenv.load_dotenv = lambda *a, **k: False
dotenv.dotenv_values = lambda *a, **k: {}
config.set_option("secrets.files", [])

import streamlit_app

decision = streamlit_app.access_decision
print(decision.kind if decision is not None else None)
"""
        env = {k: v for k, v in os.environ.items() if k != ALLOW_ANONYMOUS_ENV}
        if opt_out is not None:
            env[ALLOW_ANONYMOUS_ENV] = opt_out
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        *audit_output, printed = result.stdout.strip().splitlines()
        assert printed == expected
        (line,) = audit_lines("\n".join(audit_output))
        assert (line["event"], line["auth"], line.get("reason")) == logged


class TestAppSmoke:
    """Run the whole script under a real Streamlit runtime (not the mock).

    Catches the class of errors the whole-module ``mock_st`` MagicMock cannot. Every
    run is hermetic: a known API key and the anonymous opt-out are set, ``.env`` is
    never read (``load_dotenv`` / ``dotenv_values`` no-op'd), and ``secrets.files``
    is empty, so neither a local ``.env`` nor a developer's
    ``~/.streamlit/secrets.toml`` can change an outcome. Sign-in is configured per
    run through ``at.secrets``, and a signed-in user is simulated by patching
    ``streamlit.user_info._get_user_info`` (private, but stable while streamlit is
    ``==``-pinned; a rename fails loudly). AppTest cannot produce production's
    logged-out shape, ``{"is_logged_in": False}`` — ``at.secrets`` is invisible to
    Streamlit's own ``[auth]`` probe — so that case is covered by the mocked
    TestAccessGate. Eight runs:

    - **empty state (anonymous)** — module load (``set_page_config`` ordering, the
      ``st.form`` structure, the output fragment) plus the idle UI: the one
      anonymous-mode banner, the placeholder caption, a disabled Run button, no
      review controls or download row, the inputs | output column split, and the
      Features control order
      (``language, keyterms, smart_format, diarize, dictation, measurements, redact``).
    - **seeded diarized** — renders the transcript panel for a diarized result,
      asserting the exact 1-based color-directive speaker lines, the
      Duration/Confidence/Low-confidence words metric cards, the nothing-flagged
      caption, and the dropped-playback caption; then walks the **review gate**:
      Download locked (no deferred file registered) until Reviewed is checked; an
      applied edit exported in place of Deepgram's text; the reviewed editor frozen
      (kept across a rerun, refusing input, and dropping a forged value — with an
      enabled-editor control); unchecking re-locks; an edit + check in one rerun
      keeps the edit but leaves it unchecked, with a toast saying so; the download
      row sits above the editor; and a new ``run_id`` reseeds everything and purges
      the old keys. Deferred ``data`` callables are recorded via ``MediaFileManager.
      add_deferred``, since AppTest never executes them.
    - **seeded flat** — the non-diarized render branch: an escaped transcript with
      no speaker labels, its one low-confidence word in bold orange under the legend
      caption, and a plain-text editor seed.
    - **sign-in required** — ``[auth]`` configured, nobody signed in (AppTest's
      injected ``test@example.com`` has no ``is_logged_in``): only the Sign in button
      and prompt render — no inputs, no output, no Run — and the opt-out is ignored.
    - **signed in, allowed** — the app renders with the account block in the
      sidebar (the email as plain ``st.text``); then **Sign out** purges the
      session's results, review state, and the nonce-keyed inputs (rotating
      ``input_nonce``) and stops at the sign-in screen.
    - **signed in, denied** — an email outside the allowlist: the refusal and a
      Sign out button, nothing else.
    - **no-key state** — clears ``DEEPGRAM_API_KEY`` so the key-required warning and
      the API-Key input render.
    - **not configured** — no ``[auth]`` and no opt-out: only the generic
      "Sign-in is unavailable" error; the operator's reason goes to stderr, which
      the parent asserts.

    The parent then reads the **audit trail** off the subprocess's stdout: every
    line one JSON object, grouped by session (one per AppTest instance) — exactly
    one ``session_start`` per session that passed the gate, whatever its rerun
    count; run 2's sign-off, one ``transcript_downloaded`` per export generated, and
    the reopening; ``logout`` after run 5's sign-in; ``access_denied`` with its
    reason for runs 6 and 8; nothing for the sign-in screen — and that no seeded
    filename or transcript text reaches stdout or stderr.

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
import time
from unittest.mock import MagicMock, patch

import dotenv
from streamlit import config
from streamlit.runtime.media_file_manager import MediaFileManager
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import Block
from streamlit.testing.v1.errors import AppTestError

from tests.helpers import RUN_ID

app = sys.argv[1]

# Hermetic for every run: a known key and the anonymous opt-out (so the first runs
# need no sign-in); .env is never read (each run re-executes the app's
# `from dotenv import ...`, picking these up); and no secrets file. Later runs
# configure sign-in via at.secrets, or clear the key / the opt-out.
os.environ["DEEPGRAM_API_KEY"] = "test-key"
os.environ["NOVA_ALLOW_ANONYMOUS"] = "1"
dotenv.load_dotenv = lambda *a, **k: False
dotenv.dotenv_values = lambda *a, **k: {}
config.set_option("secrets.files", [])

AUTH = {
    "redirect_uri": "http://localhost:8501/oauth2callback",
    "cookie_secret": "apptest-cookie-secret-0123456789abcdef",
    "client_id": "id",
    "client_secret": "secret",
    "server_metadata_url": "https://idp.example/.well-known/openid-configuration",
}


def _authed():
    # Sign-in configured through the app's own st.secrets read.
    at = AppTest.from_file(app, default_timeout=30)
    at.secrets["auth"] = AUTH
    at.secrets["access"] = {"allowed_email_domains": ["hospital.org"]}
    return at


def _signed_in_as(email):
    # Claims as Streamlit stores them after a default-provider sign-in.
    return patch(
        "streamlit.user_info._get_user_info",
        return_value={
            "is_logged_in": True,
            "email": email,
            "email_verified": True,
            "sub": "abc",
            "provider": "default",
            "iat": int(time.time()),
        },
    )

# AppTest never executes a download button's deferred `data` callable (only the
# browser's click path does), so record each one as Streamlit registers it; a test
# can then call it by the button's `deferred_file_id` to see what would be saved.
deferred = {}
_add_deferred = MediaFileManager.add_deferred


def _record_deferred(self, data_callable, *args, **kwargs):
    file_id = _add_deferred(self, data_callable, *args, **kwargs)
    deferred[file_id] = data_callable
    return file_id


MediaFileManager.add_deferred = _record_deferred


def _export(at):
    return deferred[at.download_button[0].proto.deferred_file_id]()


def _forge(widget, value):
    # Queue a value the browser could not send: AppTest's public .input() refuses a
    # disabled widget, so set its pending value directly (private attribute; the
    # enabled-editor control case below proves the forge still reaches the server).
    assert hasattr(widget, "_value"), "AppTest renamed Widget._value"
    widget._value = value


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


# 1) Empty state (anonymous) — module load (set_page_config / form / output fragment)
#    plus the idle UI: the placeholder caption and a Run button disabled with no input
#    selected.
at = AppTest.from_file(app, default_timeout=30).run()
assert not at.exception, at.exception
assert at.title[0].value == "Deepgram Medical Transcription"
assert any("Select audio, then click Run" in c.value for c in at.caption), [c.value for c in at.caption]
run = [b for b in at.button if b.label == "Run"]
assert run and run[0].disabled, "Run should be disabled with no audio input"
# The one warning is the anonymous-mode banner. The key was seeded, so no api-key
# warning fires — this pins the disable to the no-input branch (`not has_input`),
# not the no-key branch (`not api_key`).
assert [w.value for w in at.warning] and len(at.warning) == 1, [w.value for w in at.warning]
assert "NOVA_ALLOW_ANONYMOUS" in at.warning[0].value, at.warning[0].value
# No results, so no review controls and no download row.
assert not at.text_area and not at.checkbox and not at.download_button

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

# Features live in the sidebar and render in the intended order: inputs (Model,
# Language, Keyterm) first, the four toggles grouped, Redact deliberately last.
order = []
_widget_keys_in_order(at.sidebar, order)
assert [k for k in order if not k.startswith("FormSubmitter")] == [
    "model",
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
seeded.session_state["run_id"] = RUN_ID
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

# 2b) Review / sign-off on the same result. The editor is seeded with the plain
#     transcript, and Download stays locked — registering no transcript at all —
#     until the result is marked reviewed.
SEED = "Speaker 1: Hello.\\nSpeaker 2: Hi."
EDITED = "Speaker 1: Hello.\\nSpeaker 2: Hi there."
assert [t.label for t in seeded.text_area] == ["Transcript to export"]
assert seeded.text_area[0].key == "transcript_" + RUN_ID + "_0"
assert seeded.text_area[0].value == SEED and not seeded.text_area[0].disabled
assert [c.label for c in seeded.checkbox] == ["Reviewed against the audio"]
assert seeded.checkbox[0].value is False
dl = seeded.download_button[0]
assert dl.disabled and "Reviewed" in dl.help, (dl.disabled, dl.help)
assert dl.label == "Download locked — 0/1 reviewed", dl.label
assert not dl.proto.deferred_file_id  # data="" while locked: nothing to fetch
# The download row renders ABOVE the panel (above the fold OUTPUT_HEIGHT was sized
# for) even though it is filled after the review widgets it reads.
order = []
_widget_keys_in_order(seeded.main.columns[1], order)
assert order.index("download_transcripts") < order.index("transcript_" + RUN_ID + "_0"), order

# An applied edit leaves it unreviewed; checking Reviewed unlocks Download and
# freezes the editor, and the export is the edited text, not Deepgram's.
seeded.text_area[0].input(EDITED).run()
assert not seeded.exception, seeded.exception
assert seeded.checkbox[0].value is False and seeded.download_button[0].disabled
seeded.checkbox[0].check().run()
assert not seeded.exception, seeded.exception
dl = seeded.download_button[0]
assert not dl.disabled and dl.label == "Download transcript" and not dl.help
assert seeded.text_area[0].disabled
assert _export(seeded) == "sample.wav\\n" + EDITED

# The frozen editor keeps its edited value across a rerun, and the export uses it.
seeded.run()
assert seeded.text_area[0].value == EDITED and seeded.text_area[0].disabled
assert _export(seeded) == "sample.wav\\n" + EDITED
try:
    seeded.text_area[0].input("typed after sign-off")
    raise AssertionError("a reviewed editor accepted input")
except AppTestError:
    pass  # a browser user cannot type into it either

# A value for the frozen editor that arrives anyway (a stale UI's late edit, or a
# forged message) is dropped server-side — and never reaches the export, because
# the gate reads the widget's own return value, not its session-state key.
_forge(seeded.text_area[0], "FORGED")
seeded.run()
assert not seeded.exception, seeded.exception
assert seeded.text_area[0].value == EDITED and seeded.checkbox[0].value is True
assert _export(seeded) == "sample.wav\\n" + EDITED

# Unchecking reopens the editor and re-locks Download; the same forge on the now
# enabled editor does land (the control proving the case above was not vacuous).
seeded.checkbox[0].uncheck().run()
assert not seeded.text_area[0].disabled and seeded.download_button[0].disabled
_forge(seeded.text_area[0], "CONTROL")
seeded.run()
assert seeded.text_area[0].value == "CONTROL", seeded.text_area[0].value

# An edit and a check landing in the same rerun — in a browser, typing then clicking
# Reviewed, since the click is the blur that applies the edit: the edit is kept, the
# check is cleared, and a toast says to check it again. (Kept checked, the editor
# would render disabled this run and Streamlit would drop the edit.)
seeded.text_area[0].input(EDITED)
seeded.checkbox[0].check()
seeded.run()
assert not seeded.exception, seeded.exception
assert seeded.checkbox[0].value is False and seeded.download_button[0].disabled
assert seeded.text_area[0].value == EDITED and not seeded.text_area[0].disabled
assert [t.value for t in seeded.toast] == [
    "Edit applied, so Reviewed was cleared — check it again to sign off."
], [t.value for t in seeded.toast]

# A new Run (a full rerun under a new run_id) starts over: a freshly seeded,
# editable editor, an unchecked box, and the old run's keys purged.
seeded.session_state["run_id"] = "1" * 32
seeded.run()
assert not seeded.exception, seeded.exception
assert seeded.text_area[0].value == SEED and seeded.checkbox[0].value is False
assert "transcript_" + RUN_ID + "_0" not in seeded.session_state
assert "reviewed_" + RUN_ID + "_0" not in seeded.session_state

# 3) Seeded flat (non-diarized) result — the other render branch: an escaped
#    transcript with no speaker labels, rebuilt from its words so the one
#    low-confidence word renders in bold orange under the legend caption.
flat_words = [_word("Patient", None), _word("is", None), _word("stable.", None, 0.5)]
flat = _resp("Patient is stable.", flat_words, 12.0, 0.88)
flat_at = AppTest.from_file(app, default_timeout=30)
flat_at.session_state["responses"] = [("note.wav", flat)]
flat_at.session_state["audio_sources"] = [None]
flat_at.session_state["run_id"] = RUN_ID
flat_at.run()
assert not flat_at.exception, flat_at.exception
assert [m.value for m in flat_at.markdown] == ["Patient is :orange[**stable.**]"]
assert not any("Speaker" in m.value for m in flat_at.markdown)
assert any("bold orange" in c.value for c in flat_at.caption), [c.value for c in flat_at.caption]
assert [m.value for m in flat_at.metric][-1] == "1"
# The editor holds the plain text — no flag markup reaches what gets exported.
assert flat_at.text_area[0].value == "Patient is stable."

# 4) Sign-in required — [auth] configured and nobody signed in (AppTest's injected
#    {"email": "test@example.com"} has no is_logged_in, which never admits anyone):
#    only the prompt and the Sign in button render. The opt-out is still set, and
#    ignored now that [auth] exists. Nothing clicks Sign in: st.login would validate
#    against the real secrets file, which at.secrets does not reach.
login = _authed().run()
assert not login.exception, login.exception
assert [b.label for b in login.button] == ["Sign in"], [b.label for b in login.button]
assert any("Sign in" in i.value for i in login.info), [i.value for i in login.info]
assert not login.main.columns  # stopped at the gate: no inputs, no output, no Run
assert not login.warning, [w.value for w in login.warning]

# 5) Signed in and allowed — the app renders, with the account block in the sidebar.
diar_signed = _resp("Hello. Hi.", [_word("Hello.", 0), _word("Hi.", 1)], 3.5, 0.95)
with _signed_in_as("Dr@Hospital.org"):
    signed = _authed()
    signed.session_state["responses"] = [("sample.wav", diar_signed)]
    signed.session_state["audio_sources"] = [None]
    signed.session_state["run_id"] = RUN_ID
    signed.run()
assert not signed.exception, signed.exception
assert any(b.label == "Run" for b in signed.button), [b.label for b in signed.button]
# The email is plain full-opacity text under the caption, never an autolinked,
# caption-dimmed mailto link.
assert any(c.value.endswith("Signed in as") for c in signed.sidebar.caption), [c.value for c in signed.sidebar.caption]
assert [t.value for t in signed.sidebar.text] == ["dr@hospital.org"], [t.value for t in signed.sidebar.text]
assert not any("dr@hospital.org" in c.value for c in signed.sidebar.caption)
assert [b.label for b in signed.sidebar.button if b.label == "Sign out"] == ["Sign out"]
assert not signed.warning, [w.value for w in signed.warning]
assert signed.text_area and signed.checkbox  # results render for a signed-in session

# Sign out: the callback purges the session's PHI, then st.logout() clears the
# sign-in, so the same rerun stops at the sign-in screen. The uploader and recorder
# are keyed by input_nonce (0 until the first sign-out), so their state is there to
# purge — which makes the "not in" checks below meaningful.
assert "uploads_0" in signed.session_state and "recording_0" in signed.session_state
signed.button(key="sign_out").click().run()
assert not signed.exception, signed.exception
assert [b.label for b in signed.button] == ["Sign in"], [b.label for b in signed.button]
for key in (
    "responses",
    "audio_sources",
    "run_id",
    "transcript_" + RUN_ID + "_0",
    "reviewed_" + RUN_ID + "_0",
    "uploads_0",
    "recording_0",
):
    assert key not in signed.session_state, key
assert signed.session_state["input_nonce"] == 1
assert signed.session_state["access_ok"] is False

# 6) Signed in but denied — an email outside the allowlist sees the refusal and a
#    Sign out button, nothing else.
with _signed_in_as("dr@evil-hospital.org"):
    denied = _authed().run()
assert not denied.exception, denied.exception
assert any("not authorized" in e.value for e in denied.error), [e.value for e in denied.error]
assert [b.label for b in denied.button] == ["Sign out"], [b.label for b in denied.button]
assert not denied.main.columns

# 7) No-key state — clear the key (.env is never read, so nothing repopulates it);
#    the key-required warning + API-key input must render (under the anonymous banner).
os.environ.pop("DEEPGRAM_API_KEY", None)
nokey = AppTest.from_file(app, default_timeout=30).run()
assert not nokey.exception, nokey.exception
assert any("API key required" in w.value for w in nokey.warning), [w.value for w in nokey.warning]
assert nokey.text_input  # the API-Key password input renders when the key is missing

# 8) Not configured — no [auth] and no opt-out: visitors see only the generic error
#    (the operator's reason is logged to stderr, asserted by the parent), and nothing
#    past the gate renders.
os.environ.pop("NOVA_ALLOW_ANONYMOUS", None)
closed = AppTest.from_file(app, default_timeout=30).run()
assert not closed.exception, closed.exception
assert [e.value for e in closed.error] == ["Sign-in is unavailable. Contact your administrator."], [e.value for e in closed.error]
assert not closed.button and not closed.text_input and not closed.warning
assert not closed.main.columns
"""
        result = subprocess.run(
            [sys.executable, "-c", code, app],
            cwd=root,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        # Run 8's reason reached the operator: nova.access's logger writes to stderr.
        assert f"Sign-in is unavailable: {PROBLEM_NOT_CONFIGURED}" in result.stderr

        # The audit trail is stdout, one JSON object per line — and nothing else is
        # printed there under AppTest (no welcome banner), so every line must be one.
        stdout = [line for line in result.stdout.splitlines() if line.strip()]
        assert all(line.startswith('{"v":2') for line in stdout), stdout
        sessions: dict[str, list[dict]] = {}
        for line in stdout:
            event = json.loads(line)
            sessions.setdefault(event["session"], []).append(event)
        # One session per AppTest instance, in run order. Run 4 (the sign-in screen)
        # logs nothing, so it has none.
        runs = list(sessions.values())
        anonymous = ("anonymous", "anonymous")
        assert [(run[0]["user"], run[0]["auth"]) for run in runs] == [
            anonymous,  # 1 empty
            anonymous,  # 2 diarized + review walk
            anonymous,  # 3 flat
            ("dr@hospital.org", "oidc"),  # 5 signed in, then signed out
            ("dr@evil-hospital.org", "oidc"),  # 6 denied
            anonymous,  # 7 no key
            (None, "none"),  # 8 not configured
        ]
        trail = [[event["event"] for event in run] for run in runs]
        # Exactly one session_start per session that passed the gate, however many
        # reruns it had (run 2 had a dozen).
        assert [t.count("session_start") for t in trail] == [1, 1, 1, 1, 0, 1, 0]
        assert trail[0] == trail[2] == trail[5] == ["session_start"]
        assert trail[3] == ["session_start", "logout"]
        assert [(e["event"], e["reason"]) for e in runs[4]] == [
            ("access_denied", "domain_not_allowed")
        ]
        assert [(e["event"], e["reason"]) for e in runs[6]] == [
            ("access_denied", "auth_not_configured")
        ]
        # Run 2's review walk: the edited sign-off, one line per export generated
        # (three), then the uncheck. The frozen editor's dropped forge logs nothing.
        assert trail[1][:6] == [
            "session_start",
            "review_signed_off",
            "transcript_downloaded",
            "transcript_downloaded",
            "transcript_downloaded",
            "review_reopened",
        ], trail[1]
        signed = runs[1][1]
        assert (signed["run"], signed["edited"], signed["n_flagged"]) == (
            RUN_ID,
            True,
            0,
        )
        assert all(
            (e["run"], e["n_results"], e["n_edited"]) == (RUN_ID, 1, 1)
            for e in runs[1]
            if e["event"] == "transcript_downloaded"
        )
        # The edit + check in one rerun: Streamlit 1.64 runs the editor's callback
        # first, so the state never changed and nothing is logged. Were the checkbox's
        # to run first, a signed-off / reopened pair would be — never a lone
        # "reopened" for a sign-off that was never logged.
        assert trail[1][6:] in ([], ["review_signed_off", "review_reopened"]), trail[1]

        # A refused account's email is shown to that user and recorded only as the
        # user of its access_denied line (workforce identity, not PHI): never on
        # stderr, and in no other line.
        assert "evil-hospital" not in result.stderr
        (denied,) = [line for line in stdout if "evil-hospital" in line]
        assert json.loads(denied)["event"] == "access_denied"
        # No seeded filename or transcript text (edits and forged values included)
        # reaches either stream.
        for marker in (
            "sample.wav",
            "note.wav",
            "Hello.",
            "Hi there",
            "Patient is",
            "stable.",
            "FORGED",
            "CONTROL",
            "typed after sign-off",
        ):
            assert marker not in result.stdout + result.stderr, marker
