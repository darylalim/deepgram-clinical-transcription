"""scripts/calibrate.py: threshold calibration on the operator's own audio.

CI never calls Deepgram; `run` takes the transcribe call as a seam. Its input is
clinical audio, so the PHI test mirrors TestRunAudit's: patient-like filenames,
reference text and transcript words go in, and none may come out."""

import importlib.util
from pathlib import Path

import pytest

from nova.config import LOW_CONFIDENCE_THRESHOLD, MODELS
from tests.test_scoring import response

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "calibrate.py"
MARKERS = ["jane", "doe", "mrn12345", "hydroxyzine", "fakepatient"]


@pytest.fixture(scope="module")
def cal():
    spec = importlib.util.spec_from_file_location("calibrate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # import only: main() is behind __main__
    return module


def _folder(tmp_path, files):
    for name, content in files.items():
        (tmp_path / name).write_bytes(
            content.encode() if isinstance(content, str) else content
        )
    return tmp_path


class TestPairs:
    def test_pairs_by_name_and_counts_unpaired_audio(self, cal, tmp_path):
        folder = _folder(
            tmp_path,
            {
                "b.wav": b"B",
                "b.txt": "second",
                "a.MP3": b"A",  # extension case is ignored
                "a.txt": "first",
                "c.flac": b"C",  # no reference
                "notes.txt": "a reference with no audio is ignored",
            },
        )

        found, unpaired = cal.pairs(folder)

        assert [(p.name, text) for p, text in found] == [
            ("a.MP3", "first"),
            ("b.wav", "second"),
        ]
        assert unpaired == 1


class TestRun:
    def test_every_model_scores_every_file_with_default_options(self, cal, tmp_path):
        folder = _folder(tmp_path, {"a.wav": b"A", "a.txt": "take apixaban today"})
        calls = []

        def transcribe(audio, opts):
            calls.append((audio, opts["model"]))
            return response(["Take", "apixaban", "today."])

        lines = cal.run(folder, transcribe)

        assert calls == [(b"A", model) for model in MODELS]
        assert sum(line.startswith("== ") for line in lines) == len(MODELS)
        assert any("WER 0.0%  dropped 0" in line for line in lines)

    def test_rule_check_flags_an_over_limit_share(self, cal, tmp_path):
        folder = _folder(tmp_path, {"a.wav": b"A", "a.txt": "take apixaban today"})
        # Every word below the threshold: 100% flagged, over the 20% limit.
        lines = cal.run(
            folder,
            lambda audio, opts: response(["Take", "apixaban", "today."], low={0, 1, 2}),
        )

        checks = [line for line in lines if "rule check" in line]
        assert len(checks) == len(MODELS)
        assert all(
            f"{LOW_CONFIDENCE_THRESHOLD:.2f} flags 100.0%" in c and "consider 0.85" in c
            for c in checks
        )

    def test_no_pairs_runs_nothing(self, cal, tmp_path):
        folder = _folder(tmp_path, {"a.wav": b"A"})

        lines = cal.run(folder, lambda audio, opts: pytest.fail("no request expected"))

        assert lines == ["0 paired file(s); 1 audio file(s) without a .txt skipped"]

    def test_output_carries_no_phi(self, cal, tmp_path):
        # A success whose reference and transcript name the patient, and a failure
        # whose exception carries the filename — as Deepgram's ApiError can.
        folder = _folder(
            tmp_path,
            {
                "Jane_Doe_MRN12345.wav": b"A",
                "Jane_Doe_MRN12345.txt": "Jane Doe takes hydroxyzine fakepatient",
                "Jane_Doe_MRN12345_b.wav": b"B",
                "Jane_Doe_MRN12345_b.txt": "Jane Doe again",
            },
        )

        class ApiError(Exception):
            status_code = 500

        def transcribe(audio, opts):
            if audio == b"B":
                raise ApiError(f"POST failed for Jane_Doe_MRN12345_b.wav {folder}")
            return response(
                ["Jane", "Doe", "takes", "hydroxyzine", "fakepatient"], low={3}
            )

        out = "\n".join(cal.run(folder, transcribe)).lower()

        assert "#2: request failed (status 500)" in out
        for marker in [*MARKERS, str(tmp_path).lower()]:
            assert marker not in out, marker
