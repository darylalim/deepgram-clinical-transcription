import io
import json
import wave
from unittest.mock import MagicMock

# A fixed, well-formed run id (32 lowercase hex, the shape `uuid.uuid4().hex` gives) for
# tests that seed or assert the per-Run review widget keys.
RUN_ID = "0" * 32


def mock_word(text: str, confidence: float, speaker: int | None = None):
    w = MagicMock()
    w.punctuated_word = text
    w.confidence = confidence
    w.speaker = speaker
    return w


def mock_upload(name: str, data: bytes, size: int | None = None):
    """Mimic a Streamlit UploadedFile with .name, .size, and .getvalue()."""
    f = MagicMock()
    f.name = name
    f.size = len(data) if size is None else size
    f.getvalue.return_value = data
    return f


def wav_bytes(seconds: int) -> bytes:
    """Minimal mono 16-bit WAV whose duration equals `seconds` (framerate = 1 Hz)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(1)
        wf.writeframes(b"\x00\x00" * seconds)
    return buf.getvalue()


def audit_lines(out: str) -> list[dict]:
    """Parse captured stdout as the audit trail: every non-empty line one JSON object.

    Strict on purpose — anything else on stdout fails the parse. Read stdout with
    `capsys`, never `caplog`: the audit logger does not propagate, and caplog then
    sees its records only depending on test order (and never the stdout line a log
    shipper reads).
    """
    return [json.loads(line) for line in out.splitlines() if line.strip()]
