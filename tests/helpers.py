import io
import wave
from unittest.mock import MagicMock


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
