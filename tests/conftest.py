from unittest.mock import MagicMock, patch

import pytest

from tests.helpers import mock_word


@pytest.fixture
def mock_deepgram_cls():
    with patch("streamlit_app.DeepgramClient") as mock_cls:
        mock_response = MagicMock()

        alt = MagicMock()
        alt.transcript = "Life moves pretty fast really."
        alt.confidence = 0.98
        alt.words = [
            mock_word("Life", 0.99),
            mock_word("moves", 0.85),  # low confidence
            mock_word("pretty", 0.97),
            mock_word("fast", 0.80),  # low confidence
            mock_word("really.", 0.96),
        ]

        channel = MagicMock()
        channel.alternatives = [alt]
        channel.detected_language = "en"

        mock_response.results.channels = [channel]
        mock_response.metadata.duration = 3.5

        mock_cls.return_value.listen.v1.media.transcribe_file.return_value = (
            mock_response
        )
        yield mock_cls


@pytest.fixture
def mock_st():
    session_state = {}
    with patch("streamlit_app.st") as mock:
        mock.session_state = session_state
        yield mock
