"""Guard for scripts/synthetic_eval.py, which CI never runs (billed, needs a key).

Keeps its requests in step with the core; its scoring lives in scripts/scoring.py
and is tested in tests/test_scoring.py."""

import importlib.util
from pathlib import Path

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


class TestRequests:
    def test_every_config_is_a_valid_request_and_covers_every_model(self, ev):
        configs = ev.configs()
        for opts in configs.values():
            build_options(**opts)
        assert {opts["model"] for opts in configs.values()} == set(MODELS)

    def test_every_scored_term_is_in_the_script(self, ev):
        reference = set(ev.norm(" ".join(line for _, line in ev.SCRIPT)))
        assert set(ev.TERMS) <= reference
