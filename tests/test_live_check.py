"""Guard for scripts/live_check.py, which CI never runs (it needs a real key).

Keeps it in step with the core: every case must still be a valid `build_options`
call for every model, so a renamed or removed option fails here instead of on the
next person who runs the script."""

import importlib.util
import inspect
from pathlib import Path

import pytest

from nova.config import MODELS
from nova.transcribe import build_options

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "live_check.py"


@pytest.fixture(scope="module")
def live_check():
    spec = importlib.util.spec_from_file_location("live_check", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # import only: main() is behind __main__
    return module


@pytest.mark.parametrize("model", list(MODELS))
def test_every_case_is_a_valid_request(live_check, model):
    for opts in live_check.CASES.values():
        assert build_options(model=model, **opts)["model"] == model


def test_cases_cover_every_feature_option(live_check):
    # Derived from build_options itself, so a new option fails here until the
    # script exercises it (`model` is varied per run, not per case).
    options = set(inspect.signature(build_options).parameters) - {"model"}
    covered = {key for opts in live_check.CASES.values() for key in opts}
    assert covered == options
