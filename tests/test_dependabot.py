"""Guard tests for the Dependabot configuration.

``.github/dependabot.yml`` is pure config: it runs no Python and is invisible to
``ruff``/``ty``/``pytest`` (which only look at ``.py`` files), so the rest of the
suite can never catch a regression in it. Yet it is load-bearing for the repo's
supply-chain posture: every GitHub Actions ``uses:`` pin under
``.github/workflows/`` (both ``ci.yml`` and ``release.yml``) is frozen to an
immutable commit SHA, and Dependabot's ``github-actions`` ecosystem is the *only*
thing that bumps those SHAs (and their ``# vX.Y.Z`` comments) when a new action
release ships. Drop or misconfigure this file and the pins silently rot.

These tests parse it with PyYAML and pin the invariants that keep that automation
alive: schema ``version: 2``, and a ``github-actions`` update rooted at ``/`` on a
schedule. The root directory is deliberate — for the ``github-actions`` ecosystem
a single entry at ``/`` monitors *every* workflow under ``.github/workflows/``, so
no per-workflow entry is needed when ``release.yml`` is added alongside ``ci.yml``.
Parsing (rather than substring matching) catches a schema bump, a narrowed
directory, or a dropped schedule that a plain ``in`` check would miss.
"""

from pathlib import Path

import pytest
import yaml

DEPENDABOT = Path(__file__).resolve().parent.parent / ".github" / "dependabot.yml"


@pytest.fixture(scope="module")
def config() -> dict:
    # A malformed config (bad indent, duplicate key, ...) raises here and fails
    # every test — the failure mode that silently stops the pin bumps.
    return yaml.safe_load(DEPENDABOT.read_text(encoding="utf-8"))


def _update(config: dict, ecosystem: str) -> dict:
    for update in config.get("updates", []):
        if isinstance(update, dict) and update.get("package-ecosystem") == ecosystem:
            return update
    raise AssertionError(f"no {ecosystem} update entry in dependabot.yml")


def _github_actions_update(config: dict) -> dict:
    return _update(config, "github-actions")


def test_config_is_valid_v2(config: dict) -> None:
    assert isinstance(config, dict)
    # Dependabot only honors version 2; any other value is silently ignored.
    assert config.get("version") == 2


def test_github_actions_ecosystem_is_rooted_at_repo_root(config: dict) -> None:
    update = _github_actions_update(config)
    # `directory: /` (or a `directories` list containing "/") makes the
    # github-actions ecosystem watch ALL workflows under .github/workflows/,
    # so ci.yml AND release.yml pins are both covered by this one entry.
    # Normalize to a list first so the membership check is always exact, never a
    # substring match (a mis-authored scalar `directories: "/foo"` must not pass).
    raw = update.get("directories")
    raw = raw if raw is not None else update.get("directory")
    dirs = raw if isinstance(raw, list) else [raw]
    assert "/" in dirs, f"github-actions updates must be rooted at '/', got {dirs!r}"


def test_github_actions_updates_are_scheduled(config: dict) -> None:
    update = _github_actions_update(config)
    # No schedule.interval => Dependabot never opens bump PRs and the pins rot.
    schedule = update.get("schedule")
    assert isinstance(schedule, dict), "github-actions update needs a schedule"
    assert schedule.get("interval"), "schedule must set an interval"


def test_python_dependencies_are_tracked_from_the_root(config: dict) -> None:
    # The uv ecosystem reads pyproject.toml + uv.lock at the repo root; without
    # this entry, releases of authlib / cryptography / streamlit (sign-in and the
    # app itself) are tracked by hand. Auto-merge never applies to these
    # (test_automerge_workflow.py::test_only_github_actions_bumps_are_auto_merged).
    update = _update(config, "uv")
    raw = update.get("directories")
    raw = raw if raw is not None else update.get("directory")
    dirs = raw if isinstance(raw, list) else [raw]
    assert "/" in dirs, f"uv updates must be rooted at '/', got {dirs!r}"
    schedule = update.get("schedule")
    assert isinstance(schedule, dict) and schedule.get("interval"), (
        "uv update needs a schedule.interval"
    )
