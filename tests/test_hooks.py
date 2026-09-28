"""Integration tests for the Claude Code hook scripts in `.claude/hooks/`.

These shell out to the real scripts with the same event JSON Claude Code feeds
them on stdin, asserting exit codes and side effects. They are skipped when the
toolchain the hooks depend on (bash/jq/uv) is not on PATH, so the pure-Python
suite still runs in a bare environment.

Two hooks, with deliberately opposite failure modes: block-secrets.sh is a
security guard and fails CLOSED (an unreadable event refuses the edit), while
gates.sh is a quality gate and fails OPEN (a missing toolchain stands aside
rather than wedging the session).
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HOOKS_DIR = REPO_ROOT / ".claude" / "hooks"
BASH = shutil.which("bash") or "bash"

# The hooks orchestrate these tools; without them the scripts can't run.
_MISSING = [tool for tool in ("bash", "jq", "uv") if shutil.which(tool) is None]
pytestmark = pytest.mark.skipif(
    bool(_MISSING), reason=f"hook tests require {_MISSING} on PATH"
)


def run_hook(script, event, *, env_overrides=None):
    """Run a hook script with `event` as stdin JSON; return the finished process.

    Invoked via an absolute `bash` (matching how settings.json now launches the
    hooks, and so a PATH override in env_overrides can't hide the interpreter),
    with CLAUDE_PROJECT_DIR pinned to the repo root.
    """
    env = {**os.environ, "CLAUDE_PROJECT_DIR": str(REPO_ROOT)}
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [BASH, str(HOOKS_DIR / script)],
        input=json.dumps(event),
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
    )


def run_hook_raw(script, raw_stdin):
    """Run a hook with arbitrary (possibly non-JSON) stdin."""
    return subprocess.run(
        [BASH, str(HOOKS_DIR / script)],
        input=raw_stdin,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env={**os.environ, "CLAUDE_PROJECT_DIR": str(REPO_ROOT)},
    )


def _edit_event(path):
    """An Edit/Write PostToolUse/PreToolUse payload targeting `path`."""
    return {"tool_input": {"file_path": str(path)}}


class TestBlockSecrets:
    @pytest.mark.parametrize(
        "relpath",
        [
            ".env",
            ".env.local",
            ".env.production",
            ".streamlit/secrets.toml",
            ".ENV",  # case-insensitive: same inode as .env on a case-insensitive FS
            ".Env",
            ".streamlit/SECRETS.TOML",
            ".streamlit/example.secrets.toml",  # *.secrets.toml — so never a template name
            ".secrets.toml",  # Dynaconf secrets file
            "secrets.toml",  # bare secrets.toml under the root (nested match)
            ".envrc",  # direnv
            ".envrc.local",  # direnv source_env_if_exists target
            ".ENVRC.LOCAL",
        ],
    )
    def test_denies_secret_files(self, relpath):
        proc = run_hook("block-secrets.sh", _edit_event(REPO_ROOT / relpath))
        assert proc.returncode == 2
        assert "Blocked" in proc.stderr

    @pytest.mark.parametrize("path", ["secrets.toml", ".env", ".ENVRC"])
    def test_denies_bare_relative_secret_paths(self, path):
        # A relative path with no directory component must still be caught.
        proc = run_hook("block-secrets.sh", {"tool_input": {"file_path": path}})
        assert proc.returncode == 2

    @pytest.mark.parametrize(
        "relpath",
        [
            ".env.example",  # tracked template
            ".ENV.EXAMPLE",  # template, case-insensitively
            "nova/config.py",
            "README.md",
            ".streamlit/config.toml",  # non-secret toml must stay editable
            ".streamlit/secrets.toml.example",  # tracked sign-in template
        ],
    )
    def test_allows_non_secret_files(self, relpath):
        proc = run_hook("block-secrets.sh", _edit_event(REPO_ROOT / relpath))
        assert proc.returncode == 0

    def test_missing_file_path_is_noop(self):
        proc = run_hook("block-secrets.sh", {"tool_input": {}})
        assert proc.returncode == 0

    def test_fails_closed_when_jq_missing(self):
        # A security guard must not silently allow edits if it can't parse the
        # event. With jq off PATH the hook refuses (exit 2) rather than no-op.
        proc = run_hook(
            "block-secrets.sh",
            _edit_event(REPO_ROOT / ".env"),
            env_overrides={"PATH": "/var/empty"},
        )
        assert proc.returncode == 2
        assert "jq not found" in proc.stderr

    def test_fails_closed_on_unparseable_event(self):
        # jq present but the event is not JSON: the guard must refuse rather than
        # capture jq's empty stdout and fall through to exit 0 (which it did).
        proc = run_hook_raw("block-secrets.sh", "not json")
        assert proc.returncode == 2
        assert "unparseable hook event" in proc.stderr


class TestGates:
    """The Stop hook. Only the short-circuit branches are exercised: the real-run
    path shells out to `uv run pytest`, which would re-collect this module and
    recurse."""

    def test_loop_guard_exits_without_running(self):
        # stop_hook_active=true means we already blocked once this stop cycle;
        # the hook must bail immediately rather than re-running the gates.
        proc = run_hook("gates.sh", {"stop_hook_active": True})

        assert proc.returncode == 0
        assert proc.stdout == ""
        assert proc.stderr == ""

    def test_fails_open_without_toolchain(self):
        # The mirror image of block-secrets.sh: a quality gate that cannot read
        # its own loop guard (no jq) must stand aside, not wedge the session by
        # blocking every stop.
        proc = run_hook("gates.sh", {}, env_overrides={"PATH": "/var/empty"})

        assert proc.returncode == 0
        assert proc.stderr == ""
