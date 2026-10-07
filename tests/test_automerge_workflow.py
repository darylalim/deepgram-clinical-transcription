"""Guard tests for the Dependabot auto-merge workflow.

``.github/workflows/dependabot-auto-merge.yml`` is pure config: it runs no Python
and is invisible to ``ruff``/``ty``/``pytest`` (which only ever look at ``.py``
files), so the rest of the suite can never catch a regression in it. These tests
parse the workflow with PyYAML and pin the invariants it exists to hold:

1. **It is the only write-privileged, PR-triggered workflow in the repo, and it
   stays safe.** ``pull_request_target`` is load-bearing — a ``pull_request`` run
   triggered by Dependabot gets a read-only ``GITHUB_TOKEN`` that a
   ``permissions:`` block cannot elevate, and enabling auto-merge needs write. The
   trade is only sound because the job **never checks the pull request out** and
   executes nothing from it; ``test_never_checks_out_pull_request_code`` is the
   test that keeps that true. A future ``actions/checkout`` step here would hand a
   writable token to fork-controlled code.
2. **It only ever acts on Dependabot's own PRs, and never on a major.** The job is
   gated on the ``dependabot[bot]`` actor *and* this repository (so a fork cannot
   auto-merge on its own copy), and the merge step is skipped for
   ``version-update:semver-major`` — the bump class where a breaking change hides.
3. **It stays least-privilege and injection-safe.** Exactly
   ``{contents: write, pull-requests: write}``, a job timeout, no cancellation
   mid-merge, every ``uses:`` SHA-pinned, and no untrusted ``github.event.*`` /
   ``head_ref`` / ``inputs.*`` interpolated into a shell ``run:`` step.

Parsing (rather than substring matching over raw text) is what lets these catch a
smuggled scope, a dropped actor gate, a checkout step, or a malformed workflow
GitHub would refuse to schedule — all of which a plain ``in`` check would miss.
"""

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "dependabot-auto-merge.yml"

# Attacker-controllable expression contexts that must never be interpolated raw
# into a shell `run:` step (the GitHub Actions script-injection class). This
# matters more here than anywhere else in the repo: this is the one workflow that
# holds a writable token on a pull-request event.
_UNTRUSTED = re.compile(r"\$\{\{[^}]*\bgithub\.(?:event|head_ref|ref)\b")
_UNTRUSTED_INPUTS = re.compile(r"\$\{\{[^}]*\binputs\b")
_SHA_PINNED = re.compile(r"\S+@[0-9a-f]{40}\b")
# Any way a step could pull the pull request's own code into the writable-token
# job: a checkout action, or a hand-rolled fetch/checkout in a run step.
_CHECKOUT_RUN = re.compile(r"\b(?:git\s+(?:checkout|fetch|clone)|gh\s+pr\s+checkout)\b")


@pytest.fixture(scope="module")
def workflow() -> dict:
    # A malformed workflow (bad indent, duplicate key, ...) raises here and fails
    # every test — the one failure mode that silently disables auto-merge.
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _triggers(workflow: dict) -> dict:
    # PyYAML parses YAML 1.1, so the bare `on:` key deserializes to the boolean
    # True, not the string "on" (the classic Actions gotcha). Accept either.
    on = workflow.get("on", workflow.get(True))
    assert isinstance(on, dict), f"unexpected `on:` shape: {on!r}"
    return on


def _job(workflow: dict) -> dict:
    return workflow["jobs"]["auto-merge"]


def _steps(workflow: dict) -> list[dict]:
    return [s for s in _job(workflow)["steps"] if isinstance(s, dict)]


def _run_steps(workflow: dict) -> list[str]:
    return [s["run"] for s in _steps(workflow) if "run" in s]


def test_workflow_is_valid_and_has_the_auto_merge_job(workflow: dict) -> None:
    assert isinstance(workflow, dict)
    assert "auto-merge" in workflow["jobs"]


def test_triggers_only_on_pull_request_target(workflow: dict) -> None:
    # `pull_request_target` is deliberate (a Dependabot-triggered `pull_request`
    # run cannot hold a writable token), and it must be the *only* trigger — a
    # `push`/`workflow_dispatch` here would run the merge outside a PR context.
    on = _triggers(workflow)
    assert set(on) == {"pull_request_target"}, f"unexpected triggers: {set(on)}"


def test_never_checks_out_pull_request_code(workflow: dict) -> None:
    # THE invariant that makes `pull_request_target` + a writable token safe.
    for step in _steps(workflow):
        uses = step.get("uses", "")
        assert "checkout" not in uses, (
            f"pull_request_target job must not check out PR code, got {uses!r}"
        )
    for run in _run_steps(workflow):
        assert not _CHECKOUT_RUN.search(run), (
            f"pull_request_target job must not fetch PR code, got {run!r}"
        )


def test_permissions_are_exactly_what_auto_merge_needs(workflow: dict) -> None:
    # Merge the PR + enable auto-merge on it, and nothing more.
    assert workflow.get("permissions") == {
        "contents": "write",
        "pull-requests": "write",
    }


def test_job_is_gated_on_dependabot_and_this_repository(workflow: dict) -> None:
    condition = _job(workflow).get("if", "")
    assert "dependabot[bot]" in condition, "job must be gated on the Dependabot actor"
    assert "github.repository ==" in condition, (
        "job must be gated on this repository so a fork cannot auto-merge"
    )


def test_major_bumps_are_never_auto_merged(workflow: dict) -> None:
    # A major is where a breaking change hides; those stay manual.
    merge_steps = [s for s in _steps(workflow) if "gh pr merge" in s.get("run", "")]
    assert merge_steps, "no `gh pr merge` step found"
    for step in merge_steps:
        condition = step.get("if", "")
        assert "version-update:semver-major" in condition, (
            f"merge step must exclude majors, got if: {condition!r}"
        )
        assert "!=" in condition, "major check must be an exclusion"


def test_only_github_actions_bumps_are_auto_merged(workflow: dict) -> None:
    # Python (uv) bumps change production code — including sign-in (authlib,
    # cryptography) — and CI mocks Deepgram and the identity provider, so they must
    # always wait for a human. fetch-metadata reports the branch's ecosystem
    # segment, so the exact value is `github_actions` (underscore).
    merge_steps = [s for s in _steps(workflow) if "gh pr merge" in s.get("run", "")]
    assert merge_steps, "no `gh pr merge` step found"
    for step in merge_steps:
        condition = step.get("if", "")
        assert (
            "steps.metadata.outputs.package-ecosystem == 'github_actions'" in condition
        ), f"merge step must be limited to github_actions, got if: {condition!r}"
        assert "&&" in condition and "||" not in condition, (
            "the ecosystem and major checks must both hold"
        )


def test_merge_step_queues_behind_required_checks(workflow: dict) -> None:
    joined = "\n".join(_run_steps(workflow))
    assert "gh pr merge" in joined
    # `--auto` is the whole point: queue the merge behind the required checks
    # rather than merging immediately. Without it this workflow would merge
    # Dependabot PRs before CI ever ran.
    assert "--auto" in joined, "merge must be queued with --auto, not immediate"


def test_a_mid_merge_run_is_never_cancelled(workflow: dict) -> None:
    concurrency = workflow.get("concurrency")
    assert isinstance(concurrency, dict), "workflow needs a concurrency group"
    assert concurrency.get("cancel-in-progress") is False


def test_job_has_a_timeout(workflow: dict) -> None:
    assert isinstance(_job(workflow).get("timeout-minutes"), int)


def test_run_steps_have_no_untrusted_interpolation(workflow: dict) -> None:
    for run in _run_steps(workflow):
        assert not _UNTRUSTED.search(run), f"untrusted github context in: {run!r}"
        assert not _UNTRUSTED_INPUTS.search(run), f"raw inputs.* in run step: {run!r}"


def test_every_action_is_sha_pinned(workflow: dict) -> None:
    used = [s["uses"] for s in _steps(workflow) if "uses" in s]
    assert used, "expected at least the dependabot/fetch-metadata step"
    for uses in used:
        assert _SHA_PINNED.match(uses), f"action must be SHA-pinned, got {uses!r}"
