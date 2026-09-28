# Deepgram Medical Transcription

[![CI](https://github.com/darylalim/deepgram-medical-transcription/actions/workflows/ci.yml/badge.svg)](https://github.com/darylalim/deepgram-medical-transcription/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Streamlit application for medical transcription using Deepgram's Nova-3 Medical model (English-only), built on a framework-free core (`nova/`) that handles option building, batching, and response parsing.

> **Reference implementation — not certified for clinical use.** You are responsible for your own Deepgram BAA and PHI handling before any real patient data flows through this app. See [License](#license).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/screenshot-dark.png">
  <img alt="The app showing a diarized clinic-visit transcript: a Features sidebar with keyterms, Diarize and Measurements enabled; an Upload tab holding clinic-visit.wav; and the Transcript tab with Duration and Confidence metrics, an audio player, and color-coded Speaker 1 / Speaker 2 lines." src="docs/screenshot-light.png">
</picture>

<sub>Screenshot uses a synthetic, fictional transcript — no real patient audio.</sub>

## Features

- **Batch transcription** from two input sources — upload files or record from the microphone.
- **Nova-3 Medical** speech-to-text across eight English variants.
- **Keyterm prompting** — boost recognition of specialized vocabulary (drug names, procedures).
- **Speaker diarization** with color-coded per-speaker transcript lines.
- **Low-confidence flags** — words Deepgram scored below 90% confidence are shown in **bold orange**, with a per-result count, so review starts where the model was least sure.
- **Redaction** — PII for de-identification, plus PHI, PCI, and number groups (PHI and Numbers also strip clinical content).
- **Smart formatting**, spoken **dictation** commands, and **measurement** abbreviation.
- **Download** — the transcript as plain text (`.txt`), with a multi-file batch combined into one file.
- **"Reading room" light & dark themes** — a clinical blue-slate palette with a teal accent that follows your OS light/dark setting (switchable in Settings), WCAG AA throughout in both modes, with self-hosted fonts (no third-party CDN).

## Prerequisites

- [uv](https://docs.astral.sh/uv/) — manages the Python toolchain and dependencies (install: `curl -LsSf https://astral.sh/uv/install.sh | sh`).
- Python 3.12+ — `uv sync` fetches a compatible interpreter if you don't already have one.
- A Deepgram API key — create a free one at the [Deepgram Console](https://console.deepgram.com).

## Setup

1. Install dependencies: `uv sync`
2. Create your env file: `cp .env.example .env`, then set `DEEPGRAM_API_KEY`.

## Usage

```bash
uv run streamlit run streamlit_app.py
```

If `DEEPGRAM_API_KEY` is not set, the app prompts for it inline.

**Select audio** from the input tabs on the left:

- **Upload** — up to 100 audio files (mp3, m4a, wav, flac, ogg; max 200 MB each)
- **Record** — record from microphone (max 30 minutes)

A **Features** panel in the left sidebar holds the request options, closed by a **Run** button. If you populate both input tabs, Run transcribes a single one by priority — **Upload, then Record** — and shows a notice naming which ran and which was ignored.

- **Language** — English variants (Nova-3 Medical is English-only)
- **Keyterm Prompting** — type specialized vocabulary (drug names, procedures, names), Enter to add each, up to 100, to boost recognition
- **Smart Format** (on by default) — punctuation, paragraph breaks, and entity formatting
- **Diarize** (off by default) — labels speaker turns as Speaker 1, Speaker 2, … in the transcript (speakers are numbered, not named by role); use it for clinician–patient encounters
- **Dictation** (off by default) — turns spoken commands like "period" / "new paragraph" into punctuation (also enables punctuation); for a single clinician dictating, not encounters. Run warns when Dictation and Diarize are both on
- **Measurements** (off by default) — abbreviates spoken units (e.g. "five milligrams" → "5 mg"). Volumes come out as lowercase "ml" / "l", which [ISMP](https://www.ismp.org/recommendations/error-prone-abbreviations-list) lists as error-prone (use mL / L), so review volumes before clinical use
- **Redact** (none by default) — replaces selected information with redaction tags. Four groups are selectable: **PII** de-identifies (names, locations, IDs); **PHI** removes clinical content itself (conditions, drugs, injuries); **PCI** redacts card numbers; **Numbers** redacts any run of three or more digits plus Deepgram's number-like entities (e.g. dates, times, ages, phone and account numbers, medical statistics, locations) — so clinical values are redacted unpredictably ("500 mg" always, shorter doses, vitals, and lab values only sometimes).

Once a request runs:

- **Live progress** — a status panel tracks the batch, with a toast when it finishes.
- **Transcript** — a single view under a **Transcript** header beside the inputs (stacked below them on narrow screens), displaying the response; multiple results are labeled and divided per file.
- **Metrics & download** — each transcript is topped with **Duration**, **Confidence**, and **Low-confidence words** metric cards, and a **Download transcript** (`.txt`) button above the panel saves every result in the batch as one file.
- **Low-confidence flags** — words scored below 90% model confidence render in **bold orange**, under a caption that says whether any were flagged. If the per-word data is missing, or does not reproduce the transcript exactly, the transcript is shown unhighlighted with a notice to review every word. See [What a flag means](#what-a-flag-means).
- **Audio player** — pinned above the scrollable transcript. Inline audio over 25 MB — a large upload or a long recording — shows a notice instead of the player to limit memory.
- **Diarized view** — with **Diarize** on, the transcript is split into color-coded `Speaker 1:`, `Speaker 2:`, … lines.

### What a flag means

A flag is a pointer to the audio, not a verdict — check flagged words first, but review the whole transcript:

- **Threshold** — a word is flagged when Deepgram's per-word confidence is **below 90%** (`LOW_CONFIDENCE_THRESHOLD` in `nova/config.py`). Deepgram describes that score as a calibrated probability; 90% is a starting point to re-evaluate on your own de-identified audio. It is fixed server-side, not a per-user setting.
- **Unmarked words can still be wrong.** A model can be confidently wrong, and **keyterm prompting inflates confidence** for the boosted terms — a misheard drug name you listed as a keyterm may come back unflagged.
- **Redaction tags are never flagged** (`[SSN_1]`, `[REDACTED]`, …): the spoken content is gone, so there is nothing to check it against.
- **A flagged formatted number means "check the whole value."** Smart Format merges a spoken number into one token (a dose, a phone number) with one confidence, so the flag covers all of it.
- The flags describe Deepgram's original words; the downloaded `.txt` carries the same words, without highlighting.

**Troubleshooting** — if transcription fails with a per-file error (rather than the app refusing to start), check that `DEEPGRAM_API_KEY` is valid and has available credit: an invalid or expired key is reported as a per-item transcription failure, not a startup error.

## Architecture

- **`nova/`** — the framework-free core (no Streamlit imports): `config` (constants), `transcribe` (`build_options` + `transcribe_batch`), `results` (response walkers and low-confidence flagging). Speakers are Deepgram's native 0-based integers here.
- **`streamlit_app.py`** — the Streamlit UI; a thin adapter over `nova/` that adds widgets, session state, and the renderers (which display speakers 1-based).

## Testing

```bash
uv run pytest         # tests
uv run ruff check .   # lint
uv run ruff format .  # format
uv run ty check .     # type check
```

Tests mock the Deepgram client — no real API calls. The core is tested directly (`tests/test_transcribe.py`, `tests/test_results.py`), the Streamlit adapter in `tests/test_streamlit_app.py`, the dev hooks in `tests/test_hooks.py`, and the project's config — the CI and release workflows, the Dependabot config, and the license — in `tests/test_ci_workflow.py`, `tests/test_release_workflow.py`, `tests/test_dependabot.py`, and `tests/test_license.py`.

**Continuous integration** — `.github/workflows/ci.yml` (GitHub Actions) runs these same four gates plus `uv sync --locked` across a Python 3.12 + 3.13 matrix on every push to `main`, every pull request, and manual dispatch. It needs no secrets: tests mock Deepgram, so CI never calls the API. The two matrix legs report as the `checks (3.12)` / `checks (3.13)` status checks that `main` requires, so the job id and matrix values are a branch-protection contract — `tests/test_ci_workflow.py` pins them.

## Releases

Releases are cut by bumping the version. Edit `[project].version` in `pyproject.toml` and land it on `main`:

```toml
version = "0.9.0"
```

`.github/workflows/release.yml` does the rest. It reads the declared version, and if `v0.9.0` is not tagged yet it runs the full CI matrix at that commit, creates the tag, and publishes a [GitHub Release](https://github.com/darylalim/deepgram-medical-transcription/releases) with notes auto-generated from the merged pull requests since the previous release. A push that does not change the version is a no-op, so re-running is always safe.

Three things worth knowing:

- **Nothing is tagged from a red commit.** The `main` ruleset requires CI but does not require a pull request, so a direct push could otherwise skip it. The workflow gates on `ci.yml` itself before tagging.
- **Hand-tagging still works** as an escape hatch — it is the only way to release a commit that is not `main`'s HEAD — but the tag must match the version declared at that commit, or the run fails rather than publishing a Release that lies.
- **It needs no secrets.** Tagging and publishing happen in one run using the default `GITHUB_TOKEN`. A split "tag here, publish there" design would need a long-lived credential, because GitHub will not start a workflow run from a ref that token created.

## Claude Code hooks

The repo ships **Claude Code hooks** in `.claude/` (shared via `settings.json`) that run these checks automatically while you work: they format, lint, and type-check edited Python, block edits to secret files (`.env`, `.streamlit/secrets.toml`), and run the test suite when a turn finishes. Newly added hooks need approval before firing (`/hooks`). Personal overrides go in `.claude/settings.local.json` (gitignored). See CLAUDE.md for the full breakdown.

## License

Released under the [MIT License](LICENSE). The software is provided "as is," without warranty of any kind; it is a reference implementation and is not certified for clinical use. Operators are responsible for their own Deepgram BAA and PHI handling before processing real patient data.
