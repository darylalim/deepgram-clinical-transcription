"""Calibrate LOW_CONFIDENCE_THRESHOLD on your own de-identified clinical audio.

Point it at a folder of audio files, each with a reference transcript of the same
name (`visit-01.wav` + `visit-01.txt`). Every model in `MODELS` transcribes every
file with the app's default options, and the transcripts are scored against the
references (scripts/scoring.py). For each model it prints, summed over all files:
word error rate, words **dropped** (missing speech, which no flag can show), and for
each candidate threshold the share of words flagged, how many wrong words the flag
caught, and how many flags were on wrong words. Use it to check the 0.90 default and
the rule in nova/config.py: fall back to 0.85 if 0.90 flags more than 20% of words.

PHI: the audio is sent to Deepgram, so a BAA must cover it even when de-identified.
Nothing identifying is printed: no filenames, transcript or reference words — files
appear as #1, #2, … in sorted order — and a failed request prints only its HTTP
status. Numbers and dates are not scored (see scripts/scoring.py), so check doses
against the audio separately.

Makes real, billed requests (one per file per model) with DEEPGRAM_API_KEY from the
environment or `.env`.

    uv run python scripts/calibrate.py path/to/folder
"""

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepgram import DeepgramClient  # noqa: E402
from dotenv import dotenv_values  # noqa: E402

from nova.config import AUDIO_EXTENSIONS, LOW_CONFIDENCE_THRESHOLD, MODELS  # noqa: E402
from nova.transcribe import build_options  # noqa: E402
from scripts.scoring import Counts, norm, score  # noqa: E402

THRESHOLDS = (0.95, LOW_CONFIDENCE_THRESHOLD, 0.85, 0.80, 0.70)
FLAG_SHARE_LIMIT = 20.0  # percent; nova/config.py's fallback rule


def pairs(folder: Path) -> tuple[list[tuple[Path, str]], int]:
    """(audio, reference text) for each audio file with a same-name `.txt`, sorted by
    name, plus how many audio files had no reference."""
    found: list[tuple[Path, str]] = []
    unpaired = 0
    for audio in sorted(folder.iterdir()):
        if not audio.is_file() or audio.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        reference = audio.with_suffix(".txt")
        if reference.is_file():
            found.append((audio, reference.read_text(encoding="utf-8")))
        else:
            unpaired += 1
    return found, unpaired


def report(model: str, total: Counts, n_files: int) -> list[str]:
    """Summary lines for one model: counts and percentages only, never words."""
    lines = [
        f"== {model}: {n_files} file(s), {total.n_ref} scored reference words",
        f"   WER {total.wer:.1f}%  dropped {total.dropped} word(s)  wrong {total.wrong} word(s)",
    ]
    for t in sorted(total.flagged, reverse=True):
        flags = total.flagged[t]
        precision = 100 * total.caught[t] / flags if flags else 0.0
        recall = 100 * total.caught[t] / total.wrong if total.wrong else 0.0
        marker = "  <- current" if t == LOW_CONFIDENCE_THRESHOLD else ""
        lines.append(
            f"   @{t:.2f}  flagged {total.flagged_pct(t):5.1f}%  "
            f"caught {total.caught[t]}/{total.wrong} ({recall:.0f}%)  "
            f"flags on wrong words {precision:.0f}%{marker}"
        )
    share = total.flagged_pct(LOW_CONFIDENCE_THRESHOLD)
    verdict = (
        "over the limit: consider 0.85"
        if share > FLAG_SHARE_LIMIT
        else "within the limit"
    )
    lines.append(
        f"   rule check: {LOW_CONFIDENCE_THRESHOLD:.2f} flags {share:.1f}% of words "
        f"(limit {FLAG_SHARE_LIMIT:.0f}%): {verdict}"
    )
    if total.dropped:
        lines.append("   note: dropped words are invisible to every threshold")
    return lines


def run(folder: Path, transcribe: Callable[[bytes, dict[str, Any]], Any]) -> list[str]:
    """Score every paired file with every model; return the printable lines.

    `transcribe(audio_bytes, options)` returns a Deepgram response (a seam for tests).
    """
    found, unpaired = pairs(folder)
    lines = [
        f"{len(found)} paired file(s); {unpaired} audio file(s) without a .txt skipped"
    ]
    if not found:
        return lines
    references = [norm(text) for _, text in found]
    for model in MODELS:
        total = Counts()
        for index, ((audio, _), reference) in enumerate(
            zip(found, references, strict=True), 1
        ):
            try:
                response = transcribe(audio.read_bytes(), build_options(model=model))
            except Exception as exc:  # report by index and status only, keep going
                lines.append(
                    f"   #{index}: request failed (status {getattr(exc, 'status_code', None)})"
                )
                continue
            total = total + score(response, reference, THRESHOLDS)
        lines.extend(report(model, total, len(found)))
    return lines


def main() -> None:
    if len(sys.argv) != 2 or not Path(sys.argv[1]).is_dir():
        sys.exit("usage: uv run python scripts/calibrate.py path/to/folder")
    key = os.environ.get("DEEPGRAM_API_KEY") or dotenv_values(".env").get(
        "DEEPGRAM_API_KEY"
    )
    if not key:
        sys.exit("DEEPGRAM_API_KEY not found in the environment or .env")
    media = DeepgramClient(api_key=key).listen.v1.media
    for line in run(
        Path(sys.argv[1]),
        lambda audio, opts: media.transcribe_file(request=audio, **opts),
    ):
        print(line)


if __name__ == "__main__":
    main()
