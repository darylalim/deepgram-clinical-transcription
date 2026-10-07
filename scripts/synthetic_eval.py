"""Synthetic-audio evaluation: compare the models on a fictional pharmacy call.

Deepgram's Aura-2 text-to-speech reads a scripted refill call (fictional patient, no
patient data), so the script is the ground truth. Each model in `MODELS` (plus the
default model with the drug names as keyterms) transcribes it, at 16 kHz and at
phone-quality 8 kHz, `RUNS` times each — TTS output varies run to run. For each it
reports word error rate, words **dropped** (missing speech, which no flag can show),
drug/specialty terms recognized, and how the low-confidence flag separates wrong
words from right ones at 0.90 and 0.85.

Generated speech is cleaner than real calls: treat results as a best case, and use
de-identified clinical audio to calibrate LOW_CONFIDENCE_THRESHOLD. Numbers and dates
are left out of scoring, since Smart Format writes "fifty" or "50" for the same speech.

Makes real, billed requests (TTS plus transcription) with DEEPGRAM_API_KEY from the
environment or `.env`. Prints counts and missed term names only; everything printed
comes from the fictional script.

    uv run python scripts/synthetic_eval.py [runs]
"""

import io
import os
import re
import sys
import wave
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepgram import DeepgramClient  # noqa: E402
from dotenv import dotenv_values  # noqa: E402

from nova.config import DEFAULT_MODEL, MODELS  # noqa: E402
from nova.results import first_alternative, is_low_confidence, word_token  # noqa: E402
from nova.transcribe import build_options  # noqa: E402

RUNS = 3
RATES = {"clean 16 kHz": 16000, "phone 8 kHz": 8000}
THRESHOLDS = (0.90, 0.85)
PHARMACIST, PATIENT = "aura-2-thalia-en", "aura-2-orion-en"
SCRIPT = [
    (PHARMACIST, "Thank you for calling the pharmacy refill line. Can I have the patient's name and date of birth?"),
    (PATIENT, "Yes, it's Maria Lopez, March 14, 1961."),
    (PHARMACIST, "Thank you. Which prescriptions would you like to refill today?"),
    (PATIENT, "I need my metoprolol succinate 50 milligrams, my apixaban 5 milligrams, and the empagliflozin 10 milligrams."),
    (PHARMACIST, "Got it. I also see levetiracetam 500 milligrams and hydroxyzine 25 milligrams on your profile."),
    (PATIENT, "The hydroxyzine, yes. Not the hydralazine, my cardiologist stopped that last month."),
    (PHARMACIST, "Understood. Your rosuvastatin and clopidogrel were filled two weeks ago, so those are not due yet."),
    (PATIENT, "Okay. Can you also check on my tamsulosin and my montelukast?"),
    (PHARMACIST, "Tamsulosin is ready. Montelukast needs a new prescription from your doctor."),
    (PATIENT, "My endocrinologist switched me from sitagliptin to Ozempic, and she mentioned tirzepatide as an option."),
    (PHARMACIST, "I see the semaglutide order. Are you still taking pantoprazole and escitalopram?"),
    (PATIENT, "Pantoprazole yes. Escitalopram I take 10 milligrams at night."),
    (PHARMACIST, "And the valacyclovir, gabapentin, and prednisone taper from last month are finished?"),
    (PATIENT, "Yes, all finished. I am still on methotrexate once a week with folic acid."),
    (PHARMACIST, "Great. I will fill the metoprolol, Eliquis, Jardiance, hydroxyzine, and tamsulosin today."),
]  # fmt: skip
DRUGS = [
    "metoprolol", "succinate", "apixaban", "empagliflozin", "levetiracetam",
    "hydroxyzine", "hydralazine", "rosuvastatin", "clopidogrel", "tamsulosin",
    "montelukast", "sitagliptin", "ozempic", "tirzepatide", "semaglutide",
    "pantoprazole", "escitalopram", "valacyclovir", "gabapentin", "prednisone",
    "methotrexate", "folic", "eliquis", "jardiance",
]  # fmt: skip
TERMS = [*DRUGS, "cardiologist", "endocrinologist"]  # scored; DRUGS are the keyterms

# Left out of scoring: numbers and the spoken date are formatting, not recognition.
NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "march",
}  # fmt: skip
UNITS = {"mg": "milligrams", "milligram": "milligrams"}


def norm(text: str) -> list[str]:
    """Lowercase words without punctuation, numbers or dates, units unified.

    Applied identically to the script and the transcript, so formatting differences
    (Smart Format writes "fifty" or "50", "03/14/1961" for "March 14, 1961") are
    not scored as recognition errors.
    """
    text = re.sub(r"[^a-z0-9' ]", " ", text.lower().replace("-", " "))
    words = [w.strip("'") for w in text.split() if w.strip("'")]
    return [UNITS.get(w, w) for w in words if not w.isdigit() and w not in NUMBER_WORDS]


@dataclass(frozen=True)
class Score:
    wer: float  # percent of reference words
    dropped: int  # reference words with no transcript word at all (missing speech)
    terms_missed: list[str]
    flagged: dict[float, float]  # threshold -> percent of transcript words flagged
    caught: dict[float, tuple[int, int]]  # threshold -> (wrong words flagged, wrong)


def score(response: Any, reference: list[str]) -> Score:
    """Score one transcript against the normalized reference.

    A transcript word is "wrong" when it is not part of an exact alignment with the
    reference. Flags use the app's own `is_low_confidence`, per Deepgram word (a
    smart-formatted word may normalize to several pieces, all sharing its flag).
    """
    pieces: list[tuple[str, Any]] = []
    for word in getattr(first_alternative(response), "words", None) or []:
        pieces.extend((p, word) for p in norm(str(word_token(word))))
    hyp = [p for p, _ in pieces]
    correct = [False] * len(hyp)
    errors = dropped = 0
    matcher = SequenceMatcher(a=reference, b=hyp, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            correct[j1:j2] = [True] * (j2 - j1)
            continue
        errors += max(i2 - i1, j2 - j1)
        dropped += max(0, (i2 - i1) - (j2 - j1))
    present = set(hyp)
    flagged: dict[float, float] = {}
    caught: dict[float, tuple[int, int]] = {}
    wrong = [not ok for ok in correct]
    for threshold in THRESHOLDS:
        flags = [is_low_confidence(w, threshold) for _, w in pieces]
        flagged[threshold] = 100 * sum(flags) / len(flags) if flags else 0.0
        caught[threshold] = (
            sum(f and x for f, x in zip(flags, wrong, strict=True)),
            sum(wrong),
        )
    return Score(
        wer=100 * errors / len(reference),
        dropped=dropped,
        terms_missed=[t for t in TERMS if t not in present],
        flagged=flagged,
        caught=caught,
    )


def synthesize(client: Any, rate: int) -> bytes:
    """Read the script with TTS (linear16 mono at `rate`) into one WAV, 0.4 s between turns."""
    frames = b""
    for voice, line in SCRIPT:
        frames += b"".join(
            client.speak.v1.audio.generate(
                text=line,
                model=voice,
                encoding="linear16",
                container="none",
                sample_rate=rate,
            )
        )
        frames += b"\0\0" * int(rate * 0.4)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(frames)
    return buf.getvalue()


def configs() -> dict[str, dict[str, Any]]:
    """Every model, plus the default model with the drug names as keyterms."""
    found: dict[str, dict[str, Any]] = {model: {"model": model} for model in MODELS}
    found[f"{DEFAULT_MODEL}+keyterms"] = {"model": DEFAULT_MODEL, "keyterms": DRUGS}
    return found


def main() -> None:
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else RUNS
    key = os.environ.get("DEEPGRAM_API_KEY") or dotenv_values(".env").get(
        "DEEPGRAM_API_KEY"
    )
    if not key:
        sys.exit("DEEPGRAM_API_KEY not found in the environment or .env")
    client = DeepgramClient(api_key=key)
    reference = norm(" ".join(line for _, line in SCRIPT))
    print(
        f"{len(reference)} scored reference words, {len(TERMS)} terms, {runs} run(s) per audio quality"
    )
    for quality, rate in RATES.items():
        for run in range(1, runs + 1):
            audio = synthesize(client, rate)
            print(f"\n== {quality} #{run}")
            for name, opts in configs().items():
                response = client.listen.v1.media.transcribe_file(
                    request=audio, **build_options(**opts)
                )
                s = score(response, reference)
                cells = "  ".join(
                    f"@{t:.2f} flagged {s.flagged[t]:4.1f}% caught {s.caught[t][0]}/{s.caught[t][1]}"
                    for t in THRESHOLDS
                )
                print(
                    f"  {name:24s} WER {s.wer:4.1f}%  dropped {s.dropped:3d}  "
                    f"terms {len(TERMS) - len(s.terms_missed)}/{len(TERMS)}  {cells}"
                )
                if s.terms_missed:
                    print(f"  {'':24s} missed: {', '.join(s.terms_missed)}")


if __name__ == "__main__":
    main()
