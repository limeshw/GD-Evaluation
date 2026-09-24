from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, NoReturn

from dotenv import load_dotenv

from speaker_names import attach_speaker_names

load_dotenv()


class PrototypeError(Exception):
    """Expected, user-facing errors for the proof-of-concept."""


def fail(message: str, code: int = 1) -> NoReturn:
    raise PrototypeError(message)


def format_timestamp(seconds: float) -> str:
    total_centiseconds = max(0, int(round(seconds * 100)))
    hours, remainder = divmod(total_centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, centiseconds = divmod(remainder, 100)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def speaker_name(index: int) -> str:
    # Spreadsheet-style labels: A, B, C, ... Z, AA, AB, ...
    letters = ""
    value = index
    while True:
        value, remainder = divmod(value, 26)
        letters = chr(ord("A") + remainder) + letters
        if value == 0:
            break
        value -= 1
    return f"SPEAKER_{letters}"


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    label: str


def interval_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def extract_speaker_turns(annotation: object) -> list[Interval]:
    turns: list[Interval] = []
    for turn, _, label in annotation.itertracks(yield_label=True):
        turns.append(Interval(float(turn.start), float(turn.end), str(label)))
    return turns


def remap_speakers(turns: Iterable[Interval]) -> tuple[list[Interval], list[str]]:
    mapping: dict[str, str] = {}
    remapped: list[Interval] = []
    ordered_speakers: list[str] = []

    for turn in turns:
        if turn.label not in mapping:
            mapped = speaker_name(len(mapping))
            mapping[turn.label] = mapped
            ordered_speakers.append(mapped)
        remapped.append(Interval(turn.start, turn.end, mapping[turn.label]))

    return remapped, ordered_speakers


def best_speaker_for_segment(
    segment_start: float,
    segment_end: float,
    turns: Iterable[Interval],
) -> str:
    best_label = "UNKNOWN"
    best_overlap = 0.0

    for turn in turns:
        overlap = interval_overlap(segment_start, segment_end, turn.start, turn.end)
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = turn.label

    return best_label if best_overlap > 0.0 else "UNKNOWN"


def load_diarization_pipeline():
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError:
        fail(
            "Missing dependency while importing pyannote.audio/torch. "
            "Install project dependencies with 'uv sync'."
        )

    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        fail("HF_TOKEN is missing. Set HF_TOKEN in your environment before running.")

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=token,
        )
    except Exception as exc:
        fail(f"Failed to load Pyannote Community-1: {exc.__class__.__name__}: {exc}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    return pipeline


def run_diarization(pipeline, audio_path: Path):
    try:
        output = pipeline(str(audio_path))
    except Exception as exc:
        fail(f"Failed to run diarization: {exc.__class__.__name__}: {exc}")

    annotation = getattr(output, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = getattr(output, "speaker_diarization", None)

    if annotation is None:
        fail("No speaker diarization output was produced.")

    turns = extract_speaker_turns(annotation)
    if not turns:
        fail("No speech or speakers were detected by Pyannote.")

    return remap_speakers(turns)


def load_whisper_model():
    try:
        import torch
        import whisper
    except ImportError:
        fail(
            "Missing dependency while importing Whisper/torch. "
            "Install project dependencies with 'uv sync'."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = os.getenv("WHISPER_MODEL", "small").strip() or "small"

    try:
        model = whisper.load_model(model_name, device=device)
    except Exception as exc:
        fail(f"Failed to load Whisper model: {exc.__class__.__name__}: {exc}")

    return model, torch.cuda.is_available()


def transcribe_audio(model, audio_path: Path, use_fp16: bool):
    try:
        result = model.transcribe(
            str(audio_path),
            fp16=use_fp16,
            language=os.getenv("WHISPER_LANGUAGE", "en").strip() or "en",
            task="transcribe",
            temperature=0.0,
            beam_size=5,
            condition_on_previous_text=False,
            initial_prompt=(
                "This is a group discussion. Each speaker may introduce themselves "
                "by saying their name. Preserve personal names accurately."
            ),
        )
    except Exception as exc:
        fail(f"Failed to transcribe audio: {exc.__class__.__name__}: {exc}")

    segments = result.get("segments") or []
    cleaned = []
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        cleaned.append(
            {
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": text,
            }
        )

    if not cleaned:
        fail("No speech was detected by Whisper.")

    return cleaned


def normalize_audio(audio_path: Path) -> Path:
    """Convert input audio to a stable mono 16 kHz PCM WAV for both models."""
    temporary_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    normalized_path = Path(temporary_file.name)
    temporary_file.close()

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-af",
        "aresample=async=1:first_pts=0,apad=pad_dur=0.1",
        "-y",
        str(normalized_path),
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        normalized_path.unlink(missing_ok=True)
        fail(
            "ffmpeg was not found. Install ffmpeg and make sure it is available "
            "on PATH. Whisper and audio normalization require it."
        )
    except subprocess.CalledProcessError as exc:
        normalized_path.unlink(missing_ok=True)
        detail = exc.stderr.strip() or "unknown ffmpeg error"
        fail(f"Could not decode or normalize the audio file: {detail}")

    return normalized_path


def align_segments(whisper_segments: list[dict], diarized_turns: list[Interval]) -> list[dict]:
    aligned = []

    for segment in whisper_segments:
        speaker = best_speaker_for_segment(
            float(segment["start"]),
            float(segment["end"]),
            diarized_turns,
        )
        aligned.append(
            {
                "speaker": speaker,
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": segment["text"],
            }
        )

    return aligned


def print_transcript(aligned_segments: list[dict]) -> None:
    for segment in aligned_segments:
        print(
            f"[{format_timestamp(segment['start'])} - {format_timestamp(segment['end'])}] "
            f"{segment['speaker']}:"
        )
        print(segment["text"])
        print()


def process_audio(audio_path: Path) -> dict:
    normalized_path = normalize_audio(audio_path)
    try:
        pipeline = load_diarization_pipeline()
        diarized_turns, speakers = run_diarization(pipeline, normalized_path)

        whisper_model, use_cuda = load_whisper_model()
        whisper_segments = transcribe_audio(
            whisper_model,
            normalized_path,
            use_fp16=use_cuda,
        )
        aligned_segments = align_segments(whisper_segments, diarized_turns)

        return attach_speaker_names({
            "speakers": speakers,
            "segments": aligned_segments,
        })
    finally:
        normalized_path.unlink(missing_ok=True)


def save_transcript(output: dict, output_path: Path = Path("transcript.json")) -> None:
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    if len(sys.argv) != 2:
        fail("Usage: uv run python version_1.py sample.wav")

    audio_path = Path(sys.argv[1]).expanduser().resolve()
    if not audio_path.exists():
        fail(f"Audio file not found: {audio_path}")
    if not audio_path.is_file():
        fail(f"Audio path is not a file: {audio_path}")

    try:
        output = process_audio(audio_path)
        save_transcript(output)
        print_transcript(output["segments"])
        return 0
    except PrototypeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
