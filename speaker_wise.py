from __future__ import annotations

import json
import sys
from pathlib import Path


INPUT_PATH = Path("transcript.json")
OUTPUT_PATH = Path("speaker_wise_transcript.json")
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}


def relative_source_path(audio_path: Path) -> str:
    try:
        relative_path = audio_path.resolve().relative_to(Path.cwd().resolve())
        return relative_path.as_posix()
    except ValueError:
        return audio_path.name


def resolve_audio_source(argument: str | None) -> Path:
    if argument:
        audio_path = Path(argument).expanduser().resolve()
        if not audio_path.exists():
            raise ValueError(f"Audio file not found: {audio_path}")
        if not audio_path.is_file():
            raise ValueError(f"Audio path is not a file: {audio_path}")
        return audio_path

    candidates = sorted(
        path
        for path in Path("audio").glob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise ValueError(
            "No audio file was found in audio/. Pass the original audio path "
            "as the second command-line argument."
        )
    raise ValueError(
        "Multiple audio files were found in audio/. Pass the specific original "
        "audio path as the second command-line argument."
    )


def build_speaker_wise_transcript(transcript: dict, audio_path: Path) -> dict:
    speaker_names = transcript.get("speaker_names", {})
    speaker_ids = transcript.get("speakers", [])
    segments_by_speaker: dict[str, list[dict]] = {}
    ids_by_speaker: dict[str, str] = {}

    for speaker_id, speaker_name in speaker_names.items():
        ids_by_speaker.setdefault(speaker_name, speaker_id)

    for speaker_name in speaker_ids:
        speaker_name = str(speaker_name)
        segments_by_speaker.setdefault(speaker_name, [])
        ids_by_speaker.setdefault(speaker_name, speaker_name)

    for segment in transcript.get("segments", []): 
        speaker = str(segment.get("speaker", "UNKNOWN"))
        speaker_id = str(segment.get("speaker_id", ids_by_speaker.get(speaker, speaker)))
        segments_by_speaker.setdefault(speaker, [])
        ids_by_speaker.setdefault(speaker, speaker_id)
        segments_by_speaker[speaker].append(
            {
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": str(segment["text"]),
            }
        )

    result: dict[str, dict] = {}
    source_file = relative_source_path(audio_path)
    for participant_number, (speaker, segments) in enumerate(
        segments_by_speaker.items(), start=1
    ):
        segments.sort(key=lambda segment: (segment["start"], segment["end"]))
        speaking_time = sum(segment["end"] - segment["start"] for segment in segments)
        full_text = " ".join(segment["text"] for segment in segments)
        transcript_data = {
            "full_text": full_text,
            "segments": segments,
        }
        result[speaker] = {
            "participant_id": f"P{participant_number}",
            "name": speaker,
            "speaker_id": ids_by_speaker[speaker],
            "segment_count": len(segments),
            "total_speaking_time": round(speaking_time, 3),
            "full_text": full_text,
            "segments": segments,
            "audio": {
                "source_file": source_file,
                "segments": [
                    {"start": segment["start"], "end": segment["end"]}
                    for segment in segments
                ],
            },
            "transcript": transcript_data,
        }

    return {"speakers": result}


def main() -> int:
    if not INPUT_PATH.exists():
        print(f"Error: input file not found: {INPUT_PATH}")
        return 1

    try:
        audio_path = resolve_audio_source(sys.argv[1] if len(sys.argv) > 1 else None)
        transcript = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
        output = build_speaker_wise_transcript(transcript, audio_path)
        OUTPUT_PATH.write_text(
            json.dumps(output, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"Error: could not organize transcript: {exc}")
        return 1

    print(f"Speakers: {len(output['speakers'])}")
    print()
    for speaker, data in output["speakers"].items():
        print(speaker)
        print(f"  Segments: {data['segment_count']}")
        print(f"  Speaking time: {data['total_speaking_time']} seconds")
        print()

    print(f"Created: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
