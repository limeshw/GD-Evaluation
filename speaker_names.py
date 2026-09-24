from __future__ import annotations

import json
import re
import sys
from pathlib import Path


NAME_PATTERNS = (
    re.compile(
        r"\b(?:i\s+am|i'm|my\s+name\s+is)\s+"
        r"([A-Za-z][A-Za-z'-]{1,30}(?:\s+[A-Za-z][A-Za-z'-]{1,30}){0,1})"
        r"(?=\s*[,.;:!?]|\s+(?:and|i|we|would|will|like|want)\b|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:let\s+me\s+introduce\s+myself[,:]?\s*)"
        r"(?:i\s+am|i'm|my\s+name\s+is)\s+"
        r"([A-Za-z][A-Za-z'-]{1,30})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*([A-Za-z][A-Za-z'-]{2,30})\s+here\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*([A-Za-z][A-Za-z'-]{2,30})\s+(?:this\s+side|speaking)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*myself\s+([A-Za-z][A-Za-z'-]{2,30})\b",
        re.IGNORECASE,
    ),
)


def extract_introduced_name(text: str) -> str | None:
    for pattern in NAME_PATTERNS:
        match = pattern.search(text)
        if match:
            return " ".join(part.capitalize() for part in match.group(1).split())
    return None


def initial_text_by_speaker(output: dict, max_segments: int = 4) -> dict[str, str]:
    first_segments: dict[str, list[dict]] = {}
    for segment in sorted(output.get("segments", []), key=lambda item: item.get("start", 0)):
        speaker = str(segment.get("speaker", "UNKNOWN"))
        if speaker == "UNKNOWN":
            continue
        first_segments.setdefault(speaker, [])
        if len(first_segments[speaker]) < max_segments:
            first_segments[speaker].append(segment)

    return {
        speaker: " ".join(str(segment.get("text", "")) for segment in segments)
        for speaker, segments in first_segments.items()
    }


def attach_speaker_names(output: dict) -> dict:
    """Attach explicit self-introduced names without guessing unknown speakers."""
    speaker_ids = [str(speaker) for speaker in output.get("speakers", [])]
    name_by_speaker: dict[str, str] = {}
    used_names: set[str] = set()

    for speaker_id, initial_text in initial_text_by_speaker(output).items():
        name = extract_introduced_name(initial_text)
        if name and name not in used_names:
            name_by_speaker[speaker_id] = name
            used_names.add(name)

    named_segments = []
    for segment in output.get("segments", []):
        speaker_id = str(segment.get("speaker", "UNKNOWN"))
        named_segment = dict(segment)
        named_segment["speaker_id"] = speaker_id
        named_segment["speaker"] = name_by_speaker.get(speaker_id, speaker_id)
        named_segments.append(named_segment)

    return {
        "speakers": [name_by_speaker.get(speaker, speaker) for speaker in speaker_ids],
        "speaker_names": name_by_speaker,
        "segments": named_segments,
    }


def main() -> int:
    input_path = Path(sys.argv[1] if len(sys.argv) > 1 else "transcript.json")
    output_path = Path(sys.argv[2] if len(sys.argv) > 2 else "transcript_named.json")

    if not input_path.exists():
        print(f"Error: transcript file not found: {input_path}", file=sys.stderr)
        return 1

    try:
        output = attach_speaker_names(json.loads(input_path.read_text(encoding="utf-8")))
        output_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Error: could not process transcript: {exc}", file=sys.stderr)
        return 1

    print(f"Named transcript written to: {output_path.resolve()}")
    return 0

 
if __name__ == "__main__":
    raise SystemExit(main())
