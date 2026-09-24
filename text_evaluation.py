from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:3b"
OUTPUT_PATH = Path("text_evaluation.json")

PARAMETERS = (
    "participation",
    "communication",
    "leadership",
    "analytical_skill",
    "quantitative_knowledge",
    "topic_relevance",
)

EVALUATION_SCHEMA = {
    "type": "object",
    "required": list(PARAMETERS),
    "properties": {
        parameter: {
            "type": "object",
            "required": ["score", "evidence", "reason"],
            "properties": {
                "score": {"type": "integer", "minimum": 0, "maximum": 10},
                "evidence": {"type": "string", "maxLength": 500},
                "reason": {"type": "string", "maxLength": 500},
            },
        }
        for parameter in PARAMETERS
    },
}


def fail(message: str) -> None:
    raise ValueError(message)


def load_features(path: Path) -> dict:
    if not path.exists():
        fail(f"Text features file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"Invalid JSON in {path}: {exc}")
    if not isinstance(data, dict) or not isinstance(data.get("speakers"), dict):
        fail("text_features.json must contain a 'speakers' object.")
    if not data["speakers"]:
        fail("text_features.json contains no speakers.")
    return data


def selected_features(data: dict) -> dict:
    allowed = {
        "word_count",
        "sentence_count",
        "average_sentence_length",
        "max_sentence_length",
        "ttr",
        "lexical_diversity",
        "mtld",
        "topic_similarity",
        "topic_coverage",
        "semantic_similarity",
        "numeric_entity_count",
        "quantity_count",
        "percentage_count",
        "date_count",
        "number_of_quantitative_statements",
    }
    return {key: data[key] for key in allowed if key in data}


def build_prompt(topic: str, participant: str, data: dict) -> str:
    transcript = str(data.get("original_text", "")).strip()
    if not transcript:
        fail(f"Speaker '{participant}' has no original transcript text.")

    prompt_input = {
        "gd_topic": topic,
        "participant": participant,
        "transcript": transcript,
        "text_features": selected_features(data),
    }
    return f"""Evaluate the participant's TEXT contribution in a Group Discussion.

Evaluate exactly these six parameters:
1. participation
2. communication
3. leadership
4. analytical_skill
5. quantitative_knowledge
6. topic_relevance

Use the transcript as the primary evidence and the scalar text features only as
supporting evidence. Do not use or discuss the embedding vector.

Do not evaluate voice, pitch, energy, pauses, speaking rate, audio quality, or vocal
confidence. More words do not automatically mean better participation. Leadership
must be supported by textual evidence. Analytical skill should reflect reasoning,
examples, comparisons, cause-effect relationships, or conclusions present in the
transcript. A lack of numbers must not automatically receive a zero for quantitative
knowledge. Do not infer qualities not supported by the transcript.

Return one JSON object with the required six keys. Each score must be an integer from
0 to 10. Evidence must quote or closely identify the participant's actual transcript.
Keep evidence and reasons concise. Do not return an overall score; Python calculates it.

Input:
{json.dumps(prompt_input, ensure_ascii=False, indent=2)}
"""


def request_ollama(prompt: str) -> dict:
    payload = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict text-evaluation extractor. "
                        "Return only the complete requested JSON object."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": EVALUATION_SCHEMA,
            "options": {"temperature": 0.1, "num_predict": 1400},
        }
    ).encode("utf-8")
    request = Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=600) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        fail(
            f"Ollama request failed with HTTP {exc.code}. "
            f"Ensure '{OLLAMA_MODEL}' is installed. {detail[:300]}"
        )
    except URLError as exc:
        fail(
            "Ollama is unavailable at http://127.0.0.1:11434. "
            f"Start Ollama and ensure '{OLLAMA_MODEL}' is installed. {exc.reason}"
        )
    except TimeoutError:
        fail("Ollama timed out while evaluating the transcript.")
    except json.JSONDecodeError as exc:
        fail(f"Ollama returned invalid JSON: {exc}")

    message = body.get("message") if isinstance(body, dict) else None
    response_text = message.get("content") if isinstance(message, dict) else None
    if not isinstance(response_text, str) or not response_text.strip():
        fail("Ollama returned no evaluation response.")
    try:
        result = json.loads(response_text)
    except json.JSONDecodeError as exc:
        fail(f"Ollama returned malformed evaluation JSON: {exc}")
    if not isinstance(result, dict):
        fail("Ollama evaluation must be a JSON object.")
    return result


def validate_evaluation(value: dict, participant: str) -> dict:
    normalized = {}
    for parameter in PARAMETERS:
        item = value.get(parameter)
        if not isinstance(item, dict):
            fail(f"Missing evaluation for '{parameter}' ({participant}).")
        score = item.get("score")
        if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 10:
            fail(f"Invalid score for '{parameter}' ({participant}).")
        evidence = item.get("evidence")
        reason = item.get("reason")
        if not isinstance(evidence, str) or not evidence.strip():
            fail(f"Missing evidence for '{parameter}' ({participant}).")
        if not isinstance(reason, str) or not reason.strip():
            fail(f"Missing reason for '{parameter}' ({participant}).")
        normalized[parameter] = {
            "score": score,
            "evidence": evidence.strip(),
            "reason": reason.strip(),
        }
    return normalized


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: uv run python text_evaluation.py text_features.json")
        return 1

    input_path = Path(sys.argv[1]).expanduser().resolve()
    try:
        data = load_features(input_path)
        topic = str(data.get("topic", "")).strip()
        if not topic:
            fail("text_features.json is missing its GD topic.")

        output = {
            "topic": topic,
            "model": OLLAMA_MODEL,
            "speakers": {},
        }
        for participant, participant_data in data["speakers"].items():
            evaluation = validate_evaluation(
                request_ollama(build_prompt(topic, participant, participant_data)),
                participant,
            )
            scores = [evaluation[parameter]["score"] for parameter in PARAMETERS]
            evaluation["text_score"] = round(sum(scores) / len(scores), 3)
            output["speakers"][participant] = {
                "speaker_id": str(participant_data.get("speaker_id", participant)),
                "text_evaluation": evaluation,
            }

        OUTPUT_PATH.write_text(
            json.dumps(output, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Created: {OUTPUT_PATH}")
    print(f"Participants evaluated: {len(output['speakers'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
