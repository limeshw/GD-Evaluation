from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "qwen3.5:9b"

OUTPUT_PATH = Path("text_evaluation.json")

PARAMETERS = (
    "participation",
    "communication",
    "leadership",
    "analytical_skill",
    "quantitative_knowledge",
    "topic_relevance",
)


# ---------------------------------------------------------------------------
# JSON schema sent to Ollama
# ---------------------------------------------------------------------------

EVALUATION_SCHEMA = {
    "type": "object",
    "required": list(PARAMETERS),
    "properties": {
        parameter: {
            "type": "object",
            "required": ["score", "evidence", "reason"],
            "properties": {
                "score": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 10,
                },
                "evidence": {
                    "type": "string",
                    "maxLength": 500,
                },
                "reason": {
                    "type": "string",
                    "maxLength": 500,
                },
            },
        }
        for parameter in PARAMETERS
    },
}


# ---------------------------------------------------------------------------
# Error helper
# ---------------------------------------------------------------------------

def fail(message: str) -> None:
    raise ValueError(message)


# ---------------------------------------------------------------------------
# Load input JSON
# ---------------------------------------------------------------------------

def load_features(path: Path) -> dict:
    if not path.exists():
        fail(f"Text features file not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"Invalid JSON in {path}: {exc}")

    if not isinstance(data, dict):
        fail("text_features.json must contain a JSON object.")

    if not isinstance(data.get("speakers"), dict):
        fail("text_features.json must contain a 'speakers' object.")

    if not data["speakers"]:
        fail("text_features.json contains no speakers.")

    return data


# ---------------------------------------------------------------------------
# Select only useful text features
# ---------------------------------------------------------------------------

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

    return {
        key: data[key]
        for key in allowed
        if key in data
    }


# ---------------------------------------------------------------------------
# Build evaluation prompt
# ---------------------------------------------------------------------------

def build_prompt(
    topic: str,
    participant: str,
    data: dict,
) -> str:

    transcript = str(data.get("original_text", "")).strip()

    if not transcript:
        fail(
            f"Speaker '{participant}' has no original transcript text."
        )

    prompt_input = {
        "gd_topic": topic,
        "participant": participant,
        "transcript": transcript,
        "text_features": selected_features(data),
    }

    return f"""
Evaluate the participant's TEXT contribution in a Group Discussion.

Evaluate exactly these six parameters:

1. participation
2. communication
3. leadership
4. analytical_skill
5. quantitative_knowledge
6. topic_relevance

IMPORTANT RULES:

- Use the transcript as the primary evidence.
- Use scalar text features only as supporting evidence.
- Do not use or discuss any embedding vector.
- Do not evaluate voice, pitch, energy, pauses, speaking rate,
  audio quality, or vocal confidence.
- More words do not automatically mean better participation.
- Leadership must be supported by textual evidence.
- Analytical skill should reflect reasoning, examples, comparisons,
  cause-effect relationships, or conclusions present in the transcript.
- A lack of numbers must NOT automatically result in zero for
  quantitative knowledge.
- Do not infer qualities that are not supported by the transcript.
- Each score must be an integer from 0 to 10.
- Evidence must quote or closely identify the participant's actual
  transcript.
- Keep evidence concise.
- Keep reasons concise.
- Do not calculate or return an overall score.
- Python will calculate the overall text score.

Return ONLY the requested JSON object.

Input:
{json.dumps(prompt_input, ensure_ascii=False, indent=2)}
""".strip()


# ---------------------------------------------------------------------------
# Extract response content from Ollama
# ---------------------------------------------------------------------------

def extract_ollama_content(body: dict) -> str:
    """
    Ollama /api/chat normally returns:

    {
        "message": {
            "role": "assistant",
            "content": "..."
        },
        ...
    }

    This function also provides useful diagnostics if the response
    structure is different.
    """

    if not isinstance(body, dict):
        fail(
            "Ollama returned a response that is not a JSON object."
        )

    message = body.get("message")

    if isinstance(message, dict):
        content = message.get("content")

        if isinstance(content, str) and content.strip():
            return content.strip()

    # Some Ollama/model combinations may return response-oriented data.
    response = body.get("response")

    if isinstance(response, str) and response.strip():
        return response.strip()

    # If neither field exists, show the actual response.
    fail(
        "Ollama returned no evaluation response.\n"
        "Full Ollama response:\n"
        + json.dumps(
            body,
            indent=2,
            ensure_ascii=False,
        )
    )


# ---------------------------------------------------------------------------
# Clean JSON returned by model
# ---------------------------------------------------------------------------

def clean_json_response(text: str) -> str:
    text = text.strip()

    # Remove markdown code fences if the model added them.
    if text.startswith("```json"):
        text = text[len("```json"):].strip()

    elif text.startswith("```"):
        text = text[len("```"):].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return text


# ---------------------------------------------------------------------------
# Call Ollama
# ---------------------------------------------------------------------------

def request_ollama(prompt: str) -> dict:

    payload_dict = {
        "model": OLLAMA_MODEL,

        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict text-evaluation extractor. "
                    "Return ONLY the requested JSON object. "
                    "Do not explain your reasoning. "
                    "Do not output markdown. "
                    "Do not output a thinking process."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],

        "stream": False,

        # Structured JSON output
        "format": EVALUATION_SCHEMA,

        # Important for Qwen thinking models:
        # we do not need reasoning for this extraction task.
        "think": False,

        "options": {
            "temperature": 0.1,

            # 1400 was too small because the model was using
            # the generation budget for thinking.
            "num_predict": 2000,
        },
    }

    payload = json.dumps(
        payload_dict,
        ensure_ascii=False,
    ).encode("utf-8")

    request = Request(
        OLLAMA_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    print(
        f"\nCalling Ollama model '{OLLAMA_MODEL}'..."
    )

    try:

        with urlopen(
            request,
            timeout=600,
        ) as response:

            raw_response = response.read().decode(
                "utf-8",
                errors="replace",
            )

    except HTTPError as exc:

        detail = exc.read().decode(
            "utf-8",
            errors="replace",
        )

        fail(
            f"Ollama request failed with HTTP {exc.code}.\n"
            f"Model: {OLLAMA_MODEL}\n"
            f"URL: {OLLAMA_URL}\n"
            f"Response: {detail[:1000]}"
        )

    except URLError as exc:

        fail(
            "Ollama is unavailable.\n"
            f"URL: {OLLAMA_URL}\n"
            f"Model: {OLLAMA_MODEL}\n"
            f"Original error: {exc.reason}"
        )

    except TimeoutError:

        fail(
            "Ollama timed out while evaluating the transcript."
        )

    except OSError as exc:

        fail(
            f"Could not connect to Ollama: {exc}"
        )

    if not raw_response.strip():

        fail(
            "Ollama returned an empty HTTP response."
        )

    try:

        body = json.loads(raw_response)

    except json.JSONDecodeError as exc:

        fail(
            "Ollama returned invalid HTTP JSON.\n"
            f"JSON error: {exc}\n"
            f"Raw response:\n{raw_response[:2000]}"
        )

    # ---------------------------------------------------------------
    # Check whether Ollama stopped because of token limit
    # ---------------------------------------------------------------

    done_reason = body.get("done_reason")

    if done_reason == "length":

        message = body.get("message", {})

        thinking = ""

        if isinstance(message, dict):
            thinking = message.get("thinking", "") or ""

        fail(
            "Ollama reached the generation limit before producing "
            "the evaluation JSON.\n"
            f"done_reason: {done_reason}\n"
            f"Thinking tokens/content were generated: "
            f"{len(thinking)} characters"
        )

    # ---------------------------------------------------------------
    # Extract content
    # ---------------------------------------------------------------

    message = body.get("message")

    response_text = None

    if isinstance(message, dict):

        content = message.get("content")

        if isinstance(content, str) and content.strip():
            response_text = content.strip()

    # Fallback for /api/generate-style responses
    if not response_text:

        response = body.get("response")

        if isinstance(response, str) and response.strip():
            response_text = response.strip()

    if not response_text:

        fail(
            "Ollama returned no evaluation response.\n"
            "Full Ollama response:\n"
            + json.dumps(
                body,
                indent=2,
                ensure_ascii=False,
            )
        )

    # ---------------------------------------------------------------
    # Clean markdown fences if necessary
    # ---------------------------------------------------------------

    cleaned_text = response_text.strip()

    if cleaned_text.startswith("```json"):

        cleaned_text = cleaned_text[
            len("```json"):
        ].strip()

    elif cleaned_text.startswith("```"):

        cleaned_text = cleaned_text[
            len("```"):
        ].strip()

    if cleaned_text.endswith("```"):

        cleaned_text = cleaned_text[:-3].strip()

    # ---------------------------------------------------------------
    # Parse JSON
    # ---------------------------------------------------------------

    try:

        result = json.loads(cleaned_text)

    except json.JSONDecodeError as exc:

        fail(
            "Ollama returned malformed evaluation JSON.\n"
            f"JSON error: {exc}\n\n"
            f"Model response:\n{cleaned_text}"
        )

    if not isinstance(result, dict):

        fail(
            "Ollama evaluation must be a JSON object."
        )

    return result

    payload_dict = {
        "model": OLLAMA_MODEL,

        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict text-evaluation extractor. "
                    "Return only the complete requested JSON object. "
                    "Do not use markdown. "
                    "Do not add explanations outside the JSON."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],

        # We need one complete JSON response rather than streaming chunks.
        "stream": False,

        # Use the actual JSON schema instead of generic JSON mode.
        "format": EVALUATION_SCHEMA,

        "options": {
            "temperature": 0.1,
            "num_predict": 1400,
        },
    }

    payload = json.dumps(
        payload_dict,
        ensure_ascii=False,
    ).encode("utf-8")

    request = Request(
        OLLAMA_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    print(
        f"\nCalling Ollama model '{OLLAMA_MODEL}'..."
    )

    try:

        with urlopen(
            request,
            timeout=600,
        ) as response:

            raw_response = response.read().decode(
                "utf-8",
                errors="replace",
            )

    except HTTPError as exc:

        detail = exc.read().decode(
            "utf-8",
            errors="replace",
        )

        fail(
            f"Ollama request failed with HTTP {exc.code}.\n"
            f"Model: {OLLAMA_MODEL}\n"
            f"URL: {OLLAMA_URL}\n"
            f"Response: {detail[:1000]}"
        )

    except URLError as exc:

        fail(
            "Ollama is unavailable.\n"
            f"URL: {OLLAMA_URL}\n"
            f"Model: {OLLAMA_MODEL}\n\n"
            "Make sure Ollama is running.\n"
            f"Original error: {exc.reason}"
        )

    except TimeoutError:

        fail(
            "Ollama timed out while evaluating the transcript."
        )

    except OSError as exc:

        fail(
            f"Could not connect to Ollama: {exc}"
        )

    # -----------------------------------------------------------------------
    # Parse HTTP response
    # -----------------------------------------------------------------------

    if not raw_response.strip():

        fail(
            "Ollama returned an empty HTTP response."
        )

    try:

        body = json.loads(raw_response)

    except json.JSONDecodeError as exc:

        fail(
            "Ollama returned invalid HTTP JSON.\n"
            f"JSON error: {exc}\n"
            f"Raw response:\n{raw_response[:2000]}"
        )

    # -----------------------------------------------------------------------
    # Extract model response
    # -----------------------------------------------------------------------

    response_text = extract_ollama_content(body)

    if not response_text:

        fail(
            "Ollama returned empty model content."
        )

    # -----------------------------------------------------------------------
    # Parse model's JSON
    # -----------------------------------------------------------------------

    cleaned_text = clean_json_response(response_text)

    try:

        result = json.loads(cleaned_text)

    except json.JSONDecodeError as exc:

        fail(
            "Ollama returned malformed evaluation JSON.\n"
            f"JSON error: {exc}\n\n"
            f"Model response:\n{cleaned_text}"
        )

    if not isinstance(result, dict):

        fail(
            "Ollama evaluation must be a JSON object."
        )

    return result


# ---------------------------------------------------------------------------
# Validate model evaluation
# ---------------------------------------------------------------------------

def validate_evaluation(
    value: dict,
    participant: str,
) -> dict:

    if not isinstance(value, dict):

        fail(
            f"Evaluation for '{participant}' is not a JSON object."
        )

    normalized = {}

    for parameter in PARAMETERS:

        item = value.get(parameter)

        if not isinstance(item, dict):

            fail(
                f"Missing evaluation for "
                f"'{parameter}' ({participant})."
            )

        score = item.get("score")

        if (
            isinstance(score, bool)
            or not isinstance(score, int)
            or not 0 <= score <= 10
        ):

            fail(
                f"Invalid score for "
                f"'{parameter}' ({participant}). "
                f"Expected integer 0-10, got: {score!r}"
            )

        evidence = item.get("evidence")
        reason = item.get("reason")

        if not isinstance(evidence, str) or not evidence.strip():

            fail(
                f"Missing evidence for "
                f"'{parameter}' ({participant})."
            )

        if not isinstance(reason, str) or not reason.strip():

            fail(
                f"Missing reason for "
                f"'{parameter}' ({participant})."
            )

        normalized[parameter] = {
            "score": score,
            "evidence": evidence.strip(),
            "reason": reason.strip(),
        }

    return normalized


# ---------------------------------------------------------------------------
# Evaluate one participant
# ---------------------------------------------------------------------------

def evaluate_participant(
    topic: str,
    participant: str,
    participant_data: dict,
) -> dict:

    print(
        f"Evaluating participant: {participant}"
    )

    prompt = build_prompt(
        topic,
        participant,
        participant_data,
    )

    raw_evaluation = request_ollama(prompt)

    evaluation = validate_evaluation(
        raw_evaluation,
        participant,
    )

    scores = [
        evaluation[parameter]["score"]
        for parameter in PARAMETERS
    ]

    text_score = round(
        sum(scores) / len(scores),
        3,
    )

    evaluation["text_score"] = text_score

    print(
        f"  Text score: {text_score}"
    )

    return evaluation


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "  uv run python text_evaluation.py text_features.json"
        )

        return 1

    input_path = (
        Path(sys.argv[1])
        .expanduser()
        .resolve()
    )

    try:

        # ---------------------------------------------------------------
        # Load input
        # ---------------------------------------------------------------

        data = load_features(
            input_path
        )

        topic = str(
            data.get("topic", "")
        ).strip()

        if not topic:

            fail(
                "text_features.json is missing its GD topic."
            )

        # ---------------------------------------------------------------
        # Prepare output
        # ---------------------------------------------------------------

        output = {
            "topic": topic,
            "model": OLLAMA_MODEL,
            "speakers": {},
        }

        speakers = data["speakers"]

        print(
            f"Topic: {topic}"
        )

        print(
            f"Participants: {len(speakers)}"
        )

        print(
            f"Ollama: {OLLAMA_URL}"
        )

        print(
            f"Model: {OLLAMA_MODEL}"
        )

        # ---------------------------------------------------------------
        # Evaluate every participant
        # ---------------------------------------------------------------

        for participant, participant_data in speakers.items():

            if not isinstance(participant_data, dict):

                fail(
                    f"Speaker '{participant}' data must be an object."
                )

            evaluation = evaluate_participant(
                topic,
                participant,
                participant_data,
            )

            output["speakers"][participant] = {
                "speaker_id": str(
                    participant_data.get(
                        "speaker_id",
                        participant,
                    )
                ),
                "text_evaluation": evaluation,
            }

        # ---------------------------------------------------------------
        # Write output
        # ---------------------------------------------------------------

        OUTPUT_PATH.write_text(
            json.dumps(
                output,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    except (OSError, ValueError) as exc:

        print(
            f"\nError: {exc}",
            file=sys.stderr,
        )

        return 1

    print(
        f"\nCreated: {OUTPUT_PATH}"
    )

    print(
        f"Participants evaluated: "
        f"{len(output['speakers'])}"
    )

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(main())
