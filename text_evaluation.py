from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# ============================================================================
# CONFIGURATION
# ============================================================================

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

VALID_CONFIDENCE = {"high", "medium", "low"}

VALID_STATUS = {
    "Scored",
    "Not Demonstrated",
}


# ============================================================================
# JSON SCHEMA SENT TO OLLAMA
# ============================================================================

PARAMETER_SCHEMA = {
    "type": "object",
    "required": [
        "status",
        "score",
        "confidence",
        "evidence",
        "justification",
    ],
    "properties": {
        "status": {
            "type": "string",
            "enum": [
                "Scored",
                "Not Demonstrated",
            ],
        },
        "score": {
            "anyOf": [
                {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 10,
                },
                {
                    "type": "null",
                },
            ]
        },
        "confidence": {
            "type": "string",
            "enum": [
                "high",
                "medium",
                "low",
            ],
        },
        "evidence": {
            "type": "string",
            "maxLength": 300,
        },
        "justification": {
            "type": "string",
            "maxLength": 500,
        },
    },
}


EVALUATION_SCHEMA = {
    "type": "object",
    "required": list(PARAMETERS),
    "properties": {
        parameter: PARAMETER_SCHEMA
        for parameter in PARAMETERS
    },
}


# ============================================================================
# ERROR HELPER
# ============================================================================

def fail(message: str) -> None:
    raise ValueError(message)


# ============================================================================
# LOAD INPUT JSON
# ============================================================================

def load_features(path: Path) -> dict:

    if not path.exists():
        fail(f"Text features file not found: {path}")

    try:
        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as exc:
        fail(
            f"Invalid JSON in {path}: {exc}"
        )

    if not isinstance(data, dict):
        fail(
            "Input JSON must contain a JSON object."
        )

    if not isinstance(
        data.get("speakers"),
        dict,
    ):
        fail(
            "Input JSON must contain a 'speakers' object."
        )

    if not data["speakers"]:
        fail(
            "Input JSON contains no speakers."
        )

    return data


# ============================================================================
# SELECT USEFUL TEXT FEATURES
# ============================================================================

def selected_features(data: dict) -> dict:

    allowed = {
        # Lexical features
        "word_count",
        "sentence_count",
        "average_sentence_length",
        "max_sentence_length",
        "ttr",
        "lexical_diversity",
        "mtld",

        # Semantic features
        "topic_similarity",
        "topic_coverage",
        "semantic_similarity",

        # Quantitative features
        "numeric_entity_count",
        "quantity_count",
        "percentage_count",
        "date_count",
        "number_of_quantitative_statements",

        # Optional participation-related features
        # These are used only if they already exist.
        "speaking_turns",
        "turn_count",
        "speaking_duration",
    }

    return {
        key: data[key]
        for key in allowed
        if key in data
    }


# ============================================================================
# NORMALIZE TEXT FOR EVIDENCE VALIDATION
# ============================================================================

def normalize_text(text: str) -> str:

    text = text.strip()

    # Remove surrounding quotation marks.
    text = text.strip("\"'")

    # Normalize whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def evidence_exists_in_transcript(
    evidence: str,
    transcript: str,
) -> bool:

    evidence = normalize_text(evidence)
    transcript = normalize_text(transcript)

    if not evidence:
        return False

    if evidence.lower() in transcript.lower():
        return True

    return False


# ============================================================================
# BUILD EVALUATION PROMPT
# ============================================================================

def build_prompt(
    topic: str,
    participant: str,
    data: dict,
) -> str:

    transcript = str(
        data.get(
            "original_text",
            "",
        )
    ).strip()

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
You are evaluating ONE participant's TEXT contribution in a Group Discussion.

Evaluate exactly these six parameters:

1. participation
2. communication
3. leadership
4. analytical_skill
5. quantitative_knowledge
6. topic_relevance


============================================================
CORE EVALUATION RULE
============================================================

The participant transcript is the PRIMARY evidence.

Text features are SUPPORTING evidence only.

Do NOT convert a feature directly into a score.

For example:

topic_similarity = 0.65

does NOT mean:

topic_relevance = 6.5

Instead, interpret the transcript together with the relevant
supporting features.


============================================================
EVIDENCE RULE
============================================================

The "evidence" field MUST be an EXACT contiguous quote copied
from the participant's transcript.

Do NOT write:

"The participant demonstrates good analytical skill."

Do NOT write:

"The participant actively engages..."

Do NOT write:

"number_of_quantitative_statements: 0"

Those are interpretations, not transcript evidence.

The evidence must be an actual sentence or short phrase spoken
by the participant.

Maximum approximately 300 characters.

If there is no suitable direct transcript evidence:

"evidence": "Insufficient direct evidence in transcript."


============================================================
JUSTIFICATION RULE
============================================================

"justification" explains WHY the quoted evidence supports
the score.

Therefore:

evidence = WHAT DID THE PARTICIPANT SAY?

justification = WHY DOES THAT SUPPORT THE SCORE?


============================================================
SCORING SCALE
============================================================

0-2  = Very limited evidence
3-4  = Limited evidence
5-6  = Moderate evidence
7-8  = Strong evidence
9-10 = Very strong evidence

Do not give 9 or 10 without strong and specific evidence.

Do not give very low scores without evidence supporting poor
performance.

When evidence is mixed, prefer a middle score.


============================================================
1. PARTICIPATION
============================================================

Evaluate:

- meaningful contributions
- number/frequency of contributions if available
- continuity of participation
- adding new information
- responding to other participants
- constructive engagement

IMPORTANT:

More words alone do NOT mean better participation.


============================================================
2. COMMUNICATION
============================================================

Evaluate text-based:

- clarity
- organization
- coherence
- sentence structure
- vocabulary
- concise expression
- understandable presentation of ideas

Do NOT evaluate:

- voice
- pitch
- loudness
- speaking speed
- pauses
- vocal confidence
- audio quality


============================================================
3. LEADERSHIP
============================================================

Evaluate observable textual/conversational behavior such as:

- initiating useful discussion
- introducing meaningful directions
- connecting ideas
- responding constructively
- building on others' points
- summarizing
- guiding the discussion
- helping move discussion forward
- constructive disagreement
- consensus-building

IMPORTANT:

Speaking the most does NOT automatically mean leadership.

Do not infer leadership merely from word count.


============================================================
4. ANALYTICAL SKILL
============================================================

Look for:

- logical reasoning
- cause-effect relationships
- comparisons
- advantages/disadvantages
- problem analysis
- examples
- evidence
- assumptions
- implications
- counterarguments
- conclusions
- synthesis of ideas


============================================================
5. QUANTITATIVE KNOWLEDGE
============================================================

Look for:

- correct numbers
- statistics
- percentages
- ratios
- calculations
- numerical comparisons
- quantitative evidence
- numerical reasoning

IMPORTANT:

A lack of numbers does NOT automatically mean score = 0.

If the discussion/participant contribution does not provide
reasonable evidence to demonstrate quantitative knowledge,
return:

"status": "Not Demonstrated"
"score": null

Do NOT convert absence of quantitative evidence into a low
numerical score.


============================================================
6. TOPIC RELEVANCE
============================================================

Evaluate:

- direct relevance to the GD topic
- topic-specific concepts
- relevant examples
- staying focused
- avoiding unrelated discussion

Use semantic text features as supporting evidence only.

The transcript remains PRIMARY.


============================================================
CONFIDENCE
============================================================

Use:

"high"   = strong and clear evidence
"medium" = some evidence but interpretation has uncertainty
"low"    = limited evidence


============================================================
IMPORTANT RESTRICTIONS
============================================================

- Evaluate ONLY this participant.
- Do not compare participants.
- Do not rank participants.
- Do not evaluate audio characteristics.
- Do not invent evidence.
- Do not invent facts.
- Do not use embedding vectors.
- Do not directly map feature values to scores.
- Evidence must come from the transcript.
- Do not calculate an overall/text score.
- Python will calculate the aggregate score.
- Return ONLY JSON.


============================================================
INPUT
============================================================

{json.dumps(
    prompt_input,
    ensure_ascii=False,
    indent=2,
)}
""".strip()


# ============================================================================
# EXTRACT OLLAMA CONTENT
# ============================================================================

def extract_ollama_content(body: dict) -> str:

    if not isinstance(body, dict):
        fail(
            "Ollama returned a response that is not a JSON object."
        )

    message = body.get("message")

    if isinstance(message, dict):

        content = message.get("content")

        if (
            isinstance(content, str)
            and content.strip()
        ):
            return content.strip()

    response = body.get("response")

    if (
        isinstance(response, str)
        and response.strip()
    ):
        return response.strip()

    fail(
        "Ollama returned no evaluation response.\n"
        "Full Ollama response:\n"
        + json.dumps(
            body,
            indent=2,
            ensure_ascii=False,
        )
    )


# ============================================================================
# CLEAN MODEL JSON
# ============================================================================

def clean_json_response(text: str) -> str:

    text = text.strip()

    if text.startswith("```json"):
        text = text[len("```json"):].strip()

    elif text.startswith("```"):
        text = text[len("```"):].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return text


# ============================================================================
# CALL OLLAMA
# ============================================================================

def request_ollama(prompt: str) -> dict:

    payload_dict = {

        "model": OLLAMA_MODEL,

        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict GD text evaluation extractor. "
                    "Return ONLY the requested JSON object. "
                    "Do not output markdown. "
                    "Do not output explanations outside JSON. "
                    "Do not output thinking or reasoning."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],

        "stream": False,

        # Structured JSON schema
        "format": EVALUATION_SCHEMA,

        # Qwen thinking disabled for this extraction task.
        "think": False,

        "options": {
            "temperature": 0.1,
            "num_predict": 2500,
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

        body = json.loads(
            raw_response
        )

    except json.JSONDecodeError as exc:

        fail(
            "Ollama returned invalid HTTP JSON.\n"
            f"JSON error: {exc}\n"
            f"Raw response:\n{raw_response[:2000]}"
        )

    # Check generation limit.
    done_reason = body.get(
        "done_reason"
    )

    if done_reason == "length":

        fail(
            "Ollama reached the generation limit before "
            "producing the complete evaluation JSON."
        )

    response_text = extract_ollama_content(
        body
    )

    cleaned_text = clean_json_response(
        response_text
    )

    try:

        result = json.loads(
            cleaned_text
        )

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


# ============================================================================
# VALIDATE ONE PARAMETER
# ============================================================================

def validate_parameter(
    parameter: str,
    item: dict,
    participant: str,
    transcript: str,
) -> dict:

    if not isinstance(item, dict):

        fail(
            f"Evaluation for '{parameter}' "
            f"({participant}) is not an object."
        )

    status = item.get("status")
    score = item.get("score")
    confidence = item.get("confidence")
    evidence = item.get("evidence")
    justification = item.get("justification")

    # ------------------------------------------------------------
    # Status
    # ------------------------------------------------------------

    if status not in VALID_STATUS:

        fail(
            f"Invalid status for '{parameter}' "
            f"({participant}): {status!r}"
        )

    # ------------------------------------------------------------
    # Score
    # ------------------------------------------------------------

    if status == "Scored":

        if (
            isinstance(score, bool)
            or not isinstance(score, int)
            or not 0 <= score <= 10
        ):

            fail(
                f"Invalid score for '{parameter}' "
                f"({participant}). "
                f"Expected integer 0-10, got {score!r}"
            )

    elif status == "Not Demonstrated":

        if score is not None:

            fail(
                f"'{parameter}' for {participant} is "
                f"'Not Demonstrated' but score is not null."
            )

    # ------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------

    if confidence not in VALID_CONFIDENCE:

        fail(
            f"Invalid confidence for '{parameter}' "
            f"({participant}): {confidence!r}"
        )

    # ------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------

    if (
        not isinstance(evidence, str)
        or not evidence.strip()
    ):

        fail(
            f"Missing evidence for '{parameter}' "
            f"({participant})."
        )

    evidence = evidence.strip()

    # Allow the explicit fallback.
    if (
        evidence
        != "Insufficient direct evidence in transcript."
    ):

        if not evidence_exists_in_transcript(
            evidence,
            transcript,
        ):

            fail(
                f"Evidence validation failed for "
                f"'{parameter}' ({participant}).\n"
                f"The evidence is not found as a direct "
                f"quote in the participant transcript.\n"
                f"Evidence: {evidence}"
            )

    # ------------------------------------------------------------
    # Justification
    # ------------------------------------------------------------

    if (
        not isinstance(justification, str)
        or not justification.strip()
    ):

        fail(
            f"Missing justification for '{parameter}' "
            f"({participant})."
        )

    return {
        "status": status,
        "score": score,
        "confidence": confidence,
        "evidence": evidence,
        "justification": justification.strip(),
    }


# ============================================================================
# VALIDATE COMPLETE EVALUATION
# ============================================================================

def validate_evaluation(
    value: dict,
    participant: str,
    transcript: str,
) -> dict:

    if not isinstance(value, dict):

        fail(
            f"Evaluation for '{participant}' "
            "is not a JSON object."
        )

    normalized = {}

    for parameter in PARAMETERS:

        item = value.get(parameter)

        if item is None:

            fail(
                f"Missing evaluation for "
                f"'{parameter}' ({participant})."
            )

        normalized[parameter] = validate_parameter(
            parameter,
            item,
            participant,
            transcript,
        )

    return normalized


# ============================================================================
# CALCULATE TEXT SCORE
# ============================================================================

def calculate_text_score(
    evaluation: dict,
) -> tuple[float | None, int]:

    valid_scores = []

    for parameter in PARAMETERS:

        score = evaluation[
            parameter
        ]["score"]

        if score is not None:
            valid_scores.append(score)

    if not valid_scores:
        return None, 0

    text_score = round(
        sum(valid_scores)
        / len(valid_scores),
        3,
    )

    return (
        text_score,
        len(valid_scores),
    )


# ============================================================================
# EVALUATE ONE PARTICIPANT
# ============================================================================

def evaluate_participant(
    topic: str,
    participant: str,
    participant_data: dict,
) -> dict:

    print(
        f"Evaluating participant: {participant}"
    )

    transcript = str(
        participant_data.get(
            "original_text",
            "",
        )
    ).strip()

    if not transcript:

        fail(
            f"Participant '{participant}' "
            "has no original transcript."
        )

    prompt = build_prompt(
        topic,
        participant,
        participant_data,
    )

    raw_evaluation = request_ollama(
        prompt
    )

    evaluation = validate_evaluation(
        raw_evaluation,
        participant,
        transcript,
    )

    # ------------------------------------------------------------
    # Python calculates aggregate score.
    # LLM does NOT calculate it.
    # ------------------------------------------------------------

    text_score, scored_parameter_count = (
        calculate_text_score(
            evaluation
        )
    )

    evaluation["text_score"] = text_score

    evaluation["scored_parameter_count"] = (
        scored_parameter_count
    )

    print(
        f"  Text score: {text_score}"
    )

    print(
        f"  Parameters scored: "
        f"{scored_parameter_count}/6"
    )

    return evaluation


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "  uv run python text_evaluation.py "
            "text_features.json"
        )

        return 1

    input_path = (
        Path(sys.argv[1])
        .expanduser()
        .resolve()
    )

    try:

        # ------------------------------------------------------------
        # Load input
        # ------------------------------------------------------------

        data = load_features(
            input_path
        )

        topic = str(
            data.get(
                "topic",
                "",
            )
        ).strip()

        if not topic:

            fail(
                "Input JSON is missing its GD topic."
            )

        # ------------------------------------------------------------
        # Prepare output
        # ------------------------------------------------------------

        output = {
            "topic": topic,
            "model": OLLAMA_MODEL,
            "evaluation_type": "text_only",
            "speakers": {},
        }

        speakers = data[
            "speakers"
        ]

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

        # ------------------------------------------------------------
        # Evaluate every participant
        # ------------------------------------------------------------

        for participant, participant_data in (
            speakers.items()
        ):

            if not isinstance(
                participant_data,
                dict,
            ):

                fail(
                    f"Speaker '{participant}' "
                    "data must be an object."
                )

            evaluation = evaluate_participant(
                topic,
                participant,
                participant_data,
            )

            output["speakers"][
                participant
            ] = {
                "speaker_id": str(
                    participant_data.get(
                        "speaker_id",
                        participant,
                    )
                ),
                "text_evaluation": evaluation,
            }

        # ------------------------------------------------------------
        # Write output
        # ------------------------------------------------------------

        OUTPUT_PATH.write_text(
            json.dumps(
                output,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    except (
        OSError,
        ValueError,
    ) as exc:

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


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    raise SystemExit(
        main()
    )