from __future__ import annotations

import json
import re
import sys
from pathlib import Path


OUTPUT_PATH = Path("text_features.json")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?%?\b")


def fail(message: str) -> None:
    raise ValueError(message)


def load_speaker_wise(path: Path) -> dict:
    if not path.exists():
        fail(f"Speaker-wise transcript not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"Invalid JSON in {path}: {exc}")
    speakers = data.get("speakers") if isinstance(data, dict) else None
    if not isinstance(speakers, dict) or not speakers:
        fail("Input must contain a non-empty 'speakers' object.")
    return data


def load_nlp():
    try:
        import spacy
    except ImportError:
        fail("Missing spaCy. Run 'uv sync'.")

    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        fail(
            "spaCy model 'en_core_web_sm' is missing. Install it with: "
            "uv run python -m spacy download en_core_web_sm"
        )


def tokenize_and_clean(doc) -> tuple[str, list[str], list[str]]:
    tokens = [
        (token.lemma_ or token.text).lower()
        for token in doc
        if token.is_alpha and not token.is_stop
    ]
    sentences = [sentence.text.strip() for sentence in doc.sents if sentence.text.strip()]
    return " ".join(tokens), tokens, sentences


def mtld_value(clean_text: str, lexical_richness) -> float:
    if not clean_text:
        return 0.0
    try:
        value = lexical_richness(clean_text).mtld(threshold=0.72)
        return round(float(value), 6)
    except (AttributeError, ValueError, ZeroDivisionError):
        return 0.0


def numeric_features(doc, sentences: list[str]) -> dict:
    labels = {"CARDINAL", "NUMBER", "PERCENT", "MONEY", "DATE", "QUANTITY"}
    entities = [entity for entity in doc.ents if entity.label_ in labels]
    entity_by_sentence = 0
    for sentence in sentences:
        if NUMBER_PATTERN.search(sentence) or any(
            entity.text in sentence for entity in entities
        ):
            entity_by_sentence += 1

    return {
        "numeric_entity_count": len(entities),
        "quantity_count": sum(entity.label_ == "QUANTITY" for entity in entities),
        "percentage_count": sum(entity.label_ == "PERCENT" for entity in entities),
        "date_count": sum(entity.label_ == "DATE" for entity in entities),
        "number_of_quantitative_statements": entity_by_sentence,
    }


def build_text_features(speaker_wise: dict, nlp, embedding_model) -> tuple[list[str], list[dict]]:
    try:
        from lexicalrichness import LexicalRichness
    except ImportError:
        fail("Missing lexicalrichness. Run 'uv sync'.")

    speaker_names = list(speaker_wise["speakers"])
    prepared = []
    for speaker in speaker_names:
        data = speaker_wise["speakers"][speaker]
        transcript = data.get("transcript", {})
        original_text = str(transcript.get("full_text", data.get("full_text", ""))).strip()
        if not original_text:
            fail(f"Speaker '{speaker}' has empty transcript text.")

        doc = nlp(original_text)
        clean_text, tokens, sentences = tokenize_and_clean(doc)
        total_words = len(tokens)
        unique_words = len(set(tokens))
        sentence_count = len(sentences)
        ttr = unique_words / total_words if total_words else 0.0
        prepared.append(
            {
                "speaker_id": str(data.get("speaker_id", speaker)),
                "name": str(data.get("name", speaker)),
                "original_text": original_text,
                "clean_text": clean_text,
                "tokens": tokens,
                "sentences": sentences,
                "word_count": total_words,
                "sentence_count": sentence_count,
                "average_sentence_length": total_words / sentence_count
                if sentence_count
                else 0.0,
                "max_sentence_length": max(
                    (len([word for word in nlp(sentence) if word.is_alpha]) for sentence in sentences),
                    default=0,
                ),
                "unique_word_count": unique_words,
                "ttr": ttr,
                "lexical_diversity": ttr,
                "mtld": mtld_value(clean_text, LexicalRichness),
                **numeric_features(doc, sentences),
            }
        )

    texts = [item["original_text"] for item in prepared]
    embeddings = embedding_model.encode(texts, normalize_embeddings=False)
    for item, embedding in zip(prepared, embeddings):
        item["embedding"] = [round(float(value), 8) for value in embedding]
    return speaker_names, prepared


def add_semantic_features(items: list[dict], topic: str, embedding_model, nlp) -> None:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    topic_embedding = embedding_model.encode([topic], normalize_embeddings=False)
    transcript_embeddings = np.asarray([item["embedding"] for item in items])
    topic_scores = cosine_similarity(transcript_embeddings, topic_embedding).ravel()
    topic_doc = nlp(topic)
    topic_tokens = {
        (token.lemma_ or token.text).lower()
        for token in topic_doc
        if token.is_alpha and not token.is_stop
    }

    for item, topic_score in zip(items, topic_scores):
        sentence_embeddings = embedding_model.encode(
            item["sentences"], normalize_embeddings=False
        )
        if len(sentence_embeddings) > 1:
            adjacent = cosine_similarity(sentence_embeddings[:-1], sentence_embeddings[1:]).diagonal()
            mean_similarity = float(np.mean(adjacent))
            std_similarity = float(np.std(adjacent))
        else:
            mean_similarity = 0.0
            std_similarity = 0.0
        item["topic_similarity"] = round(float(topic_score), 6)
        item["topic_coverage"] = round(
            len(topic_tokens.intersection(item["tokens"])) / len(topic_tokens), 6
        ) if topic_tokens else 0.0
        item["semantic_similarity"] = round(mean_similarity, 6)
        item["coherence"] = {
            "mean_adjacent_similarity": round(mean_similarity, 6),
            "similarity_std": round(std_similarity, 6),
        }


def add_normalized_vectors(items: list[dict]) -> list[str]:
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    scalar_names = [
        "word_count",
        "sentence_count",
        "average_sentence_length",
        "max_sentence_length",
        "ttr",
        "mtld",
        "topic_similarity",
        "topic_coverage",
        "semantic_similarity",
        "numeric_entity_count",
        "quantity_count",
        "percentage_count",
        "date_count",
        "number_of_quantitative_statements",
    ]
    scalar_matrix = np.asarray([[item[name] for name in scalar_names] for item in items])
    embedding_matrix = np.asarray([item["embedding"] for item in items])
    normalized_scalars = StandardScaler().fit_transform(scalar_matrix)
    normalized_embeddings = StandardScaler().fit_transform(embedding_matrix)

    for item, scalar_row, embedding_row in zip(
        items, normalized_scalars, normalized_embeddings
    ):
        item["normalized_features"] = {
            name: round(float(value), 6)
            for name, value in zip(scalar_names, scalar_row)
        }
        item["text_feature_vector"] = [
            round(float(value), 6)
            for value in np.concatenate([scalar_row, embedding_row])
        ]
    return scalar_names


def main() -> int:
    if len(sys.argv) != 3:
        print(  
            "Usage: uv run python text_feature_extraction.py "
            "speaker_wise_transcript.json \"GD topic\""
        )
        return 1

    input_path = Path(sys.argv[1]).expanduser().resolve()
    topic = sys.argv[2].strip()
    if not topic:
        print("Error: GD topic cannot be empty.", file=sys.stderr)
        return 1

    try:
        speaker_wise = load_speaker_wise(input_path)
        nlp = load_nlp()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            fail("Missing sentence-transformers. Run 'uv sync'.")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        speaker_names, items = build_text_features(speaker_wise, nlp, embedding_model)
        add_semantic_features(items, topic, embedding_model, nlp)
        scalar_names = add_normalized_vectors(items)

        output = {
            "topic": topic,
            "embedding_model": EMBEDDING_MODEL,
            "scalar_feature_names": scalar_names,
            "speakers": {
                speaker: item for speaker, item in zip(speaker_names, items)
            },
        }
        OUTPUT_PATH.write_text(
            json.dumps(output, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Created: {OUTPUT_PATH}")
    print(f"Speakers processed: {len(items)}")
    print(f"Text feature vector length: {len(items[0]['text_feature_vector'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
