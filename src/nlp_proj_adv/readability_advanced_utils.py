"""Reusable features, representations and inference for Project 20.

Sentence-level semantic similarity is a proxy for semantic continuity,
not a validated measure of discourse coherence.
"""
from __future__ import annotations

import re
from statistics import mean, pstdev

import numpy as np
import pandas as pd
from scipy import sparse
import textstat

WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "for", "nor", "so", "yet", "to", "of",
    "in", "on", "at", "by", "with", "from", "as", "is", "are", "was", "were",
    "be", "been", "being", "it", "its", "this", "that", "these", "those",
    "he", "she", "they", "we", "i", "you", "me", "him", "her", "them", "us",
    "their", "our", "your", "my", "his", "who", "which", "what", "when", "where",
    "have", "has", "had", "do", "does", "did", "not", "if", "than", "then",
    "can", "could", "would", "should", "will", "may", "might", "must", "up",
    "out", "into", "over", "under", "about", "after", "before", "between",
}
CONNECTORS = {
    "however", "therefore", "because", "although", "moreover", "furthermore",
    "consequently", "meanwhile", "finally", "nevertheless", "additionally",
    "thus", "instead", "similarly", "firstly", "secondly", "nonetheless",
}
FEATURE_NAMES = [
    "word_count", "sentence_count", "avg_sentence_length", "sentence_length_std",
    "avg_word_length", "long_word_ratio", "polysyllabic_ratio",
    "syllables_per_word", "lexical_diversity", "connector_ratio",
    "lexical_cohesion", "flesch_reading_ease", "flesch_kincaid_grade",
]
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def tokenize_words(text: str) -> list[str]:
    return WORD_RE.findall(str(text).lower())


def split_sentences(text: str) -> list[str]:
    """Normalize arbitrary line wraps; split after ., ?, ! followed by whitespace.

    This is a heuristic and can split abbreviations (Dr., Mr., etc.).
    """
    clean = re.sub(r"\s+", " ", str(text)).strip()
    if not clean:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if s.strip()]


def lexical_cohesion(sentences: list[str]) -> float:
    """Mean adjacent-sentence Jaccard overlap of content word TYPES."""
    if len(sentences) < 2:
        return 0.0
    sets = [
        {w for w in tokenize_words(s) if len(w) > 2 and w not in STOPWORDS}
        for s in sentences
    ]   # this says put in sets the words of each sentence of the passage where the length of the word is longer than 2 and don't belong to stopwords collection
    values = []
    for left, right in zip(sets, sets[1:]): # left takes values from the original sets, while right takes values from sets[1:]. lets say sets = [{"cat", "eats"}, {"cat", "sleeps"}, {"dog", "sleeps"}],so in first iteration left will have {"cat", "eats"} and right will have {"cat", "sleeps"} and it keeps going like that
        union = left | right
        values.append(len(left & right) / len(union) if union else 0.0)  # It adds the similarity score of each consecutive sentence pair to the values list.
    return float(mean(values))  # calculates the average similarity across all consecutive sentence pairs


def extract_features(text: str) -> dict[str, float]:
    text = "" if text is None else str(text)
    words = tokenize_words(text)
    sentences = split_sentences(text)
    if not words:
        return {name: 0.0 for name in FEATURE_NAMES}
    n_words = len(words)
    n_sentences = max(1, len(sentences))
    lengths = [len(tokenize_words(s)) for s in sentences] # number of words in each sentence of the passage so it represents the length of each sentence in a passage.
    values = {
        "word_count": float(n_words),
        "sentence_count": float(len(sentences)),
        "avg_sentence_length": float(n_words / n_sentences),
        "sentence_length_std": float(pstdev(lengths)) if len(lengths) > 1 else 0.0,  # sentence_length_std is a linguistic feature that helps the model understand the variation in sentence structure. A higher value means sentence lengths vary more, but it doesn't automatically mean the text is harder to read.
        "avg_word_length": float(sum(map(len, words)) / n_words),  # so the avg_word_length divides the sum of each word's length of the passsage on the number of words in that passage, A passage containing many long words may be more difficult to read than one containing mostly short words.
        "long_word_ratio": float(sum(len(w) > 6 for w in words) / n_words), # calcule la proportion of words containing more than 6 characters. this is useful because a passage that contains long words is harder to read compared to a passage that contains short words.for example words = ["I", "understand", "this", "exercise"] I = 1 character understand 10 character this 4 character exercise 8 characters so sum(false true false true) = 2 will be divided by 4 (the number of words in that passage) 2/4 = 0.5 which is 50% of the words in this passage are long 
        "polysyllabic_ratio": float(textstat.polysyllabcount(text) / n_words), # polysyllabcount function Processes the passage to identify its words. Estimates the number of syllables in each word. Counts words containing 3 or more syllables. and that will be divided by the number of words in the passage to get the purcentage of words that are pollysillabic.
        "syllables_per_word": float(textstat.syllable_count(text) / n_words), # calculates the average number of syllables per word in a passage.
        "lexical_diversity": float(len(set(words)) / n_words),  # Measures the proportion of unique words in the passage. A higher value indicates less word repetition. set(words) eliminates the second occurence of the same word for example : words = ["the", "cat", "and", "the", "dog"] set(words) va donner  {"the", "cat", "and", "dog"}
        "connector_ratio": float(sum(w in CONNECTORS for w in words) / n_words), # Calculates the proportion of words that belong to the predefined CONNECTORS collection.
        "lexical_cohesion": lexical_cohesion(sentences),  # A higher lexical_cohesion score means neighboring sentences share more vocabulary. A score of 0 means they share no words after filtering.
        "flesch_reading_ease": float(textstat.flesch_reading_ease(text)),
        "flesch_kincaid_grade": float(textstat.flesch_kincaid_grade(text)),
    }
    return {key: values[key] for key in FEATURE_NAMES}


def features_frame(texts) -> pd.DataFrame:
    return pd.DataFrame([extract_features(text) for text in texts], columns=FEATURE_NAMES)


def chunk_document(text: str, words_per_chunk: int = 90) -> list[str]:
    """Create short text chunks to reduce transformer truncation"""
    words = str(text).split()
    return [" ".join(words[i:i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)] or [""]


#this function takes all our passages, divides them into smaller chunks, generates an embedding for each chunk using MiniLM,
#then combines the chunk embeddings to create one 384-dimensional embedding array per passage.
def document_embeddings(texts, encoder, batch_size: int = 32) -> np.ndarray:
    """Encode short chunks, then average + unit-normalize to get document vectors.

    `encoder` is a pretrained, frozen SentenceTransformer; no labels are used.
    """
    chunks, spans = [], []
    for text in texts:
        beginning = len(chunks)
        chunks.extend(chunk_document(text))
        spans.append((beginning, len(chunks)))
    vectors = np.asarray(
        encoder.encode(chunks, batch_size=batch_size, show_progress_bar=True,
                       normalize_embeddings=True), dtype=np.float32
    )  # encoder.encode() is the function that generates the embeddings using our pretrained MiniLM model.
    result = np.stack([vectors[a:b].mean(axis=0) for a, b in spans])
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    return (result / np.maximum(norms, 1e-12)).astype(np.float32)


def semantic_cohesion(text: str, encoder) -> dict:
    """Neighboring sentence cosine similarities; proxy, NOT a coherence grade."""
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return {"mean": None, "pairs": [], "sentences": sentences}
    vectors = np.asarray(encoder.encode(sentences, normalize_embeddings=True), dtype=float)   
    similarities = np.sum(vectors[:-1] * vectors[1:], axis=1)
    return {
        "mean": float(similarities.mean()),
        "pairs": [float(v) for v in similarities],
        "sentences": sentences,
    }


# this code returns one big numerical matrix containing all four types of features for every passage.
def transform_texts(texts, representation: str, word_vectorizer, char_vectorizer,
                    feature_scaler, encoder=None, precomputed_features=None,
                    precomputed_embeddings=None):
    """Build exactly the feature matrix used to fit the selected regression model.

    `word_vectorizer`, `char_vectorizer`, and `feature_scaler` must ALREADY be fitted.
    """
    texts = list(texts)
    manual = (
        precomputed_features if precomputed_features is not None else features_frame(texts)
    )
    manual = feature_scaler.transform(manual).astype(np.float32)
    if representation == "Linguistic (Ridge)":
        return manual

    words = word_vectorizer.transform(texts)
    if representation == "TF-IDF words (Ridge)":
        return words
    chars = char_vectorizer.transform(texts)
    if representation == "TF-IDF + linguistic (Ridge)":
        return sparse.hstack([words, chars, sparse.csr_matrix(manual)], format="csr")

    if precomputed_embeddings is not None:
        embeddings = precomputed_embeddings
    elif encoder is not None:
        embeddings = document_embeddings(texts, encoder)
    else:
        raise ValueError("This representation requires a SentenceTransformer model or embeddings")

    if representation == "Embeddings + linguistic (Ridge)":
        return np.hstack([np.asarray(embeddings), manual])
    if representation == "Hybrid all (Ridge)":
        return sparse.hstack([
            words, chars, sparse.csr_matrix(manual), sparse.csr_matrix(embeddings)
        ], format="csr") # the function sparse.hstack simply combines the already calculated features into one matrix, ready for the model.
    # sparse.csr_matrix(manual) : Converts the numerical array into a CSR sparse matrix. 
    # A sparse matrix efficiently stores numerical data, particularly when many values are zero, as in TF-IDF.
    # Le format CSR stocke principalement les valeurs non nulles ainsi que leurs positions, 
    # au lieu de stocker explicitement tous les zéros, ce qui permet de réduire la consommation de mémoire.
    # la conversion des manual et embeddings en sparse matrice n'était pas fait dans le but de reduction de stockage,
    # par contre Je les ai converties en CSR pour avoir un format cohérent avec words et chars, qui sont déjà des matrices creuses (sparse),
    # pour que je puisse les combiner.
    # format="csr" signifie que la matrice finale retournée par sparse.hstack() doit être au format CSR
    raise ValueError(f"Unknown representation: {representation}")


def readability_level(score: float, low_cutoff: float, high_cutoff: float) -> str:
    if score < low_cutoff:
        return "Difficult (relative to CLEAR)"
    if score < high_cutoff:
        return "Moderate (relative to CLEAR)"
    return "Easier (relative to CLEAR)"


def suggest_improvements(features: dict) -> list[str]:
    messages = []
    if features["avg_sentence_length"] > 22:
        messages.append("Consider splitting long sentences where the meaning permits.")
    if features["long_word_ratio"] > 0.18 or features["polysyllabic_ratio"] > 0.15:
        messages.append("Review technical and long words; use clearer synonyms when accurate.")
    if features["sentence_count"] >= 4 and features["connector_ratio"] == 0:
        messages.append("Where appropriate, clarify transitions between ideas.")
    if features["word_count"] < 80:
        messages.append("Short input: CLEAR passages are typically longer; predictions may be less reliable.")
    if features["word_count"] > 400:
        messages.append("Long input: this passage may differ from CLEAR examples; interpret cautiously.")
    return messages or ["No strong rule-based readability issues were detected."]


def predict_text(text: str, artifact: dict, encoder=None) -> dict:
    matrix = transform_texts(
        [text], representation=artifact["representation"],
        word_vectorizer=artifact["word_vectorizer"],
        char_vectorizer=artifact["char_vectorizer"],
        feature_scaler=artifact["feature_scaler"], encoder=encoder,
    )
    score = float(artifact["model"].predict(matrix)[0])
    return {
        "score": score,
        "level": readability_level(score, artifact["low_cutoff"], artifact["high_cutoff"]),
        "features": extract_features(text),
    }


# --- External evaluation and length-aware inference utilities ---

def chunk_by_sentences(text: str, target_words: int = 180) -> list[str]:
    """Split at sentence boundaries near target_words, preserving original order.

    Falls back to word splits only if an individual sentence is too long.
    This is a heuristic; choose the chunk size from CLEAR development data.
    """
    target_words = max(40, int(target_words))
    sentences = split_sentences(text)
    if not sentences:
        return [str(text).strip()] if str(text).strip() else []

    segments = []
    for sentence in sentences:
        tokens = sentence.split()
        if len(tokens) > target_words:
            # Only split extremely long single sentences.
            segments.extend(" ".join(tokens[i:i + target_words])
                            for i in range(0, len(tokens), target_words))
        else:
            segments.append(sentence)

    chunks, current, count = [], [], 0
    for sentence in segments:
        n = len(sentence.split())
        if current and count + n > target_words:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += n
    if current:
        tail = " ".join(current)
        if chunks and count < target_words * 0.4:
            chunks[-1] += " " + tail
        else:
            chunks.append(tail)
    return chunks


def predict_batch(texts, artifact, encoder=None, batch_size=32) -> np.ndarray:
    """Predict from frozen feature extractors in moderate-size batches."""
    texts = list(texts)
    if not texts:
        return np.empty(0, dtype=float)
    outputs = []
    for start in range(0, len(texts), batch_size):
        subset = texts[start:start + batch_size]
        matrix = transform_texts(
            subset, artifact["representation"],
            artifact["word_vectorizer"], artifact["char_vectorizer"],
            artifact["feature_scaler"], encoder=encoder,
        )
        outputs.extend(np.asarray(artifact["model"].predict(matrix), dtype=float))
    return np.asarray(outputs, dtype=float)


def predict_document(text, artifact, encoder=None, mode="auto", target_words=180):
    """Whole or chunk-weighted BT prediction, with CLEAR-relative label.

    Chunk-weighted predictions are experimental and not score calibration.
    """
    if mode not in {"auto", "whole", "chunked"}:
        raise ValueError("mode must be auto, whole, or chunked")
    manual_features = extract_features(text)
    if not str(text).strip() or not manual_features["word_count"]:
        raise ValueError("Please enter nonempty English text")
    if mode == "auto":
        mode = "chunked" if manual_features["word_count"] > 1.5 * target_words else "whole"
    chunks = chunk_by_sentences(text, target_words) if mode == "chunked" else [text]
    scores = predict_batch(chunks, artifact, encoder=encoder)
    counts = np.asarray([max(1, len(tokenize_words(chunk))) for chunk in chunks], dtype=float)
    score = float(np.average(scores, weights=counts))
    return {
        "score": score,
        "level": readability_level(score, artifact["low_cutoff"], artifact["high_cutoff"]),
        "features": manual_features,
        "mode": mode,
        "num_chunks": len(chunks),
        "chunk_scores": scores.tolist(),
    }


EXTERNAL_LEVEL_FEATURES = [
    "score", "flesch_reading_ease", "flesch_kincaid_grade",
    "avg_sentence_length", "long_word_ratio",
]


def external_level_features(text: str, score: float) -> dict:
    """Produce OneStopEnglish classifier features at prediction time."""
    f = extract_features(text)
    return {
        "score": float(score),
        "flesch_reading_ease": float(f["flesch_reading_ease"]),
        "flesch_kincaid_grade": float(f["flesch_kincaid_grade"]),
        "avg_sentence_length": float(f["avg_sentence_length"]),
        "long_word_ratio": float(f["long_word_ratio"]),
    }


def predict_onestop_level(text, readability_artifact, calibration_artifact,
                          encoder=None):
    """Predict OneStopEnglish-like reading level; NOT CLEAR tertiles/CEFR."""
    if calibration_artifact["representation"] != readability_artifact["representation"]:
        raise ValueError("Calibrator and readability model have different representations")
    outcome = predict_document(
        text, readability_artifact, encoder=encoder,
        mode=calibration_artifact["mode"],
        target_words=calibration_artifact["target_words"],
    )
    row = external_level_features(text, outcome["score"])
    cols = calibration_artifact["feature_columns"]
    pred = calibration_artifact["classifier"].predict(pd.DataFrame([row])[cols])[0]
    return {"level": str(pred), "bt_score": outcome["score"],
            "mode": outcome["mode"], "num_chunks": outcome["num_chunks"]}
