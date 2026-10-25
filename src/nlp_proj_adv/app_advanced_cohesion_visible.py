"""Readability model + separately validated optional external level classifier."""
from pathlib import Path
import hashlib
import joblib
import pandas as pd
import streamlit as st
from readability_advanced_utils import (
    EMBEDDING_MODEL, predict_document, predict_onestop_level,
    semantic_cohesion, split_sentences, suggest_improvements,
)

MODEL_FILE = Path("readability_advanced_model.joblib")
LEVEL_FILE = Path("onestop_level_classifier.joblib")
st.set_page_config(page_title="Readability & Semantic Continuity", page_icon="📖", layout="wide")
st.title("📖 Readability & Semantic Continuity")
st.caption("CLEAR BT scores, optional OneStopEnglish-style levels, and exploratory semantic continuity.")

@st.cache_resource
def get_readability_model():
    return joblib.load(MODEL_FILE)

@st.cache_resource
def get_external_classifier():
    return joblib.load(LEVEL_FILE)

@st.cache_resource
def get_encoder(name):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)

if not MODEL_FILE.exists():
    st.error("No trained model found. Run the training cells in the notebook first.")
    st.stop()
artifact = get_readability_model()
calibration = get_external_classifier() if LEVEL_FILE.exists() else None
st.info(f"Readability model: {artifact['representation']} | CLEAR test MAE: {artifact['test_mae']:.3f}")

text = st.text_area(
    "English passage", height=210,
    value=("Regular exercise improves health. Walking for a short time every day "
           "helps people stay active. Over time, small habits can make a difference."),
)
mode = st.selectbox(
    "How should long texts be handled?",
    ["auto", "whole", "chunked"],
    help="Auto uses whole-text prediction for typical CLEAR lengths and chunking for longer documents."
)

if st.button("Analyze", type="primary"):
    if not text.strip():
        st.warning("Please enter some English text.")
        st.stop()
    need_encoder = True  # Cohesion scores are displayed on every analysis.
    try:
        encoder = get_encoder(artifact.get("encoder_name", EMBEDDING_MODEL)) if need_encoder else None
        target_words = int(artifact.get("clear_median_words", 180))
        with st.spinner("Analyzing passage..."):
            result = predict_document(text, artifact, encoder=encoder,
                                      mode=mode, target_words=target_words)
            sem = semantic_cohesion(text, encoder)
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        st.stop()

    a, b, c = st.columns(3)
    a.metric("Predicted BT_easiness", f"{result['score']:.3f}")
    b.metric("Words", int(result["features"]["word_count"]))
    c.metric("Chunks", result["num_chunks"])
    st.write(f"**CLEAR-relative category:** {result['level']}")
    st.caption(f"Inference mode: {result['mode']}. Higher BT = easier; CLEAR categories are not school grades.")
    if result["features"]["word_count"] > 1.5 * target_words:
        st.warning("This input is long compared with CLEAR. Exact BT score may not transfer reliably.")
    if calibration is not None:
        try:
            if (calibration.get("representation") == artifact["representation"] and
                calibration.get("readability_artifact_sha256") == hashlib.sha256(MODEL_FILE.read_bytes()).hexdigest()):
                inferred = predict_onestop_level(text, artifact, calibration, encoder=encoder)
                names = {"ele": "Elementary", "int": "Intermediate", "adv": "Advanced"}
                st.write("**OneStopEnglish-style level (separately trained):** " +
                         names.get(inferred["level"], inferred["level"]))
                st.caption("This secondary classifier was trained on OneStopEnglish articles. "
                           "It is NOT a CLEAR category, CEFR certification, or guaranteed learner level. "
                           f"Uses {inferred['mode']} input handling. "
                           f"Held-out macro-F1: {calibration['heldout_macro_f1']:.3f}.")
            else:
                st.warning("A reading-level classifier was found for a different readability model; rerun the calibration cells.")
        except Exception as exc:
            st.warning(f"The external classifier could not be used: {exc}")
    else:
        st.caption("Run the OneStopEnglish calibration section of the notebook to add separate learner-level predictions.")
    st.subheader("Cohesion / semantic continuity")
    lex_col, sem_col = st.columns(2)
    lex_col.metric("Lexical cohesion (word overlap)", f"{result['features']['lexical_cohesion']:.3f}")
    if sem["mean"] is not None:
        sem_col.metric("Semantic cohesion (cosine)", f"{sem['mean']:.3f}")
    else:
        sem_col.metric("Semantic cohesion (cosine)", "N/A")
    st.caption(
        "Lexical cohesion measures content-word overlap (0–1). "
        "Semantic cohesion measures neighboring-sentence similarity (-1 to 1). "
        "Neither is a validated logical-coherence score."
    )
    if sem["pairs"]:
        with st.expander("Show sentence-by-sentence similarity"):
            pair_df = pd.DataFrame({
                "Sentence A": sem["sentences"][:-1],
                "Sentence B": sem["sentences"][1:],
                "Cosine similarity": sem["pairs"],
            })
            st.dataframe(pair_df, use_container_width=True, hide_index=True)
            st.line_chart(pd.DataFrame({"Similarity": sem["pairs"]}).rename_axis("Adjacent sentence pair"))
    else:
        st.info("Enter at least two sentences to calculate adjacent-sentence semantic cohesion.")

    st.subheader("Readability features")
    st.dataframe(pd.Series(result["features"], name="value").to_frame())
    st.subheader("Writing suggestions (heuristic)")
    for item in suggest_improvements(result["features"]):
        st.write("•", item)
