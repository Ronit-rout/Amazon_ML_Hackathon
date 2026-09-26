"""Test inference pipeline module."""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import joblib
import pandas as pd

from src.config import (
    DEFAULT_THRESHOLD,
    SAVED_MODEL_PATH,
    SAVED_NAME_VECTORIZER_PATH,
    SAVED_ADDR_VECTORIZER_PATH,
)
from src.data_loader import load_test_data
from src.preprocessing import preprocess_dataframe
from src.blocking import generate_candidate_pairs
from src.features import FeaturePipeline
from src.model import EntityResolutionClassifier


def run_test_inference(
    s1_test_raw: pd.DataFrame,
    s2_test_raw: pd.DataFrame,
    s3_test_raw: pd.DataFrame,
    model: EntityResolutionClassifier,
    feature_pipeline: FeaturePipeline,
    threshold: float = DEFAULT_THRESHOLD,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Execute end-to-end inference over test records.

    Args:
        s1_test_raw: Raw test Source 1 dataframe.
        s2_test_raw: Raw test Source 2 dataframe.
        s3_test_raw: Raw test Source 3 dataframe.
        model: Trained entity resolution classifier.
        feature_pipeline: Fitted FeaturePipeline with TF-IDF vectorizers.
        threshold: Probability cutoff for positive match decision.

    Returns:
        Tuple of:
            - predicted_matches: dict mapping s1_id -> set of matched target IDs
            - candidate_map: dict mapping s1_id -> set of candidate target IDs
    """
    # 1. Preprocess test data
    s1_norm = preprocess_dataframe(s1_test_raw)
    s2_norm = preprocess_dataframe(s2_test_raw)
    s3_norm = preprocess_dataframe(s3_test_raw)

    all_test_s1_ids = set(s1_norm["entity_id"])
    candidate_map: Dict[str, Set[str]] = {s1: set() for s1 in all_test_s1_ids}
    predicted_matches: Dict[str, Set[str]] = {s1: set() for s1 in all_test_s1_ids}

    # 2. Candidate generation (blocking)
    candidates_df = generate_candidate_pairs(s1_norm, s2_norm, s3_norm)

    if candidates_df.empty:
        return predicted_matches, candidate_map

    # Populate candidate map
    for _, row in candidates_df.iterrows():
        candidate_map[row["source1_entity_id"]].add(row["candidate_entity_id"])

    # 3. Target combination for feature lookup
    target_combined = pd.concat([s2_norm, s3_norm], ignore_index=True)

    # 4. Feature extraction
    features_df = feature_pipeline.extract_features(
        candidates_df, s1_norm, target_combined
    )

    # 5. Predict probabilities
    probabilities = model.predict_proba(features_df)
    candidates_df["probability"] = probabilities

    # 6. Apply threshold decision
    accepted_matches = candidates_df[candidates_df["probability"] >= threshold]

    for _, row in accepted_matches.iterrows():
        predicted_matches[row["source1_entity_id"]].add(row["candidate_entity_id"])

    return predicted_matches, candidate_map


def load_artifacts_and_infer(
    threshold: float = DEFAULT_THRESHOLD,
    model_path: Path = SAVED_MODEL_PATH,
    name_vec_path: Path = SAVED_NAME_VECTORIZER_PATH,
    addr_vec_path: Path = SAVED_ADDR_VECTORIZER_PATH,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], List[str]]:
    """Load saved model and vectorizer artifacts, load test data, and run inference."""
    s1_test, s2_test, s3_test = load_test_data()
    all_s1_ids = sorted(list(s1_test["entity_id"].unique()))

    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model artifact not found at: {model_path}. Train the model first."
        )

    model = EntityResolutionClassifier.load(model_path)

    # Load fitted feature pipeline
    feature_pipeline = FeaturePipeline()
    if name_vec_path.exists() and addr_vec_path.exists():
        feature_pipeline.name_vectorizer = joblib.load(name_vec_path)
        feature_pipeline.addr_vectorizer = joblib.load(addr_vec_path)
        feature_pipeline.is_fitted = True

    preds, cands = run_test_inference(
        s1_test_raw=s1_test,
        s2_test_raw=s2_test,
        s3_test_raw=s3_test,
        model=model,
        feature_pipeline=feature_pipeline,
        threshold=threshold,
    )

    return preds, cands, all_s1_ids
