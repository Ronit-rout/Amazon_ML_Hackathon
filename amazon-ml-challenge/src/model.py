"""Supervised matching model module for binary entity pair classification."""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator

from src.config import RANDOM_SEED, NEGATIVE_SAMPLE_RATIO


def build_training_dataset(
    candidates_df: pd.DataFrame,
    features_df: pd.DataFrame,
    ground_truth_map: Dict[str, Set[str]],
    negative_sample_ratio: int = NEGATIVE_SAMPLE_RATIO,
    random_seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Build positive and negative training dataset from candidate pairs.

    Args:
        candidates_df: Candidate pairs ['source1_entity_id', 'candidate_entity_id'].
        features_df: Aligned feature matrix.
        ground_truth_map: Mapping of source1_id -> set of matched entity IDs.
        negative_sample_ratio: Ratio of negative pairs per positive pair.
        random_seed: Seed for reproducible sampling.

    Returns:
        Tuple of (sampled_candidates_df, y_labels, sampled_features_df).
    """
    labels = []
    for _, row in candidates_df.iterrows():
        s1_id = row["source1_entity_id"]
        cand_id = row["candidate_entity_id"]
        is_match = 1 if cand_id in ground_truth_map.get(s1_id, set()) else 0
        labels.append(is_match)

    labels_arr = np.array(labels, dtype=int)
    pos_indices = np.where(labels_arr == 1)[0]
    neg_indices = np.where(labels_arr == 0)[0]

    n_pos = len(pos_indices)
    if n_pos == 0:
        raise ValueError(
            "No positive training pairs found in candidate set! Check blocking recall."
        )

    n_neg = len(neg_indices)
    if n_neg == 0:
        raise ValueError(
            "No negative training pairs found in candidate set! Binary classification requires both matching and non-matching candidate pairs."
        )

    # Subsample negatives if they exceed the ratio
    max_negatives = n_pos * negative_sample_ratio
    rng = np.random.default_rng(random_seed)

    if len(neg_indices) > max_negatives:
        sampled_neg_indices = rng.choice(neg_indices, size=max_negatives, replace=False)
    else:
        sampled_neg_indices = neg_indices

    selected_indices = np.concatenate([pos_indices, sampled_neg_indices])
    rng.shuffle(selected_indices)

    return (
        candidates_df.iloc[selected_indices].reset_index(drop=True),
        labels_arr[selected_indices],
        features_df.iloc[selected_indices].reset_index(drop=True),
    )


class EntityResolutionClassifier:
    """Configurable wrapper for candidate pair matching models."""

    def __init__(self, model_type: str = "logistic_regression", **kwargs):
        self.model_type = model_type
        if model_type == "logistic_regression":
            self.model: BaseEstimator = LogisticRegression(
                C=kwargs.get("C", 1.0),
                class_weight=kwargs.get("class_weight", "balanced"),
                max_iter=kwargs.get("max_iter", 1000),
                random_state=RANDOM_SEED,
                solver="lbfgs",
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray) -> "EntityResolutionClassifier":
        """Train the classifier on candidate pair features."""
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict match probability (class 1) for candidate pairs."""
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            # Return probability of positive class (index 1)
            return probs[:, 1]
        elif hasattr(self.model, "decision_function"):
            decision = self.model.decision_function(X)
            # Sigmoid transform
            return 1.0 / (1.0 + np.exp(-decision))
        else:
            preds = self.model.predict(X)
            return preds.astype(float)

    def predict(self, X: pd.DataFrame | np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary match status using decision threshold."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def get_feature_importances(self, feature_names: List[str]) -> Dict[str, float]:
        """Extract model coefficients or feature importances."""
        if hasattr(self.model, "coef_"):
            coefs = self.model.coef_[0]
            return dict(sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True))
        return {}

    def save(self, filepath: Path) -> None:
        """Serialize trained model to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)

    @classmethod
    def load(cls, filepath: Path, model_type: str = "logistic_regression") -> "EntityResolutionClassifier":
        """Load serialized model from disk."""
        instance = cls(model_type=model_type)
        instance.model = joblib.load(filepath)
        return instance
