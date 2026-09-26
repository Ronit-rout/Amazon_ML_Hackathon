"""Evaluation module implementing the F0.5 metric, threshold sweeps, and entity-level splits."""

from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd

from src.config import RANDOM_SEED, VALIDATION_SPLIT_RATIO, THRESHOLD_GRID


def compute_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """Compute the F-beta score (default beta=0.5 for precision-heavy weighting)."""
    if precision + recall == 0.0:
        return 0.0
    beta_sq = beta ** 2
    numerator = (1.0 + beta_sq) * precision * recall
    denominator = (beta_sq * precision) + recall
    return numerator / denominator if denominator > 0 else 0.0


def evaluate_entity_resolution(
    predicted_map: Dict[str, Set[str]],
    ground_truth_map: Dict[str, Set[str]],
    all_s1_eval_ids: Set[str],
) -> Dict[str, float]:
    """Compute precision, recall, F0.5, and error analysis across evaluation entities.

    Args:
        predicted_map: s1_id -> set of predicted matched IDs.
        ground_truth_map: s1_id -> set of true matched IDs.
        all_s1_eval_ids: The universe of Source 1 entities being evaluated.

    Returns:
        Dict of computed evaluation metrics.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    total_true_singletons = 0
    correct_singletons = 0
    predicted_singletons = 0

    for s1_id in all_s1_eval_ids:
        true_set = ground_truth_map.get(s1_id, set())
        pred_set = predicted_map.get(s1_id, set())

        tp = len(pred_set.intersection(true_set))
        fp = len(pred_set.difference(true_set))
        fn = len(true_set.difference(pred_set))

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Singleton diagnostics (entities with 0 matches)
        is_true_singleton = len(true_set) == 0
        is_pred_singleton = len(pred_set) == 0

        if is_true_singleton:
            total_true_singletons += 1
            if is_pred_singleton:
                correct_singletons += 1

        if is_pred_singleton:
            predicted_singletons += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f05 = compute_f_beta(precision, recall, beta=0.5)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f0_5": float(f05),
        "total_true_positives": float(total_tp),
        "total_false_positives (false merges)": float(total_fp),
        "total_false_negatives": float(total_fn),
        "total_eval_entities": float(len(all_s1_eval_ids)),
        "true_singletons": float(total_true_singletons),
        "correct_singletons": float(correct_singletons),
        "predicted_singletons": float(predicted_singletons),
    }


def split_s1_entities(
    all_s1_ids: List[str],
    val_ratio: float = VALIDATION_SPLIT_RATIO,
    random_seed: int = RANDOM_SEED,
) -> Tuple[Set[str], Set[str]]:
    """Split Source 1 entity IDs into train and validation sets.
    Splitting at entity-level strictly prevents data leakage between training and validation.
    """
    rng = np.random.default_rng(random_seed)
    shuffled_ids = np.array(sorted(list(set(all_s1_ids))))
    rng.shuffle(shuffled_ids)

    n_val = int(len(shuffled_ids) * val_ratio)
    val_ids = set(shuffled_ids[:n_val])
    train_ids = set(shuffled_ids[n_val:])

    return train_ids, val_ids


def sweep_thresholds(
    scored_candidates_df: pd.DataFrame,
    ground_truth_map: Dict[str, Set[str]],
    val_s1_ids: Set[str],
    threshold_grid: List[float] = THRESHOLD_GRID,
) -> pd.DataFrame:
    """Evaluate candidate pairs over a range of probability thresholds to optimize F0.5.

    Args:
        scored_candidates_df: DataFrame with ['source1_entity_id', 'candidate_entity_id', 'probability'].
        ground_truth_map: Ground truth mapping s1_id -> set of matched IDs.
        val_s1_ids: Set of validation S1 IDs.
        threshold_grid: List of thresholds to test.

    Returns:
        DataFrame with columns ['threshold', 'precision', 'recall', 'f0_5', 'false_positives'].
    """
    # Filter candidate dataframe to only validation entities
    val_df = scored_candidates_df[
        scored_candidates_df["source1_entity_id"].isin(val_s1_ids)
    ]

    results = []
    for th in threshold_grid:
        # Group matches above threshold
        preds: Dict[str, Set[str]] = {s1: set() for s1 in val_s1_ids}
        accepted = val_df[val_df["probability"] >= th]
        for _, row in accepted.iterrows():
            preds[row["source1_entity_id"]].add(row["candidate_entity_id"])

        metrics = evaluate_entity_resolution(preds, ground_truth_map, val_s1_ids)
        results.append(
            {
                "threshold": round(th, 2),
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f0_5": round(metrics["f0_5"], 4),
                "false_positives": int(metrics["total_false_positives (false merges)"]),
                "true_positives": int(metrics["total_true_positives"]),
                "false_negatives": int(metrics["total_false_negatives"]),
            }
        )

    return pd.DataFrame(results)
