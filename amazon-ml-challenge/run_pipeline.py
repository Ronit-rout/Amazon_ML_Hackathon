"""Main CLI entry point for the Amazon ML Challenge 2026 Entity Resolution Baseline Pipeline."""

import argparse
import sys
from pathlib import Path
from typing import Dict, Set
import joblib
import pandas as pd

# Add project root to sys.path so imports work seamlessly
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    TRAIN_SOURCE1_PATH,
    TRAIN_SOURCE2_PATH,
    TRAIN_SOURCE3_PATH,
    TRAIN_GROUND_TRUTH_PATH,
    TEST_SOURCE1_PATH,
    TEST_SOURCE2_PATH,
    TEST_SOURCE3_PATH,
    SAVED_MODEL_PATH,
    SAVED_NAME_VECTORIZER_PATH,
    SAVED_ADDR_VECTORIZER_PATH,
    MATCHING_RESULTS_PATH,
    CANDIDATE_PAIRS_PATH,
    RANDOM_SEED,
    DEFAULT_THRESHOLD,
    VALIDATION_SPLIT_RATIO,
    NEGATIVE_SAMPLE_RATIO,
    THRESHOLD_GRID,
)
from src.data_loader import load_train_data, load_test_data, parse_ground_truth_map
from src.preprocessing import preprocess_dataframe
from src.blocking import generate_candidate_pairs, compute_blocking_recall
from src.features import FeaturePipeline, FEATURE_NAMES
from src.model import build_training_dataset, EntityResolutionClassifier
from src.evaluation import split_s1_entities, sweep_thresholds, evaluate_entity_resolution
from src.inference import run_test_inference
from src.submission import (
    generate_matching_results,
    generate_candidate_pairs as generate_submission_candidates,
    validate_submission_file,
)


def check_train_files_exist() -> bool:
    """Verify presence of all training files."""
    required = [
        TRAIN_SOURCE1_PATH,
        TRAIN_SOURCE2_PATH,
        TRAIN_SOURCE3_PATH,
        TRAIN_GROUND_TRUTH_PATH,
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("\n[!] Training file(s) missing:")
        for p in missing:
            print(f"    - {p.resolve()}")
        print("\nPlease copy the challenge dataset into 'dataset/train/' before training.")
        return False
    return True


def check_test_files_exist() -> bool:
    """Verify presence of all test files."""
    required = [
        TEST_SOURCE1_PATH,
        TEST_SOURCE2_PATH,
        TEST_SOURCE3_PATH,
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("\n[!] Test file(s) missing:")
        for p in missing:
            print(f"    - {p.resolve()}")
        print("\nPlease copy the test dataset into 'dataset/test/' before running inference.")
        return False
    return True


def run_train_and_validate(
    random_seed: int = RANDOM_SEED,
    val_ratio: float = VALIDATION_SPLIT_RATIO,
) -> float:
    """Execute training, entity-level validation, and threshold selection."""
    if not check_train_files_exist():
        sys.exit(1)

    print("\n" + "=" * 70)
    print("STEP 1: Loading and Validating Training Datasets")
    print("=" * 70)
    s1_train, s2_train, s3_train, gt_train = load_train_data()
    gt_map = parse_ground_truth_map(gt_train)
    all_s1_ids = list(gt_map.keys())

    print(f"Loaded Source 1: {len(s1_train):,} records")
    print(f"Loaded Source 2: {len(s2_train):,} records")
    print(f"Loaded Source 3: {len(s3_train):,} records")
    print(f"Loaded Ground Truth: {len(gt_train):,} S1 entities")

    print("\n" + "=" * 70)
    print("STEP 2: Preprocessing and Normalization")
    print("=" * 70)
    s1_norm = preprocess_dataframe(s1_train)
    s2_norm = preprocess_dataframe(s2_train)
    s3_norm = preprocess_dataframe(s3_train)
    targets_norm = pd.concat([s2_norm, s3_norm], ignore_index=True)
    print("Added normalized columns (business_name, business_address, country).")

    print("\n" + "=" * 70)
    print("STEP 3: Entity-Level Validation Split (No Data Leakage)")
    print("=" * 70)
    train_s1_ids, val_s1_ids = split_s1_entities(
        all_s1_ids, val_ratio=val_ratio, random_seed=random_seed
    )
    print(f"Total S1 entities: {len(all_s1_ids):,}")
    print(f"Training S1 entities: {len(train_s1_ids):,} ({(1 - val_ratio)*100:.0f}%)")
    print(f"Validation S1 entities: {len(val_s1_ids):,} ({val_ratio*100:.0f}%)")

    print("\n" + "=" * 70)
    print("STEP 4: Candidate Pair Generation (Blocking)")
    print("=" * 70)
    candidates_df = generate_candidate_pairs(s1_norm, s2_norm, s3_norm)
    print(f"Generated {len(candidates_df):,} total candidate pairs.")

    recall_metrics = compute_blocking_recall(candidates_df, gt_map)
    print(f"Blocking Recall across Ground Truth: {recall_metrics['blocking_recall']*100:.2f}% "
          f"({int(recall_metrics['found_true_matches'])}/{int(recall_metrics['total_true_matches'])} true matches)")

    print("\n" + "=" * 70)
    print("STEP 5: Feature Engineering & Vectorization")
    print("=" * 70)
    feature_pipeline = FeaturePipeline()
    all_names = list(
        pd.concat([s1_norm["business_name_normalized"], targets_norm["business_name_normalized"]])
    )
    all_addrs = list(
        pd.concat([s1_norm["business_address_normalized"], targets_norm["business_address_normalized"]])
    )
    feature_pipeline.fit_vectorizers(all_names, all_addrs)
    print("Fitted TF-IDF character n-gram vectorizers.")

    features_df = feature_pipeline.extract_features(candidates_df, s1_norm, targets_norm)
    print(f"Extracted {len(FEATURE_NAMES)} pairwise features across {len(features_df):,} candidate pairs.")

    print("\n" + "=" * 70)
    print("STEP 6: Building Balanced Training Samples")
    print("=" * 70)
    train_candidate_mask = candidates_df["source1_entity_id"].isin(train_s1_ids)
    train_cands = candidates_df[train_candidate_mask].reset_index(drop=True)
    train_feats = features_df[train_candidate_mask].reset_index(drop=True)

    sampled_cands, y_train, sampled_feats = build_training_dataset(
        train_cands,
        train_feats,
        gt_map,
        negative_sample_ratio=NEGATIVE_SAMPLE_RATIO,
        random_seed=random_seed,
    )
    print(f"Training dataset: {len(y_train):,} pairs (Positives: {sum(y_train):,}, Negatives: {len(y_train)-sum(y_train):,})")

    print("\n" + "=" * 70)
    print("STEP 7: Training Baseline Supervised Model")
    print("=" * 70)
    model = EntityResolutionClassifier(model_type="logistic_regression")
    model.fit(sampled_feats, y_train)
    print("Logistic Regression model successfully trained.")

    importances = model.get_feature_importances(FEATURE_NAMES)
    print("\nTop 5 feature weights:")
    for feat, weight in list(importances.items())[:5]:
        print(f"  {feat:30s}: {weight:+.4f}")

    print("\n" + "=" * 70)
    print("STEP 8: Validation & Threshold Sweep on F0.5")
    print("=" * 70)
    val_candidate_mask = candidates_df["source1_entity_id"].isin(val_s1_ids)
    val_cands = candidates_df[val_candidate_mask].copy().reset_index(drop=True)
    val_feats = features_df[val_candidate_mask].reset_index(drop=True)

    val_probs = model.predict_proba(val_feats)
    val_cands["probability"] = val_probs

    sweep_df = sweep_thresholds(val_cands, gt_map, val_s1_ids, THRESHOLD_GRID)
    print("\nThreshold Sweep Results (F0.5 Precision-weighted evaluation):")
    print(sweep_df.to_string(index=False))

    best_idx = sweep_df["f0_5"].idxmax()
    best_row = sweep_df.iloc[best_idx]
    best_threshold = float(best_row["threshold"])
    print(f"\n[BEST THRESHOLD]: {best_threshold:.2f} (F0.5 = {best_row['f0_5']:.4f}, Precision = {best_row['precision']:.4f}, Recall = {best_row['recall']:.4f})")

    # Save artifacts
    SAVED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(SAVED_MODEL_PATH)
    joblib.dump(feature_pipeline.name_vectorizer, SAVED_NAME_VECTORIZER_PATH)
    joblib.dump(feature_pipeline.addr_vectorizer, SAVED_ADDR_VECTORIZER_PATH)
    print(f"\nSaved model and vectorizers to: {SAVED_MODEL_PATH.parent.resolve()}")

    return best_threshold


def run_predict_pipeline(threshold: float = DEFAULT_THRESHOLD) -> None:
    """Execute inference and write final submission outputs."""
    if not check_test_files_exist():
        sys.exit(1)

    if not SAVED_MODEL_PATH.exists():
        print(f"\n[!] Model file not found at: {SAVED_MODEL_PATH.resolve()}")
        print("Please run training first: python run_pipeline.py --mode train")
        sys.exit(1)

    print("\n" + "=" * 70)
    print(f"STEP 9: Running Test Inference with Threshold = {threshold:.2f}")
    print("=" * 70)

    s1_test, s2_test, s3_test = load_test_data()
    all_test_s1_ids = sorted(list(s1_test["entity_id"].unique()))
    print(f"Loaded Test Source 1: {len(s1_test):,} records")
    print(f"Loaded Test Source 2: {len(s2_test):,} records")
    print(f"Loaded Test Source 3: {len(s3_test):,} records")

    model = EntityResolutionClassifier.load(SAVED_MODEL_PATH)
    feature_pipeline = FeaturePipeline()
    feature_pipeline.name_vectorizer = joblib.load(SAVED_NAME_VECTORIZER_PATH)
    feature_pipeline.addr_vectorizer = joblib.load(SAVED_ADDR_VECTORIZER_PATH)
    feature_pipeline.is_fitted = True

    preds, cands = run_test_inference(
        s1_test_raw=s1_test,
        s2_test_raw=s2_test,
        s3_test_raw=s3_test,
        model=model,
        feature_pipeline=feature_pipeline,
        threshold=threshold,
    )

    print("\n" + "=" * 70)
    print("STEP 10: Generating and Validating Output TSVs")
    print("=" * 70)

    # Generate matching_results.tsv
    matching_df = generate_matching_results(
        predicted_matches=preds,
        all_test_s1_ids=all_test_s1_ids,
        output_path=MATCHING_RESULTS_PATH,
    )
    validate_submission_file(
        filepath=MATCHING_RESULTS_PATH,
        expected_s1_ids=set(all_test_s1_ids),
        target_column_name="matched_entity_ids",
    )
    print(f"Successfully generated and validated: {MATCHING_RESULTS_PATH.resolve()}")

    # Generate candidate_pairs.tsv
    cand_df = generate_submission_candidates(
        candidate_map=cands,
        all_test_s1_ids=all_test_s1_ids,
        output_path=CANDIDATE_PAIRS_PATH,
    )
    validate_submission_file(
        filepath=CANDIDATE_PAIRS_PATH,
        expected_s1_ids=set(all_test_s1_ids),
        target_column_name="candidate_entity_ids",
    )
    print(f"Successfully generated and validated: {CANDIDATE_PAIRS_PATH.resolve()}")

    matches_found = sum(1 for m in preds.values() if len(m) > 0)
    print(f"\nInference Summary:")
    print(f"  Total Source 1 test records: {len(all_test_s1_ids):,}")
    print(f"  Entities with >= 1 match:    {matches_found:,} ({matches_found/len(all_test_s1_ids)*100:.1f}%)")
    print(f"  Entities with 0 matches:     {len(all_test_s1_ids)-matches_found:,} ({(len(all_test_s1_ids)-matches_found)/len(all_test_s1_ids)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Amazon ML Challenge 2026 - Business Entity Resolution Baseline Pipeline"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "predict", "full"],
        default="train",
        help="Pipeline execution mode: 'train', 'predict', or 'full' (train + predict).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Probability threshold for matching. If omitted during full/predict, optimal tuned threshold is used.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {RANDOM_SEED}).",
    )

    args = parser.parse_args()

    if args.mode in ["train", "full"]:
        best_threshold = run_train_and_validate(random_seed=args.seed)
        if args.mode == "full":
            eval_th = args.threshold if args.threshold is not None else best_threshold
            run_predict_pipeline(threshold=eval_th)
    elif args.mode == "predict":
        eval_th = args.threshold if args.threshold is not None else DEFAULT_THRESHOLD
        run_predict_pipeline(threshold=eval_th)


if __name__ == "__main__":
    main()
