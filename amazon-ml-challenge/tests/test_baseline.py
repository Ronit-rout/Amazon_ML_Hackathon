import sys
from pathlib import Path
import tempfile
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    validate_source_dataframe,
    load_source_tsv,
    load_ground_truth,
    parse_ground_truth_map,
)
from src.preprocessing import preprocess_dataframe, normalize_business_name
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


def test_synthetic_end_to_end_pipeline():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Create Synthetic Datasets
        s1_df = pd.DataFrame(
            [
                {
                    "entity_id": "S1-001",
                    "business_name": "Starbucks Coffee #102",
                    "business_address": "123 Main St, Seattle, WA",
                    "country": "US",
                },
                {
                    "entity_id": "S1-002",
                    "business_name": "Target Store",
                    "business_address": "456 Market Ave, San Jose, CA",
                    "country": "US",
                },
                {
                    "entity_id": "S1-003",
                    "business_name": "Boulangerie Patisserie Paris",
                    "business_address": "10 Rue de Rivoli, Paris",
                    "country": "France",
                },
            ]
        )

        s2_df = pd.DataFrame(
            [
                {
                    "entity_id": "S2-101",
                    "business_name": "Starbucks Coffee",
                    "business_address": "123 Main Street",
                    "country": "US",
                },
                {
                    "entity_id": "S2-102",
                    "business_name": "Target Retail Store",
                    "business_address": "456 Market Avenue, San Jose",
                    "country": "US",
                },
                {
                    "entity_id": "S2-103",
                    "business_name": "Starbucks Coffee Shop",
                    "business_address": "999 Tech Park",
                    "country": "US",
                },
            ]
        )

        s3_df = pd.DataFrame(
            [
                {
                    "entity_id": "S3-201",
                    "business_name": "Starbucks Seattle #102",
                    "business_address": "123 Main St",
                    "country": "US",
                },
                {
                    "entity_id": "S3-202",
                    "business_name": "Paris Boulangerie",
                    "business_address": "10 Rue de Rivoli, 75001 Paris",
                    "country": "France",
                },
            ]
        )

        # Ground truth:
        # S1-001 -> S2-101, S3-201 (multi-match across S2 & S3)
        # S1-002 -> S2-102 (1-to-1 match)
        # S1-003 -> S3-202 (cross-country France match)
        gt_df = pd.DataFrame(
            [
                {"source1_entity_id": "S1-001", "matched_entity_ids": "S2-101,S3-201"},
                {"source1_entity_id": "S1-002", "matched_entity_ids": "S2-102"},
                {"source1_entity_id": "S1-003", "matched_entity_ids": "S3-202"},
            ]
        )

        # Save to TSV
        s1_path = tmp_path / "s1.tsv"
        s2_path = tmp_path / "s2.tsv"
        s3_path = tmp_path / "s3.tsv"
        gt_path = tmp_path / "gt.tsv"

        s1_df.to_csv(s1_path, sep="\t", index=False)
        s2_df.to_csv(s2_path, sep="\t", index=False)
        s3_df.to_csv(s3_path, sep="\t", index=False)
        gt_df.to_csv(gt_path, sep="\t", index=False)

        # 2. Test Loading
        loaded_s1 = load_source_tsv(s1_path, "S1-", "S1 Test")
        loaded_s2 = load_source_tsv(s2_path, "S2-", "S2 Test")
        loaded_s3 = load_source_tsv(s3_path, "S3-", "S3 Test")
        loaded_gt = load_ground_truth(gt_path)

        assert len(loaded_s1) == 3
        assert len(loaded_gt) == 3
        gt_map = parse_ground_truth_map(loaded_gt)
        assert gt_map["S1-001"] == {"S2-101", "S3-201"}

        # 3. Test Preprocessing
        s1_norm = preprocess_dataframe(loaded_s1)
        s2_norm = preprocess_dataframe(loaded_s2)
        s3_norm = preprocess_dataframe(loaded_s3)
        targets_norm = pd.concat([s2_norm, s3_norm], ignore_index=True)

        assert "business_name_normalized" in s1_norm.columns
        assert s1_norm.iloc[0]["business_name_normalized"] == "starbucks coffee #102"

        # 4. Test Blocking
        candidates_df = generate_candidate_pairs(s1_norm, s2_norm, s3_norm)
        assert not candidates_df.empty

        recall_info = compute_blocking_recall(candidates_df, gt_map)
        assert recall_info["blocking_recall"] == 1.0  # All true matches found

        # 5. Test Features
        pipeline = FeaturePipeline(max_features=500)
        pipeline.fit_vectorizers(
            list(s1_norm["business_name_normalized"])
            + list(targets_norm["business_name_normalized"]),
            list(s1_norm["business_address_normalized"])
            + list(targets_norm["business_address_normalized"]),
        )
        features_df = pipeline.extract_features(candidates_df, s1_norm, targets_norm)
        assert len(features_df) == len(candidates_df)
        assert list(features_df.columns) == FEATURE_NAMES

        # 6. Test Model Dataset Builder & Classifier
        train_cands, y_train, train_feats = build_training_dataset(
            candidates_df, features_df, gt_map, negative_sample_ratio=2, random_seed=42
        )
        assert 1 in y_train
        assert 0 in y_train

        clf = EntityResolutionClassifier(model_type="logistic_regression")
        clf.fit(train_feats, y_train)
        probs = clf.predict_proba(train_feats)
        assert len(probs) == len(train_feats)
        assert all(0.0 <= p <= 1.0 for p in probs)

        # 7. Test Evaluation & Sweep
        candidates_df["probability"] = clf.predict_proba(features_df)
        sweep_res = sweep_thresholds(
            candidates_df, gt_map, set(s1_df["entity_id"]), [0.5, 0.7]
        )
        assert len(sweep_res) == 2
        assert "f0_5" in sweep_res.columns

        # 8. Test Inference
        preds, cands = run_test_inference(
            loaded_s1, loaded_s2, loaded_s3, clf, pipeline, threshold=0.5
        )
        assert "S1-001" in preds

        # 9. Test Submission TSVs
        results_tsv = tmp_path / "matching_results.tsv"
        cand_tsv = tmp_path / "candidate_pairs.tsv"

        generate_matching_results(preds, list(s1_df["entity_id"]), results_tsv)
        generate_submission_candidates(cands, list(s1_df["entity_id"]), cand_tsv)

        # Validate constraints
        assert validate_submission_file(
            results_tsv, set(s1_df["entity_id"]), "matched_entity_ids"
        )
        assert validate_submission_file(
            cand_tsv, set(s1_df["entity_id"]), "candidate_entity_ids"
        )


if __name__ == "__main__":
    test_synthetic_end_to_end_pipeline()
    print("Integration test passed successfully!")
