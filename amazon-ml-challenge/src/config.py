"""Configuration constants and paths for the Business Entity Resolution Baseline Pipeline."""

from pathlib import Path
from typing import List

# ==============================================================================
# Path Configurations
# ==============================================================================
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Training Data Paths
TRAIN_SOURCE1_PATH = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2_PATH = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3_PATH = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH_PATH = TRAIN_DIR / "train_ground_truth.tsv"

# Test Data Paths
TEST_SOURCE1_PATH = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2_PATH = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3_PATH = TEST_DIR / "test_source3.tsv"

# Output Paths
MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"
SAVED_MODEL_PATH = OUTPUT_DIR / "baseline_model.joblib"
SAVED_NAME_VECTORIZER_PATH = OUTPUT_DIR / "tfidf_name_vectorizer.joblib"
SAVED_ADDR_VECTORIZER_PATH = OUTPUT_DIR / "tfidf_addr_vectorizer.joblib"

# ==============================================================================
# Validation & Schema Requirements
# ==============================================================================
REQUIRED_ENTITY_COLUMNS: List[str] = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

REQUIRED_GROUND_TRUTH_COLUMNS: List[str] = [
    "source1_entity_id",
    "matched_entity_ids",
]

VALID_SOURCE_PREFIXES = {
    "source1": "S1-",
    "source2": "S2-",
    "source3": "S3-",
}

# ==============================================================================
# Model & Pipeline Hyperparameters
# ==============================================================================
RANDOM_SEED: int = 42
VALIDATION_SPLIT_RATIO: float = 0.2

# Blocking
MIN_BLOCKING_TOKEN_JACCARD: float = 0.3
BLOCKING_NGRAM_SIZE: int = 3
MIN_BLOCKING_NGRAM_JACCARD: float = 0.25

# Training Data Balance
NEGATIVE_SAMPLE_RATIO: int = 5  # Negatives sampled per positive candidate

# Evaluation & Decision
DEFAULT_THRESHOLD: float = 0.50
THRESHOLD_GRID: List[float] = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]

# TF-IDF Feature Config
TFIDF_MAX_FEATURES: int = 5000
