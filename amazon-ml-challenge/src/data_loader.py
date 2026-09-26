"""Data loading and validation module for the Entity Resolution pipeline."""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

from src.config import (
    REQUIRED_ENTITY_COLUMNS,
    REQUIRED_GROUND_TRUTH_COLUMNS,
    VALID_SOURCE_PREFIXES,
    TRAIN_SOURCE1_PATH,
    TRAIN_SOURCE2_PATH,
    TRAIN_SOURCE3_PATH,
    TRAIN_GROUND_TRUTH_PATH,
    TEST_SOURCE1_PATH,
    TEST_SOURCE2_PATH,
    TEST_SOURCE3_PATH,
)


def validate_source_dataframe(
    df: pd.DataFrame, source_name: str, expected_prefix: str
) -> None:
    """Validate source TSV dataframe integrity and format.

    Args:
        df: The loaded dataframe.
        source_name: Descriptive name for logging / error context.
        expected_prefix: Expected entity ID prefix (e.g., 'S1-', 'S2-', 'S3-').

    Raises:
        ValueError: If format, column counts, or prefixes fail validation.
    """
    if df.shape[1] <= 1:
        raise ValueError(
            f"[{source_name}] Loaded only {df.shape[1]} column(s). "
            f"Check if the file is valid TSV separated by tab ('\\t')."
        )

    missing_cols = [col for col in REQUIRED_ENTITY_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"[{source_name}] Missing required columns: {missing_cols}. Found: {list(df.columns)}"
        )

    if df["entity_id"].isnull().any():
        null_count = int(df["entity_id"].isnull().sum())
        raise ValueError(
            f"[{source_name}] Found {null_count} rows with null entity_id."
        )

    # Prefix verification
    invalid_prefixes = df[~df["entity_id"].astype(str).str.startswith(expected_prefix)]
    if not invalid_prefixes.empty:
        sample_ids = invalid_prefixes["entity_id"].head(3).tolist()
        raise ValueError(
            f"[{source_name}] Found entity IDs without expected prefix '{expected_prefix}'. "
            f"Examples: {sample_ids}"
        )

    # Duplicate ID check within source
    duplicate_ids = df[df["entity_id"].duplicated(keep=False)]
    if not duplicate_ids.empty:
        dup_count = int(duplicate_ids["entity_id"].nunique())
        raise ValueError(
            f"[{source_name}] Found {dup_count} duplicated entity_id(s) within the file."
        )


def load_source_tsv(filepath: Path, expected_prefix: str, source_name: str) -> pd.DataFrame:
    """Load and validate an entity TSV file.

    Args:
        filepath: Path to the TSV file.
        expected_prefix: Prefix to enforce on entity_id.
        source_name: Name for validation diagnostics.

    Returns:
        Validated pandas DataFrame with string types for primary fields.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath.resolve()}")

    df = pd.read_csv(filepath, sep="\t", dtype=str, keep_default_na=False)

    # Clean whitespace in column names
    df.columns = df.columns.str.strip()

    validate_source_dataframe(df, source_name, expected_prefix)
    return df


def load_ground_truth(filepath: Path) -> pd.DataFrame:
    """Load and validate the training ground truth file.

    Returns:
        DataFrame with columns:
            - source1_entity_id: str
            - matched_entity_ids: str (raw comma-separated or empty)
            - parsed_matched_ids: List[str] (parsed list of S2/S3 IDs)
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Ground truth file not found: {filepath.resolve()}")

    df = pd.read_csv(filepath, sep="\t", dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip()

    if df.shape[1] <= 1:
        raise ValueError(
            "Ground truth loaded with <=1 column. Ensure file is tab-separated."
        )

    missing_cols = [
        col for col in REQUIRED_GROUND_TRUTH_COLUMNS if col not in df.columns
    ]
    if missing_cols:
        raise ValueError(f"Ground truth missing required columns: {missing_cols}")

    # Ensure source1 IDs start with S1-
    invalid_s1 = df[
        ~df["source1_entity_id"].astype(str).str.startswith(VALID_SOURCE_PREFIXES["source1"])
    ]
    if not invalid_s1.empty:
        raise ValueError(
            f"Ground truth contains source1_entity_id not starting with 'S1-': "
            f"{invalid_s1['source1_entity_id'].head(3).tolist()}"
        )

    # Parse matched entity IDs into list
    def _parse_matches(raw_matches: str) -> List[str]:
        if not raw_matches or pd.isna(raw_matches):
            return []
        items = [item.strip() for item in str(raw_matches).split(",") if item.strip()]
        # Check prefix validity: only S2- or S3- allowed
        for m in items:
            if not (m.startswith(VALID_SOURCE_PREFIXES["source2"]) or m.startswith(VALID_SOURCE_PREFIXES["source3"])):
                raise ValueError(
                    f"Ground truth matched_entity_id '{m}' has invalid prefix. Only S2- and S3- allowed."
                )
        return items

    df["parsed_matched_ids"] = df["matched_entity_ids"].apply(_parse_matches)
    return df


def parse_ground_truth_map(gt_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Convert ground truth dataframe to a dictionary mapping source1_id -> set of matched_ids."""
    return dict(zip(gt_df["source1_entity_id"], gt_df["parsed_matched_ids"].apply(set)))


def load_train_data(
    source1_path: Path = TRAIN_SOURCE1_PATH,
    source2_path: Path = TRAIN_SOURCE2_PATH,
    source3_path: Path = TRAIN_SOURCE3_PATH,
    gt_path: Path = TRAIN_GROUND_TRUTH_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience loader for all training datasets."""
    s1 = load_source_tsv(source1_path, VALID_SOURCE_PREFIXES["source1"], "Train Source 1")
    s2 = load_source_tsv(source2_path, VALID_SOURCE_PREFIXES["source2"], "Train Source 2")
    s3 = load_source_tsv(source3_path, VALID_SOURCE_PREFIXES["source3"], "Train Source 3")
    gt = load_ground_truth(gt_path)
    return s1, s2, s3, gt


def load_test_data(
    source1_path: Path = TEST_SOURCE1_PATH,
    source2_path: Path = TEST_SOURCE2_PATH,
    source3_path: Path = TEST_SOURCE3_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience loader for all test datasets."""
    s1 = load_source_tsv(source1_path, VALID_SOURCE_PREFIXES["source1"], "Test Source 1")
    s2 = load_source_tsv(source2_path, VALID_SOURCE_PREFIXES["source2"], "Test Source 2")
    s3 = load_source_tsv(source3_path, VALID_SOURCE_PREFIXES["source3"], "Test Source 3")
    return s1, s2, s3
