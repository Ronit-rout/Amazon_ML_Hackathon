"""Submission formatting and validation module for challenge outputs."""

from pathlib import Path
from typing import Dict, List, Set
import pandas as pd

from src.config import (
    MATCHING_RESULTS_PATH,
    CANDIDATE_PAIRS_PATH,
    VALID_SOURCE_PREFIXES,
)


def format_grouped_ids_tsv(
    id_map: Dict[str, Set[str]],
    all_s1_ids: List[str],
    id_column_name: str,
    output_path: Path,
) -> pd.DataFrame:
    """Format an entity-to-set-of-IDs mapping into a validated output TSV.

    Args:
        id_map: Mapping of source1_id -> set of target IDs.
        all_s1_ids: Complete, ordered list of test Source 1 entity IDs.
        id_column_name: Column name for the target IDs ('matched_entity_ids' or 'candidate_entity_ids').
        output_path: Destination TSV path.

    Returns:
        The formatted pandas DataFrame.
    """
    rows = []
    # Ensure every single S1 entity appears exactly once, ordered
    for s1_id in sorted(list(all_s1_ids)):
        targets = sorted(list(id_map.get(s1_id, set())))
        # Filter strictly to valid S2 and S3 prefixes
        valid_targets = [
            t
            for t in targets
            if (
                t.startswith(VALID_SOURCE_PREFIXES["source2"])
                or t.startswith(VALID_SOURCE_PREFIXES["source3"])
            )
            and not t.startswith(VALID_SOURCE_PREFIXES["source1"])
        ]
        # Remove any internal duplicates while preserving order
        deduped_targets = list(dict.fromkeys(valid_targets))
        joined_str = ",".join(deduped_targets) if deduped_targets else ""
        rows.append(
            {
                "source1_entity_id": s1_id,
                id_column_name: joined_str,
            }
        )

    out_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, sep="\t", index=False)
    return out_df


def generate_matching_results(
    predicted_matches: Dict[str, Set[str]],
    all_test_s1_ids: List[str],
    output_path: Path = MATCHING_RESULTS_PATH,
) -> pd.DataFrame:
    """Write matching_results.tsv with exactly [source1_entity_id, matched_entity_ids]."""
    return format_grouped_ids_tsv(
        id_map=predicted_matches,
        all_s1_ids=all_test_s1_ids,
        id_column_name="matched_entity_ids",
        output_path=output_path,
    )


def generate_candidate_pairs(
    candidate_map: Dict[str, Set[str]],
    all_test_s1_ids: List[str],
    output_path: Path = CANDIDATE_PAIRS_PATH,
) -> pd.DataFrame:
    """Write candidate_pairs.tsv with exactly [source1_entity_id, candidate_entity_ids]."""
    return format_grouped_ids_tsv(
        id_map=candidate_map,
        all_s1_ids=all_test_s1_ids,
        id_column_name="candidate_entity_ids",
        output_path=output_path,
    )


def validate_submission_file(
    filepath: Path,
    expected_s1_ids: Set[str],
    target_column_name: str,
) -> bool:
    """Strictly validate submission formatting constraints.

    Checks:
        1. File exists and is tab-separated.
        2. Expected column headers exist.
        3. Every expected Source 1 entity appears exactly once.
        4. No S1- prefixes inside the target match/candidate column.
        5. No duplicate IDs within the comma-separated target list.
        6. Only S2- and S3- prefixes present in matches.

    Raises:
        ValueError if any constraint is violated.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Submission file does not exist: {filepath}")

    df = pd.read_csv(filepath, sep="\t", dtype=str, keep_default_na=False)

    if df.shape[1] != 2:
        raise ValueError(
            f"Expected exactly 2 columns in {filepath.name}, found {df.shape[1]}"
        )

    expected_cols = ["source1_entity_id", target_column_name]
    if list(df.columns) != expected_cols:
        raise ValueError(
            f"Columns in {filepath.name} must be {expected_cols}, got {list(df.columns)}"
        )

    # Row count verification
    if len(df) != len(expected_s1_ids):
        raise ValueError(
            f"Row count mismatch in {filepath.name}: expected {len(expected_s1_ids)}, found {len(df)}"
        )

    # S1 uniqueness and completeness
    found_s1_ids = set(df["source1_entity_id"])
    if found_s1_ids != expected_s1_ids:
        missing = expected_s1_ids.difference(found_s1_ids)
        extra = found_s1_ids.difference(expected_s1_ids)
        raise ValueError(
            f"S1 entity mismatch in {filepath.name}. Missing {len(missing)}, Extra {len(extra)}"
        )

    # Value verification
    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"]
        raw_val = row[target_column_name]
        if not raw_val:
            continue

        items = [i.strip() for i in raw_val.split(",") if i.strip()]

        # Duplicate check
        if len(items) != len(set(items)):
            raise ValueError(
                f"Row for {s1_id} contains duplicate IDs in {target_column_name}: {items}"
            )

        for item in items:
            # Must not be S1
            if item.startswith(VALID_SOURCE_PREFIXES["source1"]):
                raise ValueError(
                    f"Row for {s1_id} contains invalid S1 ID in {target_column_name}: {item}"
                )
            # Must be S2 or S3
            if not (
                item.startswith(VALID_SOURCE_PREFIXES["source2"])
                or item.startswith(VALID_SOURCE_PREFIXES["source3"])
            ):
                raise ValueError(
                    f"Row for {s1_id} contains invalid prefix in {target_column_name}: {item}"
                )

    return True
