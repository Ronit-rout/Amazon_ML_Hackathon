"""Candidate generation (blocking) module to reduce search space while maximizing recall."""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import pandas as pd

from src.config import (
    MIN_BLOCKING_TOKEN_JACCARD,
    BLOCKING_NGRAM_SIZE,
    MIN_BLOCKING_NGRAM_JACCARD,
)


def get_tokens(text: str) -> Set[str]:
    """Tokenize normalized text into a set of whitespace-separated words."""
    if not text:
        return set()
    return set(text.split())


def get_char_ngrams(text: str, n: int = BLOCKING_NGRAM_SIZE) -> Set[str]:
    """Extract character n-grams from normalized text."""
    clean_text = "".join(text.split())
    if len(clean_text) < n:
        return {clean_text} if clean_text else set()
    return {clean_text[i : i + n] for i in range(len(clean_text) - n + 1)}


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection_len = len(set_a.intersection(set_b))
    if intersection_len == 0:
        return 0.0
    union_len = len(set_a.union(set_b))
    return intersection_len / union_len if union_len > 0 else 0.0


class BlockingStrategy(ABC):
    """Abstract base class for modular candidate generation strategies."""

    @abstractmethod
    def generate_pairs(
        self, s1_df: pd.DataFrame, target_df: pd.DataFrame
    ) -> Set[Tuple[str, str]]:
        """Generate candidate pairs as a set of (source1_id, target_id) tuples."""
        pass


class NameTokenBlocker(BlockingStrategy):
    """Blocks entities within the same country sharing name tokens above a Jaccard threshold."""

    def __init__(self, min_jaccard: float = MIN_BLOCKING_TOKEN_JACCARD):
        self.min_jaccard = min_jaccard

    def generate_pairs(
        self, s1_df: pd.DataFrame, target_df: pd.DataFrame
    ) -> Set[Tuple[str, str]]:
        pairs: Set[Tuple[str, str]] = set()

        # Group by country to avoid cross-country false comparisons
        countries = set(s1_df["country_normalized"].unique()).intersection(
            set(target_df["country_normalized"].unique())
        )

        for country in countries:
            sub_s1 = s1_df[s1_df["country_normalized"] == country]
            sub_target = target_df[target_df["country_normalized"] == country]

            # Inverted index on target name tokens
            inverted_index: Dict[str, List[Tuple[str, Set[str]]]] = defaultdict(list)
            for _, row in sub_target.iterrows():
                t_id = row["entity_id"]
                tokens = get_tokens(row["business_name_normalized"])
                if not tokens:
                    continue
                for tok in tokens:
                    inverted_index[tok].append((t_id, tokens))

            # Query with S1
            for _, row in sub_s1.iterrows():
                s1_id = row["entity_id"]
                s1_tokens = get_tokens(row["business_name_normalized"])
                if not s1_tokens:
                    continue

                seen_targets: Set[str] = set()
                for tok in s1_tokens:
                    for target_id, target_tokens in inverted_index.get(tok, []):
                        if target_id in seen_targets:
                            continue
                        seen_targets.add(target_id)
                        sim = jaccard_similarity(s1_tokens, target_tokens)
                        if sim >= self.min_jaccard:
                            pairs.add((s1_id, target_id))

        return pairs


class NameNgramBlocker(BlockingStrategy):
    """Blocks entities within the same country sharing character n-grams.
    Catches typos, prefixes, and slight name spelling differences.
    """

    def __init__(
        self,
        ngram_size: int = BLOCKING_NGRAM_SIZE,
        min_jaccard: float = MIN_BLOCKING_NGRAM_JACCARD,
    ):
        self.ngram_size = ngram_size
        self.min_jaccard = min_jaccard

    def generate_pairs(
        self, s1_df: pd.DataFrame, target_df: pd.DataFrame
    ) -> Set[Tuple[str, str]]:
        pairs: Set[Tuple[str, str]] = set()

        countries = set(s1_df["country_normalized"].unique()).intersection(
            set(target_df["country_normalized"].unique())
        )

        for country in countries:
            sub_s1 = s1_df[s1_df["country_normalized"] == country]
            sub_target = target_df[target_df["country_normalized"] == country]

            inverted_index: Dict[str, List[Tuple[str, Set[str]]]] = defaultdict(list)
            for _, row in sub_target.iterrows():
                t_id = row["entity_id"]
                ngrams = get_char_ngrams(
                    row["business_name_normalized"], self.ngram_size
                )
                if not ngrams:
                    continue
                for ng in ngrams:
                    inverted_index[ng].append((t_id, ngrams))

            for _, row in sub_s1.iterrows():
                s1_id = row["entity_id"]
                s1_ngrams = get_char_ngrams(
                    row["business_name_normalized"], self.ngram_size
                )
                if not s1_ngrams:
                    continue

                seen_targets: Set[str] = set()
                for ng in s1_ngrams:
                    for target_id, target_ngrams in inverted_index.get(ng, []):
                        if target_id in seen_targets:
                            continue
                        seen_targets.add(target_id)
                        sim = jaccard_similarity(s1_ngrams, target_ngrams)
                        if sim >= self.min_jaccard:
                            pairs.add((s1_id, target_id))

        return pairs


class AddressTokenBlocker(BlockingStrategy):
    """Blocks entities within the same country sharing informative address tokens.
    Catches records with different brand variations but identical street addresses.
    """

    def __init__(self, min_token_overlap: int = 3):
        self.min_token_overlap = min_token_overlap

    def generate_pairs(
        self, s1_df: pd.DataFrame, target_df: pd.DataFrame
    ) -> Set[Tuple[str, str]]:
        pairs: Set[Tuple[str, str]] = set()

        countries = set(s1_df["country_normalized"].unique()).intersection(
            set(target_df["country_normalized"].unique())
        )

        for country in countries:
            sub_s1 = s1_df[s1_df["country_normalized"] == country]
            sub_target = target_df[target_df["country_normalized"] == country]

            inverted_index: Dict[str, List[Tuple[str, Set[str]]]] = defaultdict(list)
            for _, row in sub_target.iterrows():
                t_id = row["entity_id"]
                tokens = get_tokens(row["business_address_normalized"])
                if len(tokens) < self.min_token_overlap:
                    continue
                for tok in tokens:
                    inverted_index[tok].append((t_id, tokens))

            for _, row in sub_s1.iterrows():
                s1_id = row["entity_id"]
                s1_tokens = get_tokens(row["business_address_normalized"])
                if len(s1_tokens) < self.min_token_overlap:
                    continue

                seen_targets: Dict[str, int] = defaultdict(int)
                for tok in s1_tokens:
                    for target_id, _ in inverted_index.get(tok, []):
                        seen_targets[target_id] += 1

                for target_id, count in seen_targets.items():
                    if count >= self.min_token_overlap:
                        pairs.add((s1_id, target_id))

        return pairs


def generate_candidate_pairs(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    strategies: List[BlockingStrategy] = None,
) -> pd.DataFrame:
    """Generate candidate pairs between S1 <-> S2 and S1 <-> S3.

    Args:
        s1_df: Normalized Source 1 dataframe.
        s2_df: Normalized Source 2 dataframe.
        s3_df: Normalized Source 3 dataframe.
        strategies: List of blocking strategies to union.

    Returns:
        DataFrame with columns ['source1_entity_id', 'candidate_entity_id'].
    """
    if strategies is None:
        strategies = [
            NameTokenBlocker(min_jaccard=MIN_BLOCKING_TOKEN_JACCARD),
            NameNgramBlocker(
                ngram_size=BLOCKING_NGRAM_SIZE,
                min_jaccard=MIN_BLOCKING_NGRAM_JACCARD,
            ),
            AddressTokenBlocker(min_token_overlap=3),
        ]

    all_pairs: Set[Tuple[str, str]] = set()

    for strategy in strategies:
        pairs_s2 = strategy.generate_pairs(s1_df, s2_df)
        pairs_s3 = strategy.generate_pairs(s1_df, s3_df)
        all_pairs.update(pairs_s2)
        all_pairs.update(pairs_s3)

    if not all_pairs:
        return pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id"]
        )

    records = [
        {"source1_entity_id": s1, "candidate_entity_id": cand}
        for s1, cand in sorted(all_pairs)
    ]
    return pd.DataFrame(records)


def compute_blocking_recall(
    candidates_df: pd.DataFrame,
    ground_truth_map: Dict[str, Set[str]],
) -> Dict[str, float]:
    """Compute candidate generation recall against ground truth matches.

    Args:
        candidates_df: DataFrame containing ['source1_entity_id', 'candidate_entity_id'].
        ground_truth_map: Dict mapping s1_id -> set of true matched IDs.

    Returns:
        Dict with total_true_matches, found_true_matches, and blocking_recall.
    """
    candidate_lookup: Dict[str, Set[str]] = defaultdict(set)
    for _, row in candidates_df.iterrows():
        candidate_lookup[row["source1_entity_id"]].add(row["candidate_entity_id"])

    total_true = 0
    found_true = 0

    for s1_id, true_matches in ground_truth_map.items():
        total_true += len(true_matches)
        cand_matches = candidate_lookup.get(s1_id, set())
        found_true += len(true_matches.intersection(cand_matches))

    recall = (found_true / total_true) if total_true > 0 else 1.0

    return {
        "total_true_matches": float(total_true),
        "found_true_matches": float(found_true),
        "blocking_recall": float(recall),
    }
