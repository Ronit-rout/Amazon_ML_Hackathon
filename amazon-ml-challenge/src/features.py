"""Feature engineering module for candidate entity pairs."""

import re
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import TFIDF_MAX_FEATURES

# Attempt fast Levenshtein, fallback to standard dynamic programming if not installed
try:
    import Levenshtein

    def levenshtein_distance(s1: str, s2: str) -> int:
        return Levenshtein.distance(s1, s2)

except ImportError:

    def levenshtein_distance(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Normalized Levenshtein similarity in [0.0, 1.0]."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    dist = levenshtein_distance(s1, s2)
    return max(0.0, 1.0 - (dist / max_len))


def extract_tokens(text: str) -> Set[str]:
    if not text:
        return set()
    return set(text.split())


def extract_char_ngrams(text: str, n: int = 3) -> Set[str]:
    clean = "".join(text.split())
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[i : i + n] for i in range(len(clean) - n + 1)}


def token_jaccard(tokens1: Set[str], tokens2: Set[str]) -> float:
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    return intersection / union if union > 0 else 0.0


def token_overlap_ratio(tokens1: Set[str], tokens2: Set[str]) -> float:
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1.intersection(tokens2))
    min_len = min(len(tokens1), len(tokens2))
    return intersection / min_len if min_len > 0 else 0.0


def common_prefix_ratio(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    min_len = min(len(s1), len(s2))
    max_len = max(len(s1), len(s2))
    prefix_len = 0
    while prefix_len < min_len and s1[prefix_len] == s2[prefix_len]:
        prefix_len += 1
    return prefix_len / max_len if max_len > 0 else 0.0


def extract_numeric_tokens(text: str) -> Set[str]:
    if not text:
        return set()
    return set(re.findall(r"\b\d+\b", text))


FEATURE_NAMES: List[str] = [
    # Name features
    "name_exact_match",
    "name_levenshtein_sim",
    "name_jaccard_token",
    "name_token_overlap_ratio",
    "name_char_jaccard_3gram",
    "name_common_prefix_len",
    "name_length_diff",
    "name_tfidf_cosine",
    # Address features
    "addr_exact_match",
    "addr_levenshtein_sim",
    "addr_jaccard_token",
    "addr_token_overlap_ratio",
    "addr_char_jaccard_3gram",
    "addr_length_diff",
    "addr_tfidf_cosine",
    # Cross / contextual features
    "country_match",
    "name_in_addr",
    "numeric_token_overlap",
]


class FeaturePipeline:
    """Manages TF-IDF vectorizers and extracts tabular features for candidate pairs."""

    def __init__(self, max_features: int = TFIDF_MAX_FEATURES):
        self.max_features = max_features
        self.name_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=max_features,
            dtype=np.float32,
        )
        self.addr_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=max_features,
            dtype=np.float32,
        )
        self.is_fitted = False

    def fit_vectorizers(self, all_names: List[str], all_addresses: List[str]) -> None:
        """Fit character n-gram TF-IDF vectorizers on corpus texts."""
        self.name_vectorizer.fit(all_names)
        self.addr_vectorizer.fit(all_addresses)
        self.is_fitted = True

    def compute_pair_features(
        self,
        s1_name: str,
        s1_addr: str,
        s1_country: str,
        cand_name: str,
        cand_addr: str,
        cand_country: str,
        name_tfidf_sim: float = 0.0,
        addr_tfidf_sim: float = 0.0,
    ) -> List[float]:
        """Compute the feature vector for a single pair of records."""
        # Name metrics
        name_exact = 1.0 if s1_name == cand_name and s1_name != "" else 0.0
        name_lev = levenshtein_similarity(s1_name, cand_name)
        s1_name_tok = extract_tokens(s1_name)
        cand_name_tok = extract_tokens(cand_name)
        name_jacc = token_jaccard(s1_name_tok, cand_name_tok)
        name_overlap = token_overlap_ratio(s1_name_tok, cand_name_tok)
        name_ngram_jacc = token_jaccard(
            extract_char_ngrams(s1_name, 3), extract_char_ngrams(cand_name, 3)
        )
        name_prefix = common_prefix_ratio(s1_name, cand_name)
        name_len_diff = (
            abs(len(s1_name) - len(cand_name)) / max(len(s1_name), len(cand_name), 1)
        )

        # Address metrics
        addr_exact = 1.0 if s1_addr == cand_addr and s1_addr != "" else 0.0
        addr_lev = levenshtein_similarity(s1_addr, cand_addr)
        s1_addr_tok = extract_tokens(s1_addr)
        cand_addr_tok = extract_tokens(cand_addr)
        addr_jacc = token_jaccard(s1_addr_tok, cand_addr_tok)
        addr_overlap = token_overlap_ratio(s1_addr_tok, cand_addr_tok)
        addr_ngram_jacc = token_jaccard(
            extract_char_ngrams(s1_addr, 3), extract_char_ngrams(cand_addr, 3)
        )
        addr_len_diff = (
            abs(len(s1_addr) - len(cand_addr)) / max(len(s1_addr), len(cand_addr), 1)
        )

        # Context metrics
        country_match = 1.0 if s1_country == cand_country and s1_country != "" else 0.0
        name_in_addr = (
            1.0
            if (s1_name and s1_name in cand_addr) or (cand_name and cand_name in s1_addr)
            else 0.0
        )
        num_overlap = token_jaccard(
            extract_numeric_tokens(s1_addr), extract_numeric_tokens(cand_addr)
        )

        return [
            name_exact,
            name_lev,
            name_jacc,
            name_overlap,
            name_ngram_jacc,
            name_prefix,
            name_len_diff,
            float(name_tfidf_sim),
            addr_exact,
            addr_lev,
            addr_jacc,
            addr_overlap,
            addr_ngram_jacc,
            addr_len_diff,
            float(addr_tfidf_sim),
            country_match,
            name_in_addr,
            num_overlap,
        ]

    def extract_features(
        self,
        candidates_df: pd.DataFrame,
        s1_df: pd.DataFrame,
        target_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Batch extract feature matrix for candidate pairs.

        Args:
            candidates_df: DataFrame with ['source1_entity_id', 'candidate_entity_id'].
            s1_df: Normalized Source 1 dataframe.
            target_df: Combined normalized target dataframe (S2 and S3).

        Returns:
            DataFrame with FEATURE_NAMES columns aligned with candidate rows.
        """
        if candidates_df.empty:
            return pd.DataFrame(columns=FEATURE_NAMES)

        # Index records by entity_id for O(1) lookup
        s1_lookup = s1_df.set_index("entity_id").to_dict("index")
        target_lookup = target_df.set_index("entity_id").to_dict("index")

        # Compute TF-IDF matrices if fitted
        name_tfidf_dict = {}
        addr_tfidf_dict = {}

        if self.is_fitted:
            unique_s1_ids = candidates_df["source1_entity_id"].unique()
            unique_target_ids = candidates_df["candidate_entity_id"].unique()

            # Pre-transform text for candidate entities only
            s1_names = [s1_lookup[i]["business_name_normalized"] for i in unique_s1_ids if i in s1_lookup]
            s1_addrs = [s1_lookup[i]["business_address_normalized"] for i in unique_s1_ids if i in s1_lookup]
            t_names = [target_lookup[i]["business_name_normalized"] for i in unique_target_ids if i in target_lookup]
            t_addrs = [target_lookup[i]["business_address_normalized"] for i in unique_target_ids if i in target_lookup]

            s1_name_mat = self.name_vectorizer.transform(s1_names)
            s1_addr_mat = self.addr_vectorizer.transform(s1_addrs)
            t_name_mat = self.name_vectorizer.transform(t_names)
            t_addr_mat = self.addr_vectorizer.transform(t_addrs)

            s1_name_idx_map = {eid: idx for idx, eid in enumerate(unique_s1_ids)}
            s1_addr_idx_map = {eid: idx for idx, eid in enumerate(unique_s1_ids)}
            t_name_idx_map = {eid: idx for idx, eid in enumerate(unique_target_ids)}
            t_addr_idx_map = {eid: idx for idx, eid in enumerate(unique_target_ids)}

        feature_rows: List[List[float]] = []

        for _, row in candidates_df.iterrows():
            s1_id = row["source1_entity_id"]
            cand_id = row["candidate_entity_id"]

            s1_rec = s1_lookup.get(s1_id, {})
            cand_rec = target_lookup.get(cand_id, {})

            s1_name = s1_rec.get("business_name_normalized", "")
            s1_addr = s1_rec.get("business_address_normalized", "")
            s1_country = s1_rec.get("country_normalized", "")

            cand_name = cand_rec.get("business_name_normalized", "")
            cand_addr = cand_rec.get("business_address_normalized", "")
            cand_country = cand_rec.get("country_normalized", "")

            name_sim = 0.0
            addr_sim = 0.0

            if self.is_fitted and s1_id in s1_name_idx_map and cand_id in t_name_idx_map:
                v1 = s1_name_mat[s1_name_idx_map[s1_id]]
                v2 = t_name_mat[t_name_idx_map[cand_id]]
                # Dot product of unit-normalized TF-IDF rows
                name_sim = float(v1.dot(v2.T).toarray()[0][0])

            if self.is_fitted and s1_id in s1_addr_idx_map and cand_id in t_addr_idx_map:
                a1 = s1_addr_mat[s1_addr_idx_map[s1_id]]
                a2 = t_addr_mat[t_addr_idx_map[cand_id]]
                addr_sim = float(a1.dot(a2.T).toarray()[0][0])

            row_feats = self.compute_pair_features(
                s1_name,
                s1_addr,
                s1_country,
                cand_name,
                cand_addr,
                cand_country,
                name_tfidf_sim=name_sim,
                addr_tfidf_sim=addr_sim,
            )
            feature_rows.append(row_feats)

        return pd.DataFrame(feature_rows, columns=FEATURE_NAMES)
