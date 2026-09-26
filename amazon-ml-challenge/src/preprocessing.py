"""Text normalization and preprocessing module for entity records."""

import re
import unicodedata
from typing import Optional
import pandas as pd


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters using NFKD decomposition."""
    if not text:
        return ""
    return unicodedata.normalize("NFKD", str(text))


def strip_accents(text: str) -> str:
    """Remove diacritics/accents from characters while preserving base letters."""
    if not text:
        return ""
    normalized = normalize_unicode(text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def normalize_whitespace(text: str) -> str:
    """Collapse consecutive whitespace and strip edges."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_general_text(text: Optional[str]) -> str:
    """Conservative general text normalization:
    - Handle null/empty
    - Strip accents & unicode normalize
    - Lowercase
    - Replace common punctuation delimiters with space, while retaining & and #
    - Collapse extra whitespace
    """
    if text is None or pd.isna(text):
        return ""

    val = str(text)
    val = strip_accents(val).lower()

    # Replace punctuation (except alphanumeric, spaces, and useful tokens like &, #) with space
    # Keeps letters, digits, whitespace, &, #
    val = re.sub(r"[^\w\s&#]", " ", val)

    # Standardize & symbol spacing
    val = re.sub(r"\s*&\s*", " & ", val)

    return normalize_whitespace(val)


def normalize_business_name(name: Optional[str]) -> str:
    """Conservative normalization specific to business names."""
    return normalize_general_text(name)


def normalize_business_address(address: Optional[str]) -> str:
    """Conservative normalization specific to business addresses."""
    return normalize_general_text(address)


def normalize_country(country: Optional[str]) -> str:
    """Normalize country code/name."""
    if country is None or pd.isna(country):
        return ""
    return strip_accents(str(country)).lower().strip()


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized columns without mutating or dropping original data.

    Adds:
        - business_name_normalized
        - business_address_normalized
        - country_normalized

    Returns:
        A shallow copy of the dataframe with normalized columns.
    """
    df_out = df.copy()

    df_out["business_name_normalized"] = df_out["business_name"].apply(
        normalize_business_name
    )
    df_out["business_address_normalized"] = df_out["business_address"].apply(
        normalize_business_address
    )
    df_out["country_normalized"] = df_out["country"].apply(normalize_country)

    return df_out
