# Amazon ML Challenge 2026 — Business Entity Resolution (Baseline)

> **BASELINE IMPLEMENTATION**: This repository provides a clean, modular, leak-free, and reproducible baseline pipeline for the Amazon ML Challenge 2026 Business Entity Resolution challenge. It is designed for inspection, local debugging, and easy transfer to Google Colab.

---

## 1. Problem Overview

In this challenge, business records arrive from three independent sources:
- **Source 1 (`S1-`)**: Deduplicated reference source.
- **Source 2 (`S2-`)**: Noisy business records.
- **Source 3 (`S3-`)**: Noisy business records.

**Goal**: For every Source 1 entity, identify all corresponding records from Source 2 and/or Source 3 representing the same real-world business entity.

### Key Characteristics:
- A Source 1 entity can have **zero matches** (singleton), **one match**, or **multiple matches**.
- Matching target records can come from Source 2, Source 3, or both.
- Evaluation metric: **$F_{0.5}$ score** (precision-heavy; false merges are penalized heavily).

---

## 2. Dataset Structure

Place the official challenge dataset in TSV format into the following structure:

```
amazon-ml-challenge/
└── dataset/
    ├── train/
    │   ├── train_source1.tsv
    │   ├── train_source2.tsv
    │   ├── train_source3.tsv
    │   └── train_ground_truth.tsv
    └── test/
        ├── test_source1.tsv
        ├── test_source2.tsv
        └── test_source3.tsv
```

Each source file contains tab-delimited columns:
- `entity_id`: prefixed with `S1-`, `S2-`, or `S3-`
- `business_name`: raw business name
- `business_address`: raw business address
- `country`: country code / name

`train_ground_truth.tsv` contains:
- `source1_entity_id`: e.g. `S1-1001`
- `matched_entity_ids`: comma-separated matching target IDs, e.g. `S2-5012,S3-9182` (or empty for singletons)

---

## 3. Installation & Dependencies

Python 3.9+ is recommended. Install the lightweight dependencies:

```bash
cd amazon-ml-challenge
pip install -r requirements.txt
```

Core libraries used:
- `pandas`: Data loading and tabular operations
- `numpy`: Vectorized math and array operations
- `scikit-learn`: TF-IDF vectorization and Logistic Regression classifier
- `scipy`: Sparse matrix operations
- `python-Levenshtein`: Fast string edit distance calculation
- `joblib`: Model and vectorizer serialization
- `jupyter`: For EDA notebook execution

---

## 4. How to Run Exploration (EDA)

Launch Jupyter and open the exploratory notebook:

```bash
jupyter notebook notebooks/01_exploration.ipynb
```

The notebook dynamically computes:
- Row counts, missing value percentages, and duplicate ID checks.
- Country distributions across all 3 sources without hardcoded country names.
- Name and address length statistics.
- Ground-truth match distribution (singleton counts, 1-to-1, and 1-to-many matches).

---

## 5. How to Run the Pipeline

The pipeline is managed via `run_pipeline.py`.

### A. Training & Local Validation
Trains the baseline model, evaluates on an entity-split validation set, performs a threshold sweep for $F_{0.5}$, and saves model artifacts:

```bash
python run_pipeline.py --mode train
```

### B. Test Inference
Loads the trained model and fitted TF-IDF vectorizers, processes test records, and outputs final TSVs:

```bash
python run_pipeline.py --mode predict --threshold 0.70
```

### C. End-to-End Run (Train + Best Threshold Predict)
Executes training, finds the optimal $F_{0.5}$ threshold automatically on the validation set, and immediately generates test predictions:

```bash
python run_pipeline.py --mode full
```

---

## 6. How Validation Works & Data Leakage Prevention

### Entity-Level Split (Leak-Free)
A standard random split over candidate pairs causes severe **data leakage** because candidate pairs for the same Source 1 entity would appear in both train and validation sets, allowing the model to memorize entity-specific tokens.

To strictly avoid this:
1. The unique universe of `source1_entity_id` values is partitioned:
   - **80% Training S1 Entities**
   - **20% Validation S1 Entities**
2. Candidate pairs are filtered such that all pairs for validation S1 entities are **completely held out** from training.
3. Negative training examples are downsampled (default 5:1 ratio) only from the training candidate partition.

### $F_{0.5}$ Metric & Threshold Optimization
The competition evaluates with $F_{0.5}$:
$$F_{0.5} = \frac{(1 + 0.5^2) \cdot \text{Precision} \cdot \text{Recall}}{0.5^2 \cdot \text{Precision} + \text{Recall}} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$$

Because $F_{0.5}$ weights precision $4\times$ more heavily than recall, false merges (false positives) drastically degrade the score. During validation, `evaluation.py` performs a probability threshold sweep from `0.50` to `0.90` in steps of `0.05` to identify the cutoff that maximizes $F_{0.5}$.

---

## 7. How Candidate Generation (Blocking) Works

Candidate generation reduces the $O(N_1 \times (N_2 + N_3))$ search space while maximizing recall:
- **Country Partitioning**: S1 records are compared only against S2/S3 records in the same normalized country.
- **`NameTokenBlocker`**: Matches pairs sharing normalized name tokens with Jaccard similarity $\ge 0.30$ using inverted indexes.
- **`NameNgramBlocker`**: Matches pairs sharing character 3-grams with Jaccard $\ge 0.25$ (capturing minor typos and spelling variants).
- **`AddressTokenBlocker`**: Matches pairs sharing $\ge 3$ address tokens (capturing records where brand names differ but addresses match).
- The pipeline measures and logs **blocking recall** against the training ground truth:
  $$\text{Blocking Recall} = \frac{\text{True matches in candidate set}}{\text{Total true matches in ground truth}}$$

---

## 8. Feature Engineering

For each candidate pair, 18 pairwise features are extracted across names, addresses, and context:

| Category | Feature Name | Description | Range |
|---|---|---|---|
| **Name** | `name_exact_match` | Exact string equality indicator | $\{0, 1\}$ |
| | `name_levenshtein_sim` | Normalized Levenshtein similarity: $1 - \frac{\text{dist}}{\max(L_1, L_2)}$ | $[0, 1]$ |
| | `name_jaccard_token` | Word-level Jaccard similarity | $[0, 1]$ |
| | `name_token_overlap_ratio` | $\frac{\|A \cap B\|}{\min(\|A\|, \|B\|)}$ | $[0, 1]$ |
| | `name_char_jaccard_3gram` | Character 3-gram Jaccard similarity | $[0, 1]$ |
| | `name_common_prefix_len` | Longest common prefix ratio | $[0, 1]$ |
| | `name_length_diff` | Normalized relative length difference | $[0, 1]$ |
| | `name_tfidf_cosine` | Character $n$-gram TF-IDF cosine similarity | $[0, 1]$ |
| **Address** | `addr_exact_match` | Exact address string equality | $\{0, 1\}$ |
| | `addr_levenshtein_sim` | Normalized Levenshtein edit distance | $[0, 1]$ |
| | `addr_jaccard_token` | Word-level Jaccard similarity | $[0, 1]$ |
| | `addr_token_overlap_ratio` | Token overlap ratio on addresses | $[0, 1]$ |
| | `addr_char_jaccard_3gram` | Character 3-gram Jaccard similarity | $[0, 1]$ |
| | `addr_length_diff` | Normalized address length difference | $[0, 1]$ |
| | `addr_tfidf_cosine` | Address TF-IDF cosine similarity | $[0, 1]$ |
| **Context** | `country_match` | Country equality indicator | $\{0, 1\}$ |
| | `name_in_addr` | Business name contained in partner address | $\{0, 1\}$ |
| | `numeric_token_overlap` | Overlap of numeric tokens (zip codes, street numbers) | $[0, 1]$ |

---

## 9. Model & Predictions

- **Model**: `LogisticRegression(class_weight="balanced", solver="lbfgs")`.
- **Interpretability**: Inspects positive and negative feature coefficients to verify matching logic.
- **Probabilities**: Predicts calibrated match probability for each candidate pair; pairs meeting the chosen decision threshold are merged.

---

## 10. Output Formats

Outputs are written to the `output/` directory and verified against challenge specifications:

### `output/matching_results.tsv`
```tsv
source1_entity_id	matched_entity_ids
S1-00001	S2-04512,S3-01923
S1-00002	
S1-00003	S2-09812
```
- Every test Source 1 entity appears exactly once.
- Empty string for non-matches / singletons.
- Only valid `S2-` and `S3-` prefixes allowed.
- No duplicate IDs in matched lists.

### `output/candidate_pairs.tsv`
```tsv
source1_entity_id	candidate_entity_ids
S1-00001	S2-04512,S2-08812,S3-01923
S1-00002	S3-00122
S1-00003	S2-09812
```

---

## 11. Challenge Constraints Compliance

1. **No External Data**: No external APIs, knowledge graphs, or commercial business registries.
2. **No Geocoding / Web Scraping**: Purely data-driven resolution within provided text fields.
3. **No Hard-Coded Country Lists**: Accommodates any country code in the test set (e.g. France, Germany, US, India).
4. **ID Restrictions**: Source 1 IDs are never output as predicted matches; only valid `S2-` and `S3-` IDs are permitted.
5. **Singleton Awareness**: Accurately outputs empty match lists for singletons.
