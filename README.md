# Business Entity Resolution using Machine Learning

A scalable machine-learning pipeline for resolving noisy business records across multiple data sources and identifying records that refer to the same real-world business entity.

This project combines text and address normalization, multi-pass blocking, candidate generation, pairwise similarity features, and supervised machine learning to perform large-scale entity resolution.

## Overview

Business Entity Resolution (ER) is the task of determining whether records from different data sources refer to the same real-world entity.

The system was designed for noisy business records containing:

- Business names
- Business addresses
- Country information
- Source-specific entity identifiers

The data contains variations such as:

- Spelling differences and typographical errors
- Legal-suffix variations
- Multilingual names and transliteration
- Missing or incomplete addresses
- Address formatting differences
- Different address components or ordering
- Numeric variations
- Web-domain and handle artifacts

The core challenge is scalability: comparing every Source 1 record against every Source 2 and Source 3 record would require an impractical number of pairwise comparisons.

The solution therefore separates the problem into two stages:

1. **Candidate generation** — efficiently retrieve a manageable set of potentially matching records.
2. **Candidate classification** — use pairwise features and a machine-learning model to determine which candidates are likely matches.

---

## System Architecture

```text
Raw Business Records
        |
        v
+---------------------------+
| 1. Normalization          |
| Names / addresses / URL   |
| artifacts / country       |
+---------------------------+
        |
        v
+---------------------------+
| 2. Multi-Pass Blocking    |
| Seven complementary      |
| candidate-generation      |
| passes                    |
+---------------------------+
        |
        v
+---------------------------+
| 3. Candidate Generation   |
| Ranking + top-35 cap      |
+---------------------------+
        |
        v
+---------------------------+
| 4. Pairwise Features      |
| Name / address / country  |
| similarity signals        |
+---------------------------+
        |
        v
+---------------------------+
| 5. ML Matcher             |
| HistGradientBoosting      |
+---------------------------+
        |
        v
+---------------------------+
| 6. Threshold Filtering    |
| Probability >= 0.75      |
+---------------------------+
        |
        v
Matched Entity Pairs
```

Detailed documentation:

- [System Architecture](docs/architecture.md)
- [Methodology](docs/methodology.md)
- [Experiments](docs/experiments.md)

---

## Data Processing and Normalization

The normalization stage creates multiple representations of each business record rather than relying on a single normalized string.

It handles:

- Unicode normalization
- Case normalization
- Punctuation and whitespace normalization
- Legal business suffixes
- Common stopwords
- Business-name tokenization
- Address tokenization
- Address numbers
- Character-level representations
- Domain and handle artifacts
- Country-aware processing

The normalized representations are reused by both blocking and feature engineering.

Implementation:

`src/normalization.py`

---

## Multi-Pass Blocking

Blocking is the key scalability component of the system.

Instead of comparing every Source 1 record against every Source 2 and Source 3 record, multiple complementary blocking rules retrieve potentially relevant candidates.

The production implementation uses **seven blocking passes** based on combinations of normalized:

- Business-name cores
- Name tokens
- Address information
- Address numbers
- Domain-like representations
- Rare or distinctive tokens
- Other structured signals

Candidate sets from the different passes are combined and then ranked.

A maximum of **35 candidates per Source 1 entity** is retained for production inference.

This separation is important:

> Blocking determines which records are eligible for matching; the ML classifier makes the final pair-level decision.

Implementation:

`src/blocking.py`

---

## Candidate Generation Experiments

Several blocking strategies were evaluated during development.

| Strategy | Recall | Average Candidates |
|---|---:|---:|
| Exact core | 53.91% | 6.8 |
| Distinctive token | 60.66% | 63.1 |
| Name token + address number | 65.86% | 25.6 |
| Multipass union | 91.63% | 58.7 |
| Safe pruning / top 30 | 90.90% | 17.1 |

The final production approach used multi-pass candidate generation followed by candidate ranking and a top-35 candidate cap.

This provided a practical balance between candidate recall and computational cost.

---

## Pairwise Feature Engineering

For every generated candidate pair, the system computes numerical similarity and structural features.

### Business Name

- Normalized equality
- Character similarity
- Token overlap
- Fuzzy similarity
- Token-set comparisons
- Character n-gram signals

### Address

- Normalized address similarity
- Token overlap
- Address-number agreement
- Address-component similarity
- Missing-address indicators

### Structural Signals

- Country consistency
- Name/address relationships
- Candidate-source information
- Other pairwise similarity signals

Implementation:

`src/features.py`

---

## Machine Learning Matcher

The final pair classifier uses:

**HistGradientBoostingClassifier** from scikit-learn.

Production configuration:

| Parameter | Value |
|---|---:|
| max_iter | 150 |
| learning_rate | 0.1 |
| max_depth | 6 |
| random_state | 42 |
| Matching threshold | 0.75 |

The classifier produces a probability for each candidate pair.

Pairs with predicted probability greater than or equal to **0.75** are treated as predicted matches.

Implementation:

`src/matcher.py`

---

## Threshold Experiment

A development validation experiment evaluated several probability thresholds.

| Threshold | Precision | Recall | F0.5 |
|---:|---:|---:|---:|
| 0.70 | 0.2707 | 1.0000 | 0.3170 |
| 0.75 | 0.2890 | 1.0000 | 0.3369 |
| 0.80 | 0.3119 | 1.0000 | 0.3617 |
| 0.85 | 0.3381 | 1.0000 | 0.3897 |
| 0.90 | 0.3861 | 1.0000 | 0.4401 |
| 0.95 | 0.4820 | 0.9998 | 0.5377 |

These values come from a **development validation experiment**, not the competition leaderboard.

The production pipeline used a threshold of **0.75**.

---

## Large-Scale Production Run

The complete production inference was successfully executed on the full test set.

| Metric | Result |
|---|---:|
| Test S1 records | 1,732,544 |
| Candidate pairs | 44,667,200 |
| Mean candidates / S1 | 25.8 |
| Median candidates / S1 | 35 |
| P95 candidates / S1 | 35 |
| Maximum candidates / S1 | 35 |
| Predicted matches | 8,552,099 |
| S2 predicted matches | 3,279,953 |
| S3 predicted matches | 5,272,146 |
| Mean predictions / S1 | 4.94 |
| Zero-prediction S1 records | 55,332 |
| Runtime | ~11.76 hours |

The test set was processed country-by-country and in batches to control memory usage.

The country partitions processed were:

- France
- India
- United States

The full run generated approximately **44.7 million candidate pairs** and produced approximately **8.55 million predicted matches**.

---

## Output Validation

The generated outputs were checked using the official challenge validation utility.

### `matching_results.tsv`

- 1,732,544 rows
- 55,332 empty prediction rows
- 1,677,212 non-empty prediction rows

### `candidate_pairs.tsv`

- 1,732,544 rows
- 2,718 empty candidate rows
- 1,729,826 non-empty candidate rows

The official validator reported **PASS** for the submission structure.

---

## Dataset

The competition dataset is **not included in this repository**.

The original challenge data contained millions of records across Source 1, Source 2, and Source 3 and was distributed through the official challenge resources.

To reproduce the project, obtain an authorized copy of the challenge data through the official challenge distribution and place it in the expected local dataset structure.

Do not commit the challenge dataset or generated submission files to this repository.

---

## Reproducibility

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

The core dependencies are:

```text
pandas
numpy
scikit-learn
rapidfuzz
```

The core pipeline performs:

```text
Dataset loading
      |
      v
Country partitioning
      |
      v
Normalization
      |
      v
Inverted-index construction
      |
      v
Multi-pass candidate generation
      |
      v
Candidate ranking / pruning
      |
      v
Pairwise feature computation
      |
      v
HistGradientBoosting classification
      |
      v
Threshold filtering
      |
      v
Output generation
```

The exact challenge dataset layout and files are intentionally not distributed with this repository.

---

## Repository Structure

```text
business-entity-resolution-ml/
|
├── src/
│   ├── __init__.py
│   ├── blocking.py
│   ├── evaluation.py
│   ├── features.py
│   ├── matcher.py
│   ├── normalization.py
│   └── pipeline.py
│
├── docs/
│   ├── architecture.md
│   ├── methodology.md
│   └── experiments.md
│
├── examples/
│   └── README.md
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Technologies

- Python
- Pandas
- NumPy
- scikit-learn
- RapidFuzz
- Machine Learning
- Information Retrieval
- Entity Resolution
- Inverted Indexing
- Fuzzy String Matching
- Batch Processing

---

## Competition Context

This project was developed in the context of the **Amazon ML Challenge 2026**.

The challenge required identifying matching Source 2 and Source 3 records for every Source 1 business entity using noisy business names, addresses, and country information.

The evaluation metric was **macro F0.5**, a precision-heavy metric that places greater emphasis on avoiding false merges.

The repository focuses on the underlying entity-resolution and machine-learning engineering rather than distributing the competition dataset or generated submission artifacts.

---

## Official Challenge Resources

### Student Resources

[Amazon ML Challenge 2026 — Student Resources](https://cdn.unstop.com/files/6ab10eb3b23ba_student_resource.zip)

### Challenge Video

[Amazon ML Challenge 2026 — Challenge Video](https://d8it4huxumps7.cloudfront.net/files/6ab509c5b7036_ml_challenge_2026_video.mp4)

### Guidelines

[Amazon ML Challenge 2026 — Guidelines](https://d8it4huxumps7.cloudfront.net/files/6ab56657b4f1a_guidelines_and_key_instructions_amazon_ml_challenge_2026.pdf)

### Challenge Communications

[Amazon ML Challenge 2026 — Communications](https://d8it4huxumps7.cloudfront.net/files/6ab674645103d_emails_comms_amazon_ml_challenge_2026.pdf)

---

## Limitations

The current system has several practical limitations:

- Blocking can only recover matches that enter the candidate set.
- The fixed candidate cap can remove lower-ranked candidates in dense or ambiguous cases.
- Broad name-based blocking can produce many candidates when businesses share similar names.
- The classifier depends on the quality and representativeness of the training pairs.
- The production threshold is a fixed operating point.
- The current implementation is optimized around the challenge dataset structure.

One inspected production case showed that records with highly similar business names but different addresses could survive blocking and receive predictions. This indicates that stronger address-aware discrimination and hard-negative training would be useful future improvements.

---

## Future Improvements

Potential improvements include:

- Stronger hard-negative mining
- Better probability calibration
- Adaptive candidate budgets instead of a fixed top-K
- More address-aware candidate ranking
- Improved multilingual and transliteration handling
- More systematic error analysis
- Model comparison and ablation studies
- More efficient indexing for larger datasets
- Confidence-aware matching thresholds
- A lightweight reproducible demonstration using synthetic or sample data

---

## Author

**Thota Neha**

GitHub: [Thotaneha007](https://github.com/Thotaneha007)
