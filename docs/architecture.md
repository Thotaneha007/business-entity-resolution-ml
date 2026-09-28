# System Architecture

## Overview

The project uses a multi-stage entity-resolution architecture designed to reduce the computational cost of matching large collections of noisy business records.

The pipeline separates the problem into two major stages:

1. **Candidate generation** — identify a manageable set of potentially matching records.
2. **Candidate classification** — compute pairwise features and use a machine-learning model to determine which candidates are likely matches.

## Pipeline

`	ext
Input S1 / S2 / S3 Records
          |
          v
+---------------------------+
| 1. Data Loading           |
| TSV records by country    |
+---------------------------+
          |
          v
+---------------------------+
| 2. Normalization          |
| Names / addresses / URL   |
| artifacts / country       |
+---------------------------+
          |
          v
+---------------------------+
| 3. Multi-Pass Blocking    |
| Multiple complementary    |
| candidate-generation keys |
+---------------------------+
          |
          v
+---------------------------+
| 4. Pairwise Features      |
| Name / address / country  |
| similarity features       |
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
| Probability >= threshold  |
+---------------------------+
          |
          v
Matched Entity Pairs
`

The architecture is intentionally modular: normalization prepares comparable representations, blocking controls the search space, feature engineering converts candidate pairs into numerical signals, and the classifier performs the final pair-level decision.


## 1. Normalization

Raw business records can contain inconsistent formatting, legal suffixes, punctuation, whitespace, URLs, and other artifacts. The normalization stage creates comparable representations before candidate generation and feature computation.

The normalization module produces structured representations for each entity, including normalized business-name and address forms and token-based representations used by downstream blocking and matching logic.

Key normalization operations include:

- Lowercasing text and standardizing whitespace.
- Removing or standardizing punctuation and formatting artifacts.
- Normalizing common legal-entity suffixes.
- Handling URL and domain-related artifacts in business names.
- Creating token-based representations for names and addresses.
- Preserving country information so that matching can be performed within the appropriate country partition.

Normalization is deliberately separated from matching so that the same representations can be reused by both the blocking and feature-engineering stages.


## 2. Multi-Pass Blocking

Comparing every source-1 entity against every source-2 and source-3 entity would require an impractical number of pairwise comparisons. The blocking stage therefore generates a smaller candidate set using several complementary retrieval passes.

The implementation uses seven blocking passes. Each pass uses a different combination of normalized name, address, token, and identifier-like information. The candidate sets produced by the passes are combined so that a potential match missed by one blocking rule can still be retrieved by another.

The blocking strategy is designed around two competing requirements:

- **High candidate recall:** true matches should survive the blocking stage.
- **Controlled candidate volume:** the number of pairs passed to the machine-learning matcher must remain computationally manageable.

After the multi-pass candidate generation, candidates are ranked and a maximum candidate limit is applied per source-1 entity. The production pipeline uses a cap of 35 candidates per source-1 entity.

This separation is important because blocking is a retrieval problem rather than the final matching decision. A candidate that survives blocking is only a possible match; the ML matcher makes the subsequent pair-level decision.


## 3. Pairwise Feature Engineering

Each blocked candidate pair is converted into a numerical feature vector before being passed to the classifier. The features capture complementary evidence from the business name, address, and country fields.

The feature-generation stage includes string-similarity and structural signals such as:

- Business-name similarity measures.
- Address similarity measures.
- Token overlap between normalized representations.
- Character-level similarity signals for noisy text.
- Length and structural differences between the compared fields.
- Country consistency information.

The purpose of combining multiple signals is to avoid relying on a single exact-match rule. For example, a business name may contain a typo or transliteration while the address remains highly similar, or the address may be incomplete while the business name provides strong evidence.

The resulting feature vector is supplied to the machine-learning matcher for pair-level classification.


## 4. Machine-Learning Matcher

The final matching decision is made by a supervised gradient-boosting classifier. The production implementation uses HistGradientBoostingClassifier from scikit-learn.

The matcher is trained on labeled entity pairs derived from the training data and their known match relationships. It learns how combinations of name, address, and structural similarity signals correspond to matching and non-matching entity pairs.

The production configuration uses:

- Model: HistGradientBoostingClassifier
- Maximum iterations: 150
- Learning rate: 0.1
- Maximum tree depth: 6
- Random state: 42

For each candidate pair, the classifier produces a probability representing the estimated likelihood that the pair is a true match. This probability is then used by the final decision stage.

Using a learned classifier rather than a single manually selected similarity threshold allows different evidence sources to be combined into a nonlinear decision function.


## 5. Thresholding and Output Generation

The classifier produces a match probability for each candidate pair. The production pipeline accepts a candidate as a predicted match when its probability is at least the configured decision threshold.

The production inference threshold used for the final run was **0.75**. This threshold controls the precision-recall trade-off at the pair-selection stage and was kept fixed throughout the production inference run.

The pipeline then aggregates the accepted matches for each source-1 entity and writes two outputs:

- matching_results.tsv — the predicted matched entity IDs for every test source-1 entity.
- candidate_pairs.tsv — the final candidate IDs presented to the matcher for every test source-1 entity.

The candidate output is retained separately because candidate generation and final matching are distinct stages. This makes the system easier to audit and allows candidate recall and prediction behavior to be analyzed independently.


## 6. Large-Scale Inference

The complete production run was executed on the full test population rather than on a reduced sample. The pipeline processed **1,732,544 source-1 test entities** across the three country partitions.

The final run produced:

| Metric | Result |
|---|---:|
| Test S1 entities processed | 1,732,544 |
| Total candidate pairs | 44,667,200 |
| Mean candidates per S1 | 25.8 |
| Median candidates per S1 | 35 |
| P95 candidates per S1 | 35 |
| Maximum candidates per S1 | 35 |
| Predicted matches | 8,552,099 |
| Mean predicted matches per S1 | 4.94 |
| Runtime | 42,342 seconds (~11.76 hours) |

The predicted matches consisted of **3,279,953 S2 records** and **5,272,146 S3 records**.

The final output contained predictions for 1,677,212 of the 1,732,544 source-1 entities, while 55,332 source-1 entities had no predicted matches. The production run therefore demonstrates that the pipeline can execute end-to-end over millions of records while keeping the per-entity candidate search bounded.


## 7. Design Rationale

The architecture separates retrieval from classification because the two stages have different objectives.

### Why normalization?

Business records are rarely standardized. Normalization reduces superficial differences before similarity calculations while retaining representations useful for different matching situations.

### Why multi-pass blocking?

A single blocking rule can miss valid matches when a particular field is noisy or incomplete. Multiple complementary passes improve the chance that true matches enter the candidate set without requiring an exhaustive comparison against every record.

### Why machine learning?

Different entity pairs can exhibit different combinations of evidence. A supervised classifier can learn interactions between name, address, and structural features instead of relying on one fixed similarity rule.

### Why keep candidate generation separate?

Candidate generation determines which records are eligible for classification, while the matcher determines which eligible records are predicted matches. Keeping these stages separate makes recall, precision, computational cost, and failure modes easier to analyze.


## 8. Reproducibility

The repository contains the core Python modules required to understand and reproduce the entity-resolution pipeline. The original challenge dataset is intentionally not included in the repository.

To run the pipeline with an authorized copy of the dataset, install the dependencies listed in requirements.txt and place the dataset in the expected local directory structure. The pipeline then performs normalization, country-level indexing, candidate generation, feature computation, classification, and output generation.

The generated submission files should remain outside version control because they are large derived artifacts. The repository is intended to contain the implementation, documentation, and methodology rather than the competition dataset or generated submission files.


## 9. Limitations

The system has several practical limitations.

- Blocking can only recover matches that enter the generated candidate set. A true match excluded during blocking cannot be recovered by the classifier.
- The fixed candidate cap limits computational cost but can remove lower-ranked candidates in dense or ambiguous regions.
- The classifier depends on the quality and representativeness of the labeled training pairs.
- No external business directory or web data is used, so the system relies entirely on the information available in the supplied datasets.
- The production threshold is a fixed operating point and may require recalibration for a different dataset or error-cost preference.
- The current pipeline is optimized around the challenge dataset layout and is not presented as a general-purpose entity-resolution framework for arbitrary schemas.

These limitations are useful directions for future work, including adaptive candidate budgets, stronger hard-negative mining, model calibration, and more systematic error analysis.

