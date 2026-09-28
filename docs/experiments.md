# Experiments and Results

This document records the main experiments performed while developing the business entity-resolution pipeline. The experiments focus on candidate-generation quality, model behavior, threshold selection, and large-scale inference.

## 1. Experimental Goals

The development process focused on four main questions:

1. How can candidate generation retain true matches while keeping the candidate set manageable?
2. Which combination of blocking strategies provides useful candidate recall?
3. How should the machine-learning decision threshold be selected?
4. Can the final pipeline process the complete test population within practical computational limits?

## 2. Blocking Strategy Experiments

Candidate generation was evaluated by measuring how many known training matches were recovered by each blocking strategy and how many candidates were produced per Source 1 entity.

The main benchmark results were:

| Strategy | Candidate Recall | Avg. Candidates / S1 |
|---|---:|---:|
| Exact core | 53.91% | 6.8 |
| Distinctive token | 60.66% | 63.1 |
| Name token + address number | 65.86% | 25.6 |
| Multipass union | 91.63% | 58.7 |
| Safe pruning / top 30 | 90.90% | 17.1 |

The results show the central trade-off in blocking: stricter rules produce fewer candidates but can miss more true matches, while broader retrieval improves candidate recall at the cost of a larger downstream workload.

The final implementation uses a multi-pass blocking design followed by candidate ranking and a production cap of 35 candidates per Source 1 entity. This preserves the benefits of complementary retrieval rules while bounding the computational cost of pairwise feature computation and classification.


## 3. Threshold Experiment

A threshold sweep was performed during development to examine how the classifier operating point affected precision, recall, and F0.5 on the validation setup.

| Threshold | Precision | Recall | F0.5 |
|---:|---:|---:|---:|
| 0.70 | 0.2707 | 1.0000 | 0.3170 |
| 0.75 | 0.2890 | 1.0000 | 0.3369 |
| 0.80 | 0.3119 | 1.0000 | 0.3617 |
| 0.85 | 0.3381 | 1.0000 | 0.3897 |
| 0.90 | 0.3861 | 1.0000 | 0.4401 |
| 0.95 | 0.4820 | 0.9998 | 0.5377 |

These measurements show that increasing the decision threshold reduced the number of accepted false-positive candidates in this validation setup while retaining almost all of the observed recall.

The experiment should be interpreted as a development validation result rather than as the competition leaderboard score. The final production inference run retained the previously configured threshold of **0.75**.


## 4. Final Production Run

After evaluating the candidate-generation and threshold behavior, the complete pipeline was executed on the full test dataset.

The production configuration used the multi-pass blocking strategy, a maximum of 35 candidates per Source 1 entity, the HistGradientBoostingClassifier configuration documented in the methodology, and a probability threshold of **0.75**.

The full run processed **1,732,544 Source 1 entities** and generated **44,667,200 candidate pairs**.

The resulting predictions contained:

| Metric | Result |
|---|---:|
| Source 1 entities processed | 1,732,544 |
| Candidate pairs | 44,667,200 |
| Predicted matches | 8,552,099 |
| S2 predicted matches | 3,279,953 |
| S3 predicted matches | 5,272,146 |
| Mean candidates / S1 | 25.8 |
| Mean predictions / S1 | 4.94 |
| Runtime | 42,342 seconds (~11.76 hours) |

The production run completed successfully and generated the required matching and candidate-pair output files. The outputs were subsequently validated using the supplied submission validator.


## 5. Lessons Learned and Future Experiments

The experiments highlighted several important properties of large-scale entity resolution.

- Candidate generation has a major effect on both recall and computational cost.
- Combining multiple blocking rules is more robust to heterogeneous record noise than relying on a single blocking key.
- Candidate caps provide predictable computational bounds, but they introduce a trade-off between efficiency and coverage.
- The classifier threshold directly affects the precision-recall operating point.
- Full-scale inference is substantially more expensive than small validation experiments, making efficient indexing and bounded candidate generation essential.

Potential future experiments include adaptive candidate limits based on candidate density, stronger hard-negative sampling, probability calibration, alternative tree-based classifiers, and more detailed analysis of false-positive and false-negative cases.

