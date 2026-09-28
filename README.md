# Business Entity Resolution using Machine Learning

A machine-learning-based system for identifying records that refer to the same real-world business entity across multiple noisy data sources.

## Overview

Business entity resolution determines whether records from different data sources refer to the same real-world entity.

The project handles:
- Business-name spelling variations and legal-suffix differences
- Multilingual text and transliteration
- Missing or incomplete addresses
- Address formatting differences
- Numeric and address-component variations
- Web-domain and handle artifacts

The system combines deterministic normalization, multi-pass blocking, candidate generation, pairwise similarity features, and machine-learning-based matching.

## Architecture

Raw Business Records
        |
        v
Text & Address Normalization
        |
        v
Multi-Pass Blocking
        |
        v
Candidate Generation
        |
        v
Pairwise Feature Engineering
        |
        v
HistGradientBoosting Matcher
        |
        v
Threshold-Based Matching
        |
        v
Matched Entity Pairs

## Machine Learning

The final matcher uses HistGradientBoostingClassifier from scikit-learn.

Production configuration:
- max_iter = 150
- learning_rate = 0.1
- max_depth = 6
- random_state = 42
- Matching threshold = 0.75

## Large-Scale Run

| Metric | Result |
|---|---:|
| Test S1 records | 1,732,544 |
| Candidate pairs | 44,667,200 |
| Mean candidates / S1 | 25.8 |
| Median candidates / S1 | 35 |
| P95 candidates / S1 | 35 |
| Maximum candidates / S1 | 35 |
| Predicted matches | 8,552,099 |
| Mean predictions / S1 | 4.94 |
| Runtime | ~11.76 hours |

Inference was performed country-by-country and in batches to control memory usage.

## Dataset

The challenge dataset is not included in this repository.

Obtain the dataset through the official challenge distribution and place it locally under the expected dataset/ directory.

Do not commit challenge data or generated submission files.

## Technologies

- Python
- Pandas
- NumPy
- scikit-learn
- RapidFuzz
- Machine Learning
- Information Retrieval
- Entity Resolution

## Repository Structure

business-entity-resolution-ml/
├── src/
│   ├── blocking.py
│   ├── evaluation.py
│   ├── features.py
│   ├── matcher.py
│   ├── normalization.py
│   └── pipeline.py
├── docs/
├── examples/
├── .gitignore
├── README.md
└── requirements.txt

## Competition Context

This project was developed in the context of the Amazon ML Challenge 2026.

The repository focuses on the underlying entity-resolution and machine-learning engineering rather than distributing the competition dataset.

## Official Challenge Resources

Challenge video:
https://d8it4huxumps7.cloudfront.net/files/6ab509c5b7036_ml_challenge_2026_video.mp4

Guidelines:
https://d8it4huxumps7.cloudfront.net/files/6ab56657b4f1a_guidelines_and_key_instructions_amazon_ml_challenge_2026.pdf

Challenge communications:
https://d8it4huxumps7.cloudfront.net/files/6ab674645103d_emails_comms_amazon_ml_challenge_2026.pdf

## Author

Thota Neha
