# Methodology

## 1. Problem Formulation

The task is to identify records in Source 2 and Source 3 that refer to the same real-world business as each Source 1 entity.

Each entity contains three primary attributes:

- Business name
- Business address
- Country

The core challenge is that equivalent businesses may appear with different spellings, formatting, transliterations, missing address components, URL artifacts, or other textual variations.

The system therefore treats entity resolution as a two-stage retrieval and classification problem:

1. Generate a high-recall candidate set for each Source 1 entity.
2. Classify candidate pairs using learned similarity features.

The final objective is to identify true entity matches while controlling false positive predictions.

## 2. Data Preparation and Normalization

The input data is processed separately by source and country so that candidate generation operates within the appropriate country partition.

The normalization stage converts raw text into consistent representations used by both blocking and pairwise feature computation. The process addresses common variations in business records, including:

- Case and whitespace differences.
- Punctuation and formatting variation.
- Common legal-entity suffixes.
- URL and domain artifacts embedded in business names.
- Token-level variation in business names and addresses.
- Missing or incomplete address components.

Multiple normalized representations are retained because different matching situations require different forms of evidence. Token-based representations are useful for blocking, while normalized strings and structural information are used during pairwise comparison.

Country is retained as a matching constraint. The training analysis showed that known matched pairs belong to the same country, so the production pipeline performs country-level partitioning before candidate generation.


## 3. Candidate Generation and Blocking

The full search space is too large for exhaustive pairwise comparison. Candidate generation therefore reduces the number of pairs that reach the machine-learning stage.

The blocking implementation uses seven complementary passes. Each pass retrieves candidates using a different combination of normalized business-name and address information. The resulting candidate sets are merged so that different types of record variation can be handled by different blocking rules.

The blocking strategy was evaluated during development using candidate recall and candidate volume. The selected strategy prioritized high recall while keeping the downstream matching workload bounded.

After candidates are generated, they are ranked and limited to a maximum of 35 candidates for each Source 1 entity in the production pipeline. This provides a predictable upper bound on the number of pairs that require feature computation and classification.

Blocking is intentionally treated as a retrieval stage rather than a matching decision. A candidate being retrieved does not mean that it is considered a true match; it only means that the pair is sufficiently plausible to be evaluated by the classifier.


## 4. Feature Engineering

For every blocked candidate pair, the system computes a numerical feature vector describing how similar the two entity records are.

The features combine evidence from multiple fields rather than relying on a single string comparison. The feature set includes name-based similarity, address-based similarity, token overlap, character-level similarity, and structural properties of the compared records.

Important feature categories include:

- Normalized business-name similarity.
- Address similarity.
- Token overlap between corresponding fields.
- Character-level similarity for handling spelling variation and typos.
- Relative and absolute text-length differences.
- Structural signals derived from the normalized representations.
- Country consistency.

This feature design allows the model to handle cases where one field is strong while another is noisy or incomplete. For example, a business name may differ because of transliteration while the address provides stronger evidence, or the name may be highly similar while the address contains missing components.

The feature-generation stage produces the same fixed feature representation for both training pairs and inference candidates, ensuring that the classifier receives consistent inputs.


## 5. Model Training and Classification

The matching model is trained as a binary classifier over labeled entity pairs. Positive pairs represent known matches from the training ground truth, while negative examples represent candidate pairs that are not known to match.

The production implementation uses scikit-learn HistGradientBoostingClassifier. Gradient boosting is suitable for this task because the decision depends on combinations of heterogeneous similarity features rather than on a single linear signal.

The production model configuration is:

| Parameter | Value |
|---|---:|
| Model | HistGradientBoostingClassifier |
| Maximum iterations | 150 |
| Learning rate | 0.1 |
| Maximum depth | 6 |
| Random state | 42 |

For each candidate pair, the trained classifier outputs a probability for the positive class. The probability is subsequently compared with the production decision threshold to determine whether the pair is emitted as a predicted match.


## 6. Threshold Selection and Evaluation

The classifier produces a probability for each candidate pair, so a decision threshold is required to convert probabilities into predicted matches.

The challenge evaluation metric is macro F0.5, which places greater emphasis on precision than recall. This makes false positive control particularly important: predicting an incorrect entity match can be more costly than leaving a possible match unresolved.

During development, multiple threshold values were evaluated on a validation setup. The production inference run used a threshold of **0.75**. This threshold was kept fixed for the complete production run rather than changing it during inference.

The evaluation process considers both candidate-generation quality and final matching quality. Candidate recall measures whether known matches enter the candidate set, while precision and recall of the final predictions measure the quality of the classifier decisions.

The separation between these measurements is important. A true match that is never generated by blocking cannot be recovered by the classifier, while an incorrect candidate that reaches the classifier can still be rejected by the model.


## 7. Full-Scale Inference Results

The final pipeline was executed on the complete test population of **1,732,544 Source 1 entities** across the available country partitions.

The production run generated **44,667,200 candidate pairs**, corresponding to an average of 25.8 candidates per Source 1 entity. The median and 95th-percentile candidate counts were both 35 because of the production candidate cap.

The complete run produced **8,552,099 predicted matches**, consisting of **3,279,953 Source 2 records** and **5,272,146 Source 3 records**. The average number of predicted matches per Source 1 entity was 4.94.

The complete inference run took **42,342 seconds, approximately 11.76 hours**.

Of the 1,732,544 Source 1 entities, 55,332 received no predicted matches, while 1,677,212 received at least one predicted match.

These results demonstrate the computational behavior of the complete pipeline and provide a reproducible reference point for future optimization experiments.


## 8. Error Analysis and Limitations

The main sources of difficulty in this task are noisy business names, transliteration, incomplete addresses, formatting differences, URL artifacts, and records that share similar business names.

The two-stage design creates two distinct classes of failure:

- **Blocking errors:** a true match is not included in the candidate set and therefore cannot reach the classifier.
- **Classification errors:** a candidate reaches the matcher but is incorrectly accepted or rejected.

The fixed candidate cap also introduces a trade-off between computational cost and candidate coverage. In dense candidate groups, lower-ranked candidates may be excluded before classification.

The classifier is also dependent on the training examples used to learn the relationship between pairwise features and match labels. Differences between training and test distributions can therefore affect performance.

Future improvements could include adaptive candidate budgets, stronger hard-negative generation, probability calibration, more systematic error analysis, and more targeted blocking rules for difficult record types.

