"""
Business Entity Resolution â€” Matcher Models Module

Implements progressively stronger baseline matching models:
1. RuleBasedMatcher (Weighted heuristic baseline)
2. LogisticRegressionMatcher (L2-regularized linear model)
3. GradientBoostedMatcher (HistGradientBoosting non-linear ensemble)
Supports global and source-specific (S2 vs S3) threshold calibration.
"""

from typing import List, Dict, Tuple, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

class RuleBasedMatcher:
    """
    Weighted similarity heuristic baseline.
    """
    def __init__(self, name_weight=0.65, addr_weight=0.35):
        self.name_weight = name_weight
        self.addr_weight = addr_weight

    def fit(self, X: List[List[float]], y: List[int]):
        pass # No training required for rule-based

    def predict_proba(self, X: List[List[float]]) -> np.ndarray:
        probs = []
        for feat in X:
            # feat indices from FEATURE_NAMES:
            # 2: name_lev_sim, 8: name_tok_jacc, 15: addr_lev_sim, 17: addr_tok_jacc, 21: addr_has_shared_num, 23: is_missing
            name_score = 0.5 * feat[2] + 0.5 * feat[8]
            is_missing = feat[23]
            
            if is_missing > 0.5:
                # If address missing, rely entirely on name with penalty
                score = name_score * 0.95
            else:
                addr_score = 0.4 * feat[15] + 0.3 * feat[17] + 0.3 * feat[21]
                score = self.name_weight * name_score + self.addr_weight * addr_score
                
            probs.append(min(max(score, 0.0), 1.0))
        return np.array(probs)

class LogisticRegressionMatcher:
    """
    L2-regularized Logistic Regression model.
    """
    def __init__(self, C=1.0, max_iter=200):
        self.clf = LogisticRegression(C=C, max_iter=max_iter, random_state=42)

    def fit(self, X: List[List[float]], y: List[int]):
        self.clf.fit(X, y)

    def predict_proba(self, X: List[List[float]]) -> np.ndarray:
        if not X: return np.array([])
        return self.clf.predict_proba(X)[:, 1]

class GradientBoostedMatcher:
    """
    Lightweight, fast HistGradientBoosting classifier.
    Handles non-linear cross-field interactions naturally.
    """
    def __init__(self, max_iter=100, learning_rate=0.1, max_depth=6, random_state=42):
        self.clf = HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )

    def fit(self, X: List[List[float]], y: List[int]):
        self.clf.fit(X, y)

    def predict_proba(self, X: List[List[float]]) -> np.ndarray:
        if not X: return np.array([])
        return self.clf.predict_proba(X)[:, 1]
    
    @property
    def feature_importances_(self):
        # Permutation or tree-based approximation
        return None

