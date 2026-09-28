"""
Business Entity Resolution â€” Evaluation Module

Implements the official competition macro-averaged F_0.5 evaluation metric,
handling singletons, multiple matches, source-specific metrics, and prediction distributions.
"""

from typing import Dict, List, Set, Tuple, Any
import numpy as np

def compute_entity_f05(true_set: Set[str], pred_set: Set[str]) -> Tuple[float, int, int, int]:
    """
    Computes F_0.5 score for a single Source 1 entity.
    Returns: (f05_score, tp, fp, fn)
    """
    is_singleton = (len(true_set) == 0)
    
    if is_singleton:
        if len(pred_set) == 0:
            return 1.0, 0, 0, 0
        else:
            return 0.0, 0, len(pred_set), 0
    else:
        if len(pred_set) == 0:
            return 0.0, 0, 0, len(true_set)
        
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        prec = tp / len(pred_set) if len(pred_set) > 0 else 0.0
        rec = tp / len(true_set) if len(true_set) > 0 else 0.0
        
        denom = 0.25 * prec + rec
        if denom > 0:
            score = (1.25 * prec * rec) / denom
        else:
            score = 0.0
        return score, tp, fp, fn

def evaluate_predictions(
    ground_truth_map: Dict[str, Set[str]],
    predictions_map: Dict[str, Set[str]],
    s1_ids: List[str]
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of predictions against ground truth for all S1 entities.
    """
    entity_scores = []
    singleton_total = 0
    singleton_correct = 0
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    s2_tp = 0
    s2_fp = 0
    s2_fn = 0
    s3_tp = 0
    s3_fp = 0
    s3_fn = 0
    
    s1_with_missed_match = 0
    s1_zero_preds = 0
    s1_overpredicted = 0
    s1_underpredicted = 0
    
    true_counts = []
    pred_counts = []
    
    for s1_id in s1_ids:
        true_set = ground_truth_map.get(s1_id, set())
        pred_set = predictions_map.get(s1_id, set())
        
        true_counts.append(len(true_set))
        pred_counts.append(len(pred_set))
        
        if len(pred_set) == 0:
            s1_zero_preds += 1
            
        if len(pred_set) > len(true_set):
            s1_overpredicted += 1
        elif len(pred_set) < len(true_set):
            s1_underpredicted += 1
            
        score, tp, fp, fn = compute_entity_f05(true_set, pred_set)
        entity_scores.append(score)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        if len(true_set) == 0:
            singleton_total += 1
            if len(pred_set) == 0:
                singleton_correct += 1
        else:
            if fn > 0:
                s1_with_missed_match += 1
                
        # Source-specific accounting
        for m in (true_set & pred_set):
            if m.startswith("S2-"): s2_tp += 1
            else: s3_tp += 1
        for m in (pred_set - true_set):
            if m.startswith("S2-"): s2_fp += 1
            else: s3_fp += 1
        for m in (true_set - pred_set):
            if m.startswith("S2-"): s2_fn += 1
            else: s3_fn += 1

    macro_f05 = float(np.mean(entity_scores))
    singleton_acc = singleton_correct / singleton_total if singleton_total > 0 else 1.0
    
    p_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    p_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    p_f05 = (1.25 * p_prec * p_rec) / (0.25 * p_prec + p_rec) if (0.25 * p_prec + p_rec) > 0 else 0.0
    
    # S2 & S3 specific metrics
    s2_prec = s2_tp / (s2_tp + s2_fp) if (s2_tp + s2_fp) > 0 else 0.0
    s2_rec = s2_tp / (s2_tp + s2_fn) if (s2_tp + s2_fn) > 0 else 0.0
    s2_f05 = (1.25 * s2_prec * s2_rec) / (0.25 * s2_prec + s2_rec) if (0.25 * s2_prec + s2_rec) > 0 else 0.0
    
    s3_prec = s3_tp / (s3_tp + s3_fp) if (s3_tp + s3_fp) > 0 else 0.0
    s3_rec = s3_tp / (s3_tp + s3_fn) if (s3_tp + s3_fn) > 0 else 0.0
    s3_f05 = (1.25 * s3_prec * s3_rec) / (0.25 * s3_prec + s3_rec) if (0.25 * s3_prec + s3_rec) > 0 else 0.0

    return {
        "macro_f05": macro_f05,
        "singleton_acc": singleton_acc,
        "singleton_total": singleton_total,
        "singleton_correct": singleton_correct,
        "pairwise_prec": p_prec,
        "pairwise_rec": p_rec,
        "pairwise_f05": p_f05,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "s2_metrics": {"prec": s2_prec, "rec": s2_rec, "f05": s2_f05, "tp": s2_tp, "fp": s2_fp, "fn": s2_fn},
        "s3_metrics": {"prec": s3_prec, "rec": s3_rec, "f05": s3_f05, "tp": s3_tp, "fp": s3_fp, "fn": s3_fn},
        "s1_with_missed_match": s1_with_missed_match,
        "s1_zero_preds": s1_zero_preds,
        "s1_overpredicted": s1_overpredicted,
        "s1_underpredicted": s1_underpredicted,
        "true_counts": true_counts,
        "pred_counts": pred_counts
    }

