"""
Business Entity Resolution â€” Pairwise Feature Engineering Module

Extracts comprehensive, domain-engineered pairwise similarity features across:
- Business Names (character edit, token sets, n-grams, legal suffix logic)
- Addresses (token overlap, numeric/postal overlap, missing-address indicators)
- Cross-Field Interactions (complementary evidence, multilingual & missing-address bridges)
- Source Indicators (S2 vs S3, country agreement)
"""

import math
from typing import List, Dict, Any
from rapidfuzz import fuzz, distance
from src.normalization import NormalizedEntity

FEATURE_NAMES = [
    # Business Name Features
    "name_exact_match",
    "name_core_match",
    "name_lev_sim",
    "name_fuzz_ratio",
    "name_partial_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_3gram_jaccard",
    "name_token_jaccard",
    "name_shared_tokens_count",
    "name_len_diff",
    "name_token_count_diff",
    "name_first_token_match",
    "name_last_token_match",
    
    # Address Features
    "addr_exact_match",
    "addr_lev_sim",
    "addr_fuzz_token_set",
    "addr_token_jaccard",
    "addr_shared_tokens_count",
    "addr_num_jaccard",
    "addr_num_overlap_count",
    "addr_has_shared_num",
    "addr_len_diff",
    "is_c_addr_missing",
    
    # Cross-Field Features
    "cross_name_and_addr_num",
    "cross_core_match_and_addr_tok",
    "cross_weak_name_strong_addr",
    "cross_strong_name_missing_addr",
    "cross_name_addr_geom_mean",
    
    # Source & Country Features
    "is_s2",
    "is_s3",
    "is_same_country"
]

def jaccard(set_a, set_b) -> float:
    if not set_a or not set_b:
        return 0.0
    union_len = len(set_a | set_b)
    return len(set_a & set_b) / union_len if union_len > 0 else 0.0

def compute_pairwise_features(s1: NormalizedEntity, c: NormalizedEntity) -> List[float]:
    """
    Computes a vector of numerical similarity features for an (S1, Candidate) pair.
    """
    # 1. Business Name Features
    name_exact = 1.0 if (s1.name_ascii and s1.name_ascii == c.name_ascii) else 0.0
    name_core = 1.0 if (s1.name_core and s1.name_core == c.name_core) else 0.0
    
    name_lev = float(distance.Levenshtein.normalized_similarity(s1.name_ascii, c.name_ascii))
    name_ratio = float(fuzz.ratio(s1.name_ascii, c.name_ascii)) / 100.0
    name_partial = float(fuzz.partial_ratio(s1.name_ascii, c.name_ascii)) / 100.0
    name_sort = float(fuzz.token_sort_ratio(s1.name_ascii, c.name_ascii)) / 100.0
    name_set = float(fuzz.token_set_ratio(s1.name_ascii, c.name_ascii)) / 100.0
    
    name_3g_jacc = jaccard(s1.name_char_3grams, c.name_char_3grams)
    name_tok_jacc = jaccard(s1.name_tokens_set, c.name_tokens_set)
    shared_tok_count = float(len(s1.name_tokens_set & c.name_tokens_set))
    
    len_diff = float(abs(len(s1.name_ascii) - len(c.name_ascii)))
    tok_count_diff = float(abs(len(s1.name_tokens) - len(c.name_tokens)))
    
    first_tok = 1.0 if (s1.name_tokens and c.name_tokens and s1.name_tokens[0] == c.name_tokens[0]) else 0.0
    last_tok = 1.0 if (s1.name_tokens and c.name_tokens and s1.name_tokens[-1] == c.name_tokens[-1]) else 0.0

    # 2. Address Features
    if c.is_addr_missing or s1.is_addr_missing:
        addr_exact = 0.0
        addr_lev = 0.0
        addr_fuzz_set = 0.0
        addr_tok_jacc = 0.0
        addr_shared_tok = 0.0
        addr_num_jacc = 0.0
        addr_num_shared = 0.0
        addr_has_num = 0.0
        addr_len_diff = float(len(s1.addr_ascii))
        is_missing = 1.0
    else:
        addr_exact = 1.0 if s1.addr_ascii == c.addr_ascii else 0.0
        addr_lev = float(distance.Levenshtein.normalized_similarity(s1.addr_ascii, c.addr_ascii))
        addr_fuzz_set = float(fuzz.token_set_ratio(s1.addr_ascii, c.addr_ascii)) / 100.0
        addr_tok_jacc = jaccard(s1.addr_tokens_set, c.addr_tokens_set)
        addr_shared_tok = float(len(s1.addr_tokens_set & c.addr_tokens_set))
        addr_num_jacc = jaccard(s1.addr_numbers_set, c.addr_numbers_set)
        addr_num_shared = float(len(s1.addr_numbers_set & c.addr_numbers_set))
        addr_has_num = 1.0 if addr_num_shared > 0 else 0.0
        addr_len_diff = float(abs(len(s1.addr_ascii) - len(c.addr_ascii)))
        is_missing = 0.0

    # 3. Cross-Field Interaction Features
    cross_name_addr_num = name_tok_jacc * addr_has_num
    cross_core_addr_tok = name_core * addr_tok_jacc
    
    # Weak name but very strong address match (e.g. Indic transliterated name with same address)
    is_weak_name_strong_addr = 1.0 if (name_tok_jacc < 0.4 and (addr_has_num == 1.0 or addr_tok_jacc >= 0.4)) else 0.0
    
    # Strong name with missing candidate address
    is_strong_name_missing_addr = 1.0 if (name_tok_jacc >= 0.75 and is_missing == 1.0) else 0.0
    
    # Geometric mean agreement
    cross_geom_mean = math.sqrt(name_tok_jacc * addr_tok_jacc)

    # 4. Source & Country Features
    is_s2 = 1.0 if c.entity_id.startswith("S2-") else 0.0
    is_s3 = 1.0 if c.entity_id.startswith("S3-") else 0.0
    same_country = 1.0 if (s1.country and c.country and s1.country == c.country) else 0.0

    return [
        name_exact,
        name_core,
        name_lev,
        name_ratio,
        name_partial,
        name_sort,
        name_set,
        name_3g_jacc,
        name_tok_jacc,
        shared_tok_count,
        len_diff,
        tok_count_diff,
        first_tok,
        last_tok,
        addr_exact,
        addr_lev,
        addr_fuzz_set,
        addr_tok_jacc,
        addr_shared_tok,
        addr_num_jacc,
        addr_num_shared,
        addr_has_num,
        addr_len_diff,
        is_missing,
        cross_name_addr_num,
        cross_core_addr_tok,
        is_weak_name_strong_addr,
        is_strong_name_missing_addr,
        cross_geom_mean,
        is_s2,
        is_s3,
        same_country
    ]

