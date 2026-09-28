"""
Business Entity Resolution â€” Full End-to-End Production Pipeline (Step 5)

Executes:
1. Final Matcher Training on comprehensive labeled training data using HistGradientBoostingClassifier.
2. Strategy D Multi-Pass Inverted Indexing with Dynamic Country Partitioning.
3. Country-by-country streaming inference over the 1,732,544 test entities.
4. Generates output/matching_results.tsv and output/candidate_pairs.tsv strictly complying with rules.
"""

import sys
import io
import os
import re
import time
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

from src.normalization import normalize_entity, NormalizedEntity, LEGAL_SUFFIXES, COMMON_STOPWORDS, remove_accents
from src.features import compute_pairwise_features, FEATURE_NAMES
from src.matcher import GradientBoostedMatcher

# ---------------------------------------------------------------------------
# STRATEGY D INVERTED INDEX IMPLEMENTATION
# ---------------------------------------------------------------------------

def clean_domain_core(name: str) -> str:
    n = re.sub(r'https?://|www\.', '', str(name).lower())
    n = re.sub(r'\.(com|org|net|in|co|us|gov|io|biz|info)\b', '', n)
    n = re.sub(r'[^a-z0-9\s]', ' ', n)
    words = [w for w in n.split() if w not in LEGAL_SUFFIXES and w not in COMMON_STOPWORDS]
    return "".join(words)

def clean_text(text: str) -> str:
    if not text or str(text).lower() == "nan": return ""
    text = remove_accents(str(text)).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def extract_name_tokens(name: str):
    cleaned = clean_text(name)
    return [t for t in cleaned.split() if len(t) >= 2 and t not in LEGAL_SUFFIXES and t not in COMMON_STOPWORDS]

def extract_numbers(address: str):
    if not address or str(address).lower() == "nan": return []
    return [num for num in re.findall(r'\b\d+\b', str(address)) if len(num) >= 1]

def extract_address_tokens(address: str):
    cleaned = clean_text(address)
    return [w for w in cleaned.split() if len(w) >= 3 and w not in COMMON_STOPWORDS and not w.isdigit()]

class StrategyDCountryIndex:
    """
    In-memory Inverted Index for a single country partition under Strategy D (Multi-Pass Union).
    """
    def __init__(self, country: str, max_token_freq=80, max_street_freq=350):
        self.country = country
        self.max_token_freq = max_token_freq
        self.max_street_freq = max_street_freq
        
        self.idx_core = defaultdict(list)
        self.idx_domain = defaultdict(list)
        self.idx_token2 = defaultdict(list)
        self.idx_name_num = defaultdict(list)
        self.idx_pref_num = defaultdict(list)
        self.idx_addr_spec = defaultdict(list)
        self.idx_name_rare = defaultdict(list)
        
        self.records = {} # eid -> NormalizedEntity
        self.block_records = {}
        self.name_tok_freq = defaultdict(int)
        self.street_tok_freq = defaultdict(int)

    def build_from_stream(self, s2_path: str, s3_path: str):
        t0 = time.time()
        print(
            f"[{self.country}] Reading S2 & S3 for country '{self.country}' "
            f"(lightweight blocking pass)...",
            flush=True
        )

        # Full NormalizedEntity objects are NOT created here.
        # They are created lazily only for candidates that reach the matcher.
        self.records = {}          # eid -> NormalizedEntity cache
        self.raw_records = {}      # eid -> (name, address, country)
        self.block_records = {}    # eid -> (name_tokens_set, nums_set, addr_tokens_set, core)

        # ------------------------------------------------------------
        # PASS 1: collect lightweight blocking data + frequencies
        # ------------------------------------------------------------
        total = 0

        for src_path in (s2_path, s3_path):
            with open(src_path, "r", encoding="utf-8") as f:
                f.readline()

                for line in f:
                    parts = line.rstrip("\n").split("\t")

                    if len(parts) < 4 or parts[3].strip() != self.country:
                        continue

                    eid, name, addr = parts[0], parts[1], parts[2]

                    toks = extract_name_tokens(name)
                    nums = extract_numbers(addr)
                    atoks = extract_address_tokens(addr)
                    cn = "".join(toks)

                    tok_set = set(toks)
                    num_set = set(nums)
                    atok_set = set(atoks)

                    self.raw_records[eid] = (name, addr, self.country)
                    self.block_records[eid] = (
                        tok_set,
                        num_set,
                        atok_set,
                        cn
                    )

                    for t in tok_set:
                        self.name_tok_freq[t] += 1

                    for a in atok_set:
                        self.street_tok_freq[a] += 1

                    total += 1

        print(
            f"[{self.country}] Loaded {total:,} records in "
            f"{time.time()-t0:.1f}s. Building posting lists...",
            flush=True
        )

        # ------------------------------------------------------------
        # PASS 2: build the seven blocking indexes
        # ------------------------------------------------------------
        for eid, (tok_set, num_set, atok_set, cn) in self.block_records.items():

            toks = tuple(tok_set)
            nums = tuple(num_set)
            atoks = tuple(atok_set)

            # Pass 1: Core name
            if cn:
                self.idx_core[cn].append(eid)

            # Pass 2: Domain core
            name = self.raw_records[eid][0]
            d_core = clean_domain_core(name)

            if len(d_core) >= 5 and d_core != cn:
                self.idx_domain[d_core].append(eid)

            # Pass 3: First two tokens
            # Preserve deterministic behavior.
            toks_ordered = extract_name_tokens(name)

            if len(toks_ordered) >= 2:
                self.idx_token2[(toks_ordered[0], toks_ordered[1])].append(eid)
            elif len(toks_ordered) == 1:
                self.idx_token2[(toks_ordered[0], "")].append(eid)

            # Pass 4 & 5: First token + number / prefix + number
            first_tok = toks_ordered[0] if toks_ordered else ""

            if len(first_tok) >= 3:
                for n in nums[:2]:
                    self.idx_name_num[(first_tok, n)].append(eid)

                    if len(first_tok) >= 4:
                        self.idx_pref_num[(first_tok[:4], n)].append(eid)

            # Pass 6: Number + street token
            if nums and atoks:
                for n in nums[:2]:
                    for st in atoks[:3]:
                        if (
                            len(st) >= 4
                            and self.street_tok_freq[st] <= self.max_street_freq
                        ):
                            self.idx_addr_spec[(n, st)].append(eid)

            # Pass 7: rare name token
            for t in tok_set:
                if (
                    len(t) >= 3
                    and self.name_tok_freq[t] <= self.max_token_freq
                ):
                    self.idx_name_rare[t].append(eid)

        print(
            f"[{self.country}] Inverted index ready in "
            f"{time.time()-t0:.1f}s.",
            flush=True
        )

    def get_normalized(self, eid: str) -> NormalizedEntity:
        """Lazily normalize only a target record that became a candidate."""
        cached = self.records.get(eid)

        if cached is not None:
            return cached

        name, addr, country = self.raw_records[eid]

        norm = normalize_entity(
            eid,
            name,
            addr,
            country
        )

        self.records[eid] = norm
        return norm

    def query(self, s1: NormalizedEntity, max_cands=35) -> list:
        cands = set()
        
        # Pass 1
        if s1.name_core:
            cands.update(self.idx_core.get(s1.name_core, []))
            
        # Pass 2
        d_core = clean_domain_core(s1.raw_name)
        if len(d_core) >= 5:
            cands.update(self.idx_domain.get(d_core, []))
            
        # Pass 3
        toks = s1.name_tokens
        if len(toks) >= 2:
            cands.update(self.idx_token2.get((toks[0], toks[1]), []))
        elif len(toks) == 1:
            cands.update(self.idx_token2.get((toks[0], ""), []))
            
        # Pass 4 & 5
        nums = s1.addr_numbers
        first_tok = toks[0] if toks else ""
        if len(first_tok) >= 3:
            for n in nums[:2]:
                cands.update(self.idx_name_num.get((first_tok, n), []))
                if len(first_tok) >= 4:
                    cands.update(self.idx_pref_num.get((first_tok[:4], n), []))
                    
        # Pass 6
        atoks = s1.addr_tokens
        if nums and atoks:
            for n in nums[:2]:
                for st in atoks[:3]:
                    if len(st) >= 4:
                        m = self.idx_addr_spec.get((n, st), [])
                        if len(m) <= 15:
                            cands.update(m)
                            
        # Pass 7
        for t in toks:
            if len(t) >= 3:
                m = self.idx_name_rare.get(t, [])
                if len(m) <= self.max_token_freq:
                    cands.update(m)
                    
        if len(cands) <= max_cands:
            return sorted(cands)
            
        # Safe rank if exceeds max_cands
        s1_toks = s1.name_tokens_set
        s1_nums = s1.addr_numbers_set
        s1_atoks = s1.addr_tokens_set
        
        scored = []
        for cid in cands:
            # Production index: compact blocking representation.
            if cid in self.block_records:
                c_toks, c_nums, c_atoks, c_cn = self.block_records[cid]
                name_sc = len(s1_toks & c_toks) * 3
                num_sc = len(s1_nums & c_nums) * 2 if (s1_nums and c_nums) else 0
                addr_sc = len(s1_atoks & c_atoks)
                if s1.name_core and c_cn and s1.name_core == c_cn:
                    name_sc += 5

            # Training index: full normalized records.
            else:
                c = self.records[cid]
                name_sc = len(s1_toks & c.name_tokens_set) * 3
                num_sc = len(s1_nums & c.addr_numbers_set) * 2 if (s1_nums and c.addr_numbers_set) else 0
                addr_sc = len(s1_atoks & c.addr_tokens_set)
                if s1.name_core and c.name_core and s1.name_core == c.name_core:
                    name_sc += 5

            scored.append((name_sc + num_sc + addr_sc, cid))
        scored.sort(key=lambda x: x[0], reverse=True)
        return sorted([cid for _, cid in scored[:max_cands]])

# ---------------------------------------------------------------------------
# MODEL TRAINER
# ---------------------------------------------------------------------------

def train_production_matcher(n_train_s1=12000):
    t0 = time.time()
    print("=" * 80)
    print(f"TRAINING PRODUCTION MATCHING MODEL (Sample: {n_train_s1:,} S1 entities)")
    print("=" * 80)
    
    # 1. Load S1 training records
    s1_df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", nrows=n_train_s1)
    s1_dict = {r["entity_id"]: normalize_entity(r["entity_id"], r["business_name"], r["business_address"], r["country"]) for _, r in s1_df.iterrows()}
    all_s1_ids = list(s1_df["entity_id"])
    del s1_df
    
    # 2. Load ground truth
    gt_map = {}
    target_ids = set()
    with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if parts[0] in s1_dict:
                mstr = parts[1].strip() if len(parts) > 1 else ""
                if mstr and mstr != "nan":
                    mids = set(m.strip() for m in mstr.split(",") if m.strip())
                    gt_map[parts[0]] = mids
                    target_ids.update(mids)
                else:
                    gt_map[parts[0]] = set()
                if len(gt_map) == n_train_s1: break
                
    print(f"Ground truth loaded: {sum(len(v) for v in gt_map.values()):,} true matches across {len(gt_map):,} entities.")
    
    # 3. Stream target pool for training
    target_pool = {}
    unfound = set(target_ids)
    for src in ("train_source2.tsv", "train_source3.tsv"):
        with open(os.path.join("dataset", "train", src), "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                eid = p[0]
                if eid in unfound:
                    target_pool[eid] = normalize_entity(eid, p[1] if len(p)>1 else "", p[2] if len(p)>2 else "", p[3] if len(p)>3 else "")
                    unfound.remove(eid)
                elif len(target_pool) < 250000:
                    target_pool[eid] = normalize_entity(eid, p[1] if len(p)>1 else "", p[2] if len(p)>2 else "", p[3] if len(p)>3 else "")
                if len(unfound) == 0 and len(target_pool) >= 250000: break

    print(f"Loaded {len(target_pool):,} target pool records for training.")
    
    # 4. Index targets by country
    country_indexes = {}
    for c_label in ("US", "India"):
        c_idx = StrategyDCountryIndex(country=c_label)
        c_list = [norm for norm in target_pool.values() if norm.country == c_label]
        for norm in c_list:
            c_idx.records[norm.entity_id] = norm
            for t in norm.name_tokens_set: c_idx.name_tok_freq[t] += 1
            for a in norm.addr_tokens_set: c_idx.street_tok_freq[a] += 1
            
            # populating passes
            if norm.name_core: c_idx.idx_core[norm.name_core].append(norm.entity_id)
            d_core = clean_domain_core(norm.raw_name)
            if len(d_core) >= 5 and d_core != norm.name_core: c_idx.idx_domain[d_core].append(norm.entity_id)
            toks = norm.name_tokens
            if len(toks) >= 2: c_idx.idx_token2[(toks[0], toks[1])].append(norm.entity_id)
            elif len(toks) == 1: c_idx.idx_token2[(toks[0], "")].append(norm.entity_id)
            first_tok = toks[0] if toks else ""
            if len(first_tok) >= 3:
                for n in norm.addr_numbers[:2]:
                    c_idx.idx_name_num[(first_tok, n)].append(norm.entity_id)
                    if len(first_tok) >= 4: c_idx.idx_pref_num[(first_tok[:4], n)].append(norm.entity_id)
            for t in toks:
                if len(t) >= 3 and c_idx.name_tok_freq[t] <= 80: c_idx.idx_name_rare[t].append(norm.entity_id)
        country_indexes[c_label] = c_idx

    # 5. Extract training features
    print("Extracting pairwise training features...", flush=True)
    X_train, y_train = [], []
    for s1_id in all_s1_ids:
        s1 = s1_dict[s1_id]
        true_mids = gt_map[s1_id]
        c_idx = country_indexes.get(s1.country)
        if not c_idx: continue
        cands = c_idx.query(s1, max_cands=35)
        for cid in cands:
            c = target_pool[cid]
            X_train.append(compute_pairwise_features(s1, c))
            y_train.append(1 if cid in true_mids else 0)

    print(f"Training dataset: {len(X_train):,} pairs (Positives: {sum(y_train):,}, Negatives: {len(y_train)-sum(y_train):,})", flush=True)
    clf = GradientBoostedMatcher(max_iter=150, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)
    print(f"Model trained in {time.time()-t0:.1f}s.")
    return clf

# ---------------------------------------------------------------------------
# FULL TEST INFERENCE & STREAMING RUNNER
# ---------------------------------------------------------------------------

def run_test_inference(clf, is_dry_run=False, dry_run_limit=3000):
    t_global = time.time()
    print("=" * 80)
    print(f"RUNNING TEST INFERENCE (Mode: {'DRY RUN' if is_dry_run else 'FULL PRODUCTION'})")
    print("=" * 80)
    
    test_s1_path = os.path.join("dataset", "test", "test_source1.tsv")
    test_s2_path = os.path.join("dataset", "test", "test_source2.tsv")
    test_s3_path = os.path.join("dataset", "test", "test_source3.tsv")
    
    matching_out = os.path.join("output", "matching_results.tsv")
    candidate_out = os.path.join("output", "candidate_pairs.tsv")
    
    os.makedirs("output", exist_ok=True)
    
    # 1. Discover countries in test_source1
    print("Discovering country partitions in test_source1.tsv...", flush=True)
    country_counts = defaultdict(int)
    test_s1_by_country = defaultdict(list)
    
    with open(test_s1_path, "r", encoding="utf-8") as f:
        f.readline()
        for idx, line in enumerate(f):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                country = parts[3].strip()
                country_counts[country] += 1
                if is_dry_run:
                    if len(test_s1_by_country[country]) < (dry_run_limit // 3):
                        test_s1_by_country[country].append(parts)
                else:
                    test_s1_by_country[country].append(parts)
                    
    total_test_s1 = sum(len(v) for v in test_s1_by_country.values())
    print(f"Total test S1 entities to process: {total_test_s1:,}")
    print(f"Country breakdown: {dict(country_counts)}")

    # Open output files with exact headers
    with open(matching_out, "w", encoding="utf-8") as f_match, \
         open(candidate_out, "w", encoding="utf-8") as f_cand:
        
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        
        # Diagnostics
        total_cand_pairs = 0
        total_pred_pairs = 0
        pred_counts_hist = defaultdict(int)
        s2_pred_count = 0
        s3_pred_count = 0
        zero_pred_count = 0
        cand_counts_all = []
        
        # 2. Process Country Partition by Country Partition
        # Process order: France -> US -> India
        countries_to_process = sorted(test_s1_by_country.keys())
        
        for c_idx, country in enumerate(countries_to_process):
            t_country = time.time()
            s1_records = test_s1_by_country[country]
            print(f"\n[{c_idx+1}/{len(countries_to_process)}] Processing Country Partition: '{country}' ({len(s1_records):,} S1 entities)...", flush=True)
            
            # Build country index
            idx_engine = StrategyDCountryIndex(country=country)
            idx_engine.build_from_stream(test_s2_path, test_s3_path)
            
            # Batch inference in chunks of 10,000 S1 entities
            chunk_size = 10000
            for start_i in range(0, len(s1_records), chunk_size):
                chunk = s1_records[start_i:start_i+chunk_size]
                chunk_cands = []
                chunk_s1_norm = []
                
                # Query candidates
                for p in chunk:
                    s1_norm = normalize_entity(p[0], p[1], p[2], p[3])
                    cands = idx_engine.query(s1_norm, max_cands=35)
                    chunk_cands.append(cands)
                    chunk_s1_norm.append(s1_norm)
                    cand_counts_all.append(len(cands))
                    total_cand_pairs += len(cands)
                    
                # Feature extraction and prediction in batches
                pair_keys = []
                pair_feats = []
                for s1_norm, cands in zip(chunk_s1_norm, chunk_cands):
                    for cid in cands:
                        c_norm = idx_engine.get_normalized(cid)
                        pair_keys.append((s1_norm.entity_id, cid))
                        pair_feats.append(compute_pairwise_features(s1_norm, c_norm))
                        
                # Predict
                if pair_feats:
                    probs = clf.predict_proba(pair_feats)
                else:
                    probs = []
                    
                s1_matches = defaultdict(list)
                for (s1_id, cid), p in zip(pair_keys, probs):
                    if p >= 0.75: # Validated threshold
                        s1_matches[s1_id].append(cid)
                        if cid.startswith("S2-"): s2_pred_count += 1
                        elif cid.startswith("S3-"): s3_pred_count += 1
                        
                # Write results
                for s1_norm, cands in zip(chunk_s1_norm, chunk_cands):
                    s1_id = s1_norm.entity_id
                    # 1. candidate_pairs.tsv: EXACT candidates scored by model
                    f_cand.write(f"{s1_id}\t{','.join(cands)}\n")
                    
                    # 2. matching_results.tsv: matched IDs
                    m_list = sorted(s1_matches[s1_id])
                    f_match.write(f"{s1_id}\t{','.join(m_list)}\n")
                    
                    m_count = len(m_list)
                    pred_counts_hist[m_count] += 1
                    total_pred_pairs += m_count
                    if m_count == 0: zero_pred_count += 1

                print(f"  Processed {min(start_i+chunk_size, len(s1_records)):,} / {len(s1_records):,} [{country}] entities...", flush=True)

            print(f"Partition '{country}' completed in {time.time()-t_country:.1f}s.")
            # Clear index from memory
            del idx_engine

    elapsed = time.time() - t_global
    print("\n" + "=" * 80)
    print("INFERENCE SUMMARY & PREDICTION SANITY CHECK")
    print("=" * 80)
    print(f"Total Test S1 Entities Processed: {total_test_s1:,}")
    print(f"Total Candidate Pairs Generated : {total_cand_pairs:,}")
    print(f"Candidates per S1               : Mean={np.mean(cand_counts_all):.1f}, Med={np.median(cand_counts_all):.0f}, P95={np.percentile(cand_counts_all, 95):.0f}, Max={max(cand_counts_all)}")
    print(f"Total Predicted Matches         : {total_pred_pairs:,}")
    print(f"  - S2 Predicted Matches        : {s2_pred_count:,} ({s2_pred_count/total_pred_pairs*100 if total_pred_pairs else 0:.1f}%)")
    print(f"  - S3 Predicted Matches        : {s3_pred_count:,} ({s3_pred_count/total_pred_pairs*100 if total_pred_pairs else 0:.1f}%)")
    print(f"Singletons (0 matches predicted): {zero_pred_count:,} ({zero_pred_count/total_test_s1*100:.2f}%)")
    print(f"Mean Predictions per S1         : {total_pred_pairs/total_test_s1:.2f}")
    print(f"Total Inference Runtime         : {elapsed:.2f}s ({elapsed/60:.2f} min)")
    
    print("\nPrediction Count Distribution:")
    for k in sorted(pred_counts_hist.keys()):
        cnt = pred_counts_hist[k]
        print(f"  Matches = {k:2d}: {cnt:7,} ({cnt/total_test_s1*100:5.2f}%)")
        if k >= 10: break
    gt10 = sum(pred_counts_hist[k] for k in pred_counts_hist if k > 10)
    if gt10 > 0:
        print(f"  Matches > 10: {gt10:7,} ({gt10/total_test_s1*100:5.2f}%)")
        
    return {
        "total_test_s1": total_test_s1,
        "total_cand_pairs": total_cand_pairs,
        "mean_cands": float(np.mean(cand_counts_all)),
        "med_cands": float(np.median(cand_counts_all)),
        "p95_cands": float(np.percentile(cand_counts_all, 95)),
        "max_cands": int(max(cand_counts_all)),
        "total_preds": total_pred_pairs,
        "s2_preds": s2_pred_count,
        "s3_preds": s3_pred_count,
        "zero_preds": zero_pred_count,
        "elapsed_sec": elapsed
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run on small deterministic test subset")
    parser.add_argument("--full", action="store_true", help="Run full test inference")
    args = parser.parse_args()
    
    clf = train_production_matcher()
    if args.dry_run:
        run_test_inference(clf, is_dry_run=True, dry_run_limit=3000)
    elif args.full:
        run_test_inference(clf, is_dry_run=False)
    else:
        print("Please specify --dry-run or --full")

