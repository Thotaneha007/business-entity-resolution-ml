"""
Business Entity Resolution â€” Production Blocking & Candidate Generation Module

Incorporates Step 4.5 Targeted Recovery Passes:
- Pass 1: Country + Exact Normalized Core Name (concatenated tokens, legal suffixes stripped)
- Pass 2: Country + Domain/Handle Cleansed Core Name (strips .com, .org, @, #, etc.)
- Pass 3: Country + First Two Significant Name Tokens
- Pass 4: Country + Primary Name Token + Address Numeric Component (plot/building/PIN)
- Pass 5: Country + 4-char Name Prefix + Address Number (recovers phonetic & suffix typos)
- Pass 6: Country + Address Number + Street Token (recovers multilingual scripts & severe name corruption)
- Pass 7: Country + Low-Frequency Name Token (fallback when address is missing)
- Safe Candidate Ranking & Budget Enforcement (Top-K per S1)
"""

import re
import unicodedata
from collections import defaultdict
from typing import List, Dict, Set
from src.normalization import LEGAL_SUFFIXES, COMMON_STOPWORDS, remove_accents

def clean_domain_core(name: str) -> str:
    """Strips web protocols, TLDs, handle symbols, and legal suffixes to expose core brand."""
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

class MultiPassBlockingIndex:
    """
    Enhanced Multi-Pass Inverted Index with dynamic country partitioning and targeted recovery.
    """
    def __init__(self, max_token_freq=60, max_street_freq=300):
        self.max_token_freq = max_token_freq
        self.max_street_freq = max_street_freq
        
        self.idx_core = defaultdict(list)
        self.idx_domain = defaultdict(list)
        self.idx_token2 = defaultdict(list)
        self.idx_name_num = defaultdict(list)
        self.idx_pref_num = defaultdict(list)
        self.idx_addr_spec = defaultdict(list)
        self.idx_name_rare = defaultdict(list)
        
        self.records = {}
        self.name_tok_freq = defaultdict(int)
        self.street_tok_freq = defaultdict(int)

    def index_target_records(self, records_iterable):
        temp_data = []
        for eid, name, addr, country in records_iterable:
            toks = extract_name_tokens(name)
            atoks = extract_address_tokens(addr)
            nums = extract_numbers(addr)
            
            for t in set(toks): self.name_tok_freq[(country, t)] += 1
            for a in set(atoks): self.street_tok_freq[(country, a)] += 1
            temp_data.append((eid, name, addr, country, toks, nums, atoks))

        for eid, name, addr, country, toks, nums, atoks in temp_data:
            cn = "".join(toks)
            self.records[eid] = (set(toks), set(nums), set(atoks), cn)
            
            # Pass 1: Exact core name
            if cn: self.idx_core[(country, cn)].append(eid)
            
            # Pass 2: Domain/handle core
            d_core = clean_domain_core(name)
            if len(d_core) >= 5 and d_core != cn:
                self.idx_domain[(country, d_core)].append(eid)
                
            # Pass 3: First 2 tokens
            if len(toks) >= 2:
                self.idx_token2[(country, toks[0], toks[1])].append(eid)
            elif len(toks) == 1:
                self.idx_token2[(country, toks[0], "")].append(eid)
                
            # Pass 4 & 5: First token & prefix + address number
            first_tok = toks[0] if toks else ""
            if len(first_tok) >= 3:
                for n in nums[:2]:
                    self.idx_name_num[(country, first_tok, n)].append(eid)
                    if len(first_tok) >= 4:
                        self.idx_pref_num[(country, first_tok[:4], n)].append(eid)
                        
            # Pass 6: Pure Address Specificity Key (Number + Street Token)
            if nums and atoks:
                for n in nums[:2]:
                    for st in atoks[:3]:
                        if len(st) >= 4 and self.street_tok_freq[(country, st)] <= self.max_street_freq:
                            self.idx_addr_spec[(country, n, st)].append(eid)
                            
            # Pass 7: Low-frequency name token fallback (for missing address)
            for t in toks:
                if len(t) >= 3 and self.name_tok_freq[(country, t)] <= self.max_token_freq:
                    self.idx_name_rare[(country, t)].append(eid)

    def query(self, s1_id: str, country: str, name: str, address: str, max_candidates=35) -> list:
        toks = extract_name_tokens(name)
        cn = "".join(toks)
        nums = extract_numbers(address)
        atoks = extract_address_tokens(address)
        
        cands = set()
        if cn: cands.update(self.idx_core.get((country, cn), []))
        
        d_core = clean_domain_core(name)
        if len(d_core) >= 5: cands.update(self.idx_domain.get((country, d_core), []))
        
        if len(toks) >= 2:
            cands.update(self.idx_token2.get((country, toks[0], toks[1]), []))
        elif len(toks) == 1:
            cands.update(self.idx_token2.get((country, toks[0], ""), []))
            
        first_tok = toks[0] if toks else ""
        if len(first_tok) >= 3:
            for n in nums[:2]:
                cands.update(self.idx_name_num.get((country, first_tok, n), []))
                if len(first_tok) >= 4:
                    cands.update(self.idx_pref_num.get((country, first_tok[:4], n), []))
                    
        if nums and atoks:
            for n in nums[:2]:
                for st in atoks[:3]:
                    if len(st) >= 4:
                        m = self.idx_addr_spec.get((country, n, st), [])
                        if len(m) <= 15:
                            cands.update(m)
                            
        for t in toks:
            if len(t) >= 3:
                m = self.idx_name_rare.get((country, t), [])
                if len(m) <= self.max_token_freq:
                    cands.update(m)
                    
        if len(cands) <= max_candidates:
            return sorted(cands)
            
        # Safe Pruning & Ranking
        s1_toks = set(toks)
        s1_nums = set(nums)
        s1_atoks = set(atoks)
        
        scored = []
        for cid in cands:
            c_toks, c_nums, c_atoks, c_cn = self.records[cid]
            name_sc = len(s1_toks & c_toks) * 3
            num_sc = len(s1_nums & c_nums) * 2 if (s1_nums and c_nums) else 0
            addr_sc = len(s1_atoks & c_atoks)
            if cn and c_cn and cn == c_cn:
                name_sc += 5
            total = name_sc + num_sc + addr_sc
            scored.append((total, cid))
            
        scored.sort(key=lambda x: x[0], reverse=True)
        return sorted([cid for _, cid in scored[:max_candidates]])

