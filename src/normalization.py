"""
Business Entity Resolution â€” Normalization Module

Preserves multiple textual representations (original, unicode-normalized, case-normalized,
punctuation-cleaned, whitespace-collapsed, tokenized, and legal-suffix-stripped core).
Supports multilingual scripts (Indic, Latin, French diacritics) without information destruction.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Set

LEGAL_SUFFIXES = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'pvt', 'private', 'co', 'company', 'services', 'enterprises', 'solutions',
    'technologies', 'group', 'holdings', 'international', 'center', 'centre',
    'industries', 'associates', 'consultants', 'agency', 'partners', 'llp',
    'sarl', 'sa', 'sas', 'gmbh', 'bv', 'foundation', 'trust'
}

COMMON_STOPWORDS = {
    'the', 'and', 'of', 'for', 'in', 'on', 'at', 'to', 'a', 'an', '&', '-',
    'india', 'us', 'usa', 'france', 'north', 'south', 'east', 'west',
    'road', 'rd', 'street', 'st', 'lane', 'avenue', 'ave', 'floor', 'fl',
    'shop', 'plot', 'near', 'beside', 'opp', 'opposite', 'behind'
}

def remove_accents(text: str) -> str:
    """Normalize unicode and remove diacritics / accents using NFKD decomposition."""
    if not text:
        return ""
    nfkd = unicodedata.normalize('NFKD', str(text))
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def clean_punctuation_and_space(text: str) -> str:
    """Replaces non-alphanumeric characters with spaces and collapses whitespace."""
    if not text or str(text).lower() == "nan":
        return ""
    # Retain all alphanumeric (including non-Latin unicode characters)
    t = re.sub(r'[^\w\s]', ' ', str(text))
    return re.sub(r'\s+', ' ', t).strip()

@dataclass
class NormalizedEntity:
    entity_id: str
    country: str
    raw_name: str
    raw_addr: str
    
    # Name representations
    name_clean: str
    name_ascii: str
    name_core: str
    name_tokens: List[str]
    name_tokens_set: Set[str]
    name_char_3grams: Set[str]
    
    # Address representations
    addr_clean: str
    addr_ascii: str
    addr_tokens: List[str]
    addr_tokens_set: Set[str]
    addr_numbers: List[str]
    addr_numbers_set: Set[str]
    is_addr_missing: bool

def normalize_entity(entity_id: str, business_name: str, business_address: str, country: str) -> NormalizedEntity:
    """
    Constructs a multi-representation NormalizedEntity from raw tabular inputs.
    """
    raw_name = str(business_name) if business_name is not None else ""
    raw_addr = str(business_address) if business_address is not None else ""
    c_str = str(country).strip() if country is not None else ""

    # Name representations
    name_clean = clean_punctuation_and_space(raw_name).lower()
    name_ascii = remove_accents(name_clean)
    
    # Tokenization & core extraction
    words = name_ascii.split()
    name_toks = [w for w in words if len(w) >= 2 and w not in LEGAL_SUFFIXES and w not in COMMON_STOPWORDS]
    if not name_toks:
        # Fallback to all words if all were stopwords/suffixes
        name_toks = [w for w in words if len(w) >= 1]
    name_core = "".join(name_toks)
    
    # Character 3-grams
    if len(name_ascii) < 3:
        name_3g = {name_ascii} if name_ascii else set()
    else:
        name_3g = {name_ascii[i:i+3] for i in range(len(name_ascii)-2)}

    # Address representations
    is_missing = (not raw_addr or raw_addr.lower() == "nan" or raw_addr.strip() == "")
    if is_missing:
        addr_clean = ""
        addr_ascii = ""
        addr_toks = []
        nums = []
    else:
        addr_clean = clean_punctuation_and_space(raw_addr).lower()
        addr_ascii = remove_accents(addr_clean)
        addr_toks = [w for w in addr_ascii.split() if len(w) >= 3 and w not in COMMON_STOPWORDS and not w.isdigit()]
        nums = [num for num in re.findall(r'\b\d+\b', addr_ascii) if len(num) >= 1]

    return NormalizedEntity(
        entity_id=entity_id,
        country=c_str,
        raw_name=raw_name,
        raw_addr=raw_addr,
        name_clean=name_clean,
        name_ascii=name_ascii,
        name_core=name_core,
        name_tokens=name_toks,
        name_tokens_set=set(name_toks),
        name_char_3grams=name_3g,
        addr_clean=addr_clean,
        addr_ascii=addr_ascii,
        addr_tokens=addr_toks,
        addr_tokens_set=set(addr_toks),
        addr_numbers=nums,
        addr_numbers_set=set(nums),
        is_addr_missing=is_missing
    )

