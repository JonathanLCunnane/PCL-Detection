import numpy as np
from collections import Counter
from enum import Enum

class FeatureType(Enum):
    POS = "pos"
    NER = "ner"
    ZSCORE = "zscore"
    KEYWORD = "keyword"  # one-hot encoding of the article keyword metadata field

# POS/NER feature extraction

def extract_pos_features(doc, POS_TAGS) -> dict[int, dict[str, float]]:
    counts = Counter()
    total_tokens = len(doc)

    for token in doc:
        if not token.is_punct and not token.is_space:
            counts[token.pos_] += 1
    all_pos_density = sum(counts.values()) / total_tokens if total_tokens > 0 else 0
    pos_densities = {
        tag: counts.get(tag, 0) / total_tokens if total_tokens > 0
        else 0 for tag in POS_TAGS
    }
    return {**pos_densities, "all_pos_density": all_pos_density}


def extract_ner_features(doc, NER_TYPES) -> dict[int, dict[str, float]]:
    ent_counts = Counter()
    total_tokens = len(doc)
    for ent in doc.ents:
        ent_counts[ent.label_] += 1
    total_tokens = max(len(doc), 1)
    all_ner_density = len(doc.ents) / total_tokens if total_tokens > 0 else 0
    ner_densities = {
        ner_type: ent_counts.get(ner_type, 0) / total_tokens if total_tokens > 0 
        else 0 for ner_type in NER_TYPES
    }
    return {**ner_densities, "all_ner_density": all_ner_density}


def extract_pos_ner_features(doc, POS_TAGS, NER_TYPES) -> tuple[dict, dict]:
    return (extract_pos_features(doc, POS_TAGS), extract_ner_features(doc, NER_TYPES))


def extract_keyword_feature(keyword: str, keyword_to_idx: dict) -> np.ndarray:
    """
    One-hot encode a keyword value into a vector of length len(keyword_to_idx).
    Keywords not present in keyword_to_idx (e.g. from dev set) get a zero vector.
    """
    vec = np.zeros(len(keyword_to_idx), dtype=np.float32)
    idx = keyword_to_idx.get(keyword)
    if idx is not None:
        vec[idx] = 1.0
    return vec