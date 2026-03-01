import numpy as np
from collections import Counter
from logging import getLogger

LOG = getLogger(__name__)

def compute_fightin_words_zscores(paras, labels, ns=(1,2,3), min_count=10):
    """
    Compute Fightin' Words z-scores from a list of spaCy docs and a list/array of labels.
    Returns z_scores, log_odds, pos_counts, neg_counts for EDA printout.
    """
    if set(labels) == {0, 1}:
        pos_cond = lambda lbl: lbl == 1
        LOG.info(f"Detected binary label.")
    else:
        pos_cond = lambda lbl: lbl >= 2
    pos_counts = Counter()
    neg_counts = Counter()

    for para, label in zip(paras, labels):
        tokens = [t.lemma_.lower() if t.is_alpha else "###" for t in para]
        ngrams = []
        for n in ns:
            for i in range(len(tokens) - n + 1):
                chunk = tokens[i:i + n]
                if "###" in chunk:
                    continue
                ngrams.append(" ".join(chunk))
        if pos_cond(label):
            pos_counts.update(ngrams)
        else:
            neg_counts.update(ngrams)
    bg_counts = pos_counts + neg_counts
    scored = {w for w, c in bg_counts.items() if c >= min_count}
    n_pos = sum(pos_counts.values())
    n_neg = sum(neg_counts.values())
    alpha_0 = sum(bg_counts.values())
    z_scores = {}
    log_odds = {}
    for w in scored:
        y_pos = pos_counts[w]
        y_neg = neg_counts[w]
        alpha_w = bg_counts[w]
        omega_pos = np.log((y_pos + alpha_w) / (n_pos + alpha_0 - y_pos - alpha_w))
        omega_neg = np.log((y_neg + alpha_w) / (n_neg + alpha_0 - y_neg - alpha_w))
        delta = omega_pos - omega_neg
        var = (1.0 / (y_pos + alpha_w)) + (1.0 / (y_neg + alpha_w))
        z_scores[w] = delta / np.sqrt(var)
        log_odds[w] = delta
    return z_scores, log_odds, pos_counts, neg_counts


def build_topk_ngrams(z_dict: dict, k: int = 50) -> list[str]:
    """
    Return the top-k n-grams sorted by |z-score| (most discriminative first).
    These are the n-grams most strongly associated with either PCL or non-PCL class.
    """
    return sorted(z_dict, key=lambda w: abs(z_dict[w]), reverse=True)[:k]


def extract_topk_zscore_features(doc, topk_ngrams: list[str], ns: tuple = (1, 2, 3)) -> np.ndarray:
    """
    Binary indicator vector of shape (len(topk_ngrams),):
    1.0 if the n-gram appears in the document, 0.0 otherwise.

    Preserves the identity of which specific PCL-leaning n-grams appear,
    rather than aggregate statistics (cf. extract_zscore_features).
    """
    tokens = [t.lemma_.lower() if t.is_alpha else "###" for t in doc]
    present = set()
    for n in ns:
        for i in range(len(tokens) - n + 1):
            chunk = tokens[i:i + n]
            if "###" in chunk:
                continue
            present.add(" ".join(chunk))

    return np.array(
        [1.0 if ng in present else 0.0 for ng in topk_ngrams],
        dtype=np.float32,
    )


def extract_zscore_features(doc, z_dict, ns=(1,2,3), zscore_threshold=1.96):
    tokens = [t.lemma_.lower() if t.is_alpha else "###" for t in doc]
    ngram_zscores = []
    for n in ns:
        for i in range(len(tokens) - n + 1):
            chunk = tokens[i:i + n]
            if "###" in chunk:
                continue
            ngram = " ".join(chunk)
            if ngram in z_dict:
                ngram_zscores.append(z_dict[ngram])
    if len(ngram_zscores) == 0:
        return np.zeros(6, dtype=np.float32)
    arr = np.array(ngram_zscores)
    return np.array([
        arr.mean(),
        arr.max(),
        arr.min(),
        arr.std(),
        (arr > zscore_threshold).mean(),
        (arr < -zscore_threshold).mean(),
    ], dtype=np.float32)
