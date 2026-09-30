from itertools import product

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import normalize

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

# Fixed vocabularies: only the 20 standard amino acids, so rare characters
# (X, U, B, Z, ...) can never create junk columns.
VOCAB_1MER = AA_LIST
VOCAB_2MER = ["".join(p) for p in product(AA_LIST, repeat=2)]  # 400
VOCAB_3MER = ["".join(p) for p in product(AA_LIST, repeat=3)]  # 8000

_VOCABS = {1: VOCAB_1MER, 2: VOCAB_2MER, 3: VOCAB_3MER}


def _kmer_frequencies(train_data, dev_data, test_data, n):
    """Returns L1-normalized n-mer frequency matrices (sparse, float32) for each split."""
    vec = CountVectorizer(
        vocabulary=_VOCABS[n],
        analyzer="char",
        ngram_range=(n, n),
        lowercase=False,
        dtype=np.float32,
    )
    # With a fixed vocabulary, fit() learns nothing, so transform() is enough
    # and we avoid a separate pass over the training data.
    X_train = vec.transform(train_data)
    X_dev = vec.transform(dev_data)
    X_test = vec.transform(test_data)

    # L1 normalization divides each row by its total k-mer count, giving
    # frequencies. Empty rows stay zero (no division-by-zero problem).
    return (
        normalize(X_train, norm="l1", copy=False),
        normalize(X_dev, norm="l1", copy=False),
        normalize(X_test, norm="l1", copy=False),
    )


def _select_and_densify(X_train, y_train, X_dev, X_test, k):
    """Optionally keeps the top-k features (fit on train only) and returns dense float32 arrays."""
    if k is not None and k < X_train.shape[1]:
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train = selector.fit_transform(X_train, y_train)
        X_dev = selector.transform(X_dev)
        X_test = selector.transform(X_test)

    return tuple(
        (X.toarray() if sp.issparse(X) else np.asarray(X)).astype(np.float32, copy=False)
        for X in (X_train, X_dev, X_test)
    )


# amino acid composition (20 features)
def get_aac_features(train_data, y_train, dev_data, test_data, k=None):
    """k=None keeps all 20 amino acid frequencies."""
    X_train, X_dev, X_test = _kmer_frequencies(train_data, dev_data, test_data, n=1)
    return _select_and_densify(X_train, y_train, X_dev, X_test, k)


# dipeptide frequency (400 possible -> top k)
def get_dipeptide_features(train_data, y_train, dev_data, test_data, k=200):
    X_train, X_dev, X_test = _kmer_frequencies(train_data, dev_data, test_data, n=2)
    return _select_and_densify(X_train, y_train, X_dev, X_test, k)


# tripeptide frequency (8000 possible -> top k)
def get_tripeptide_frequency(train_data, y_train, dev_data, test_data, k=200):
    X_train, X_dev, X_test = _kmer_frequencies(train_data, dev_data, test_data, n=3)
    return _select_and_densify(X_train, y_train, X_dev, X_test, k)


# SVD embedding of 3-mer frequencies (unsupervised, no labels needed)
def get_svd_embedding_features(train_data, dev_data, test_data, n_components=128, random_state=42):
    X_train, X_dev, X_test = _kmer_frequencies(train_data, dev_data, test_data, n=3)

    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    X_train_emb = svd.fit_transform(X_train)
    X_dev_emb = svd.transform(X_dev)
    X_test_emb = svd.transform(X_test)

    return (
        X_train_emb.astype(np.float32),
        X_dev_emb.astype(np.float32),
        X_test_emb.astype(np.float32),
    )


# all hand-crafted features side by side
def get_combined_features(train_data, y_train, dev_data, test_data, k_di=200, k_tri=200):
    """AAC (20) + top-k dipeptides + top-k tripeptides, concatenated."""
    parts = [
        get_aac_features(train_data, y_train, dev_data, test_data),
        get_dipeptide_features(train_data, y_train, dev_data, test_data, k=k_di),
        get_tripeptide_frequency(train_data, y_train, dev_data, test_data, k=k_tri),
    ]
    return tuple(np.hstack([part[i] for part in parts]) for i in range(3))
