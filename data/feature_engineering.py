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


def _select_and_densify(X_train, y_train, X_dev, X_test, k, names):
    """Optionally keeps the top-k features (fit on train only). Returns (dense float32 arrays, kept names)."""
    if k is not None and k < X_train.shape[1]:
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train = selector.fit_transform(X_train, y_train)
        X_dev = selector.transform(X_dev)
        X_test = selector.transform(X_test)
        names = [n for n, keep in zip(names, selector.get_support()) if keep]

    arrays = tuple(
        (X.toarray() if sp.issparse(X) else np.asarray(X)).astype(np.float32, copy=False)
        for X in (X_train, X_dev, X_test)
    )
    return arrays, names


def _build_kmer_features(train_data, y_train, dev_data, test_data, n, k, prefix, return_names):
    X_train, X_dev, X_test = _kmer_frequencies(train_data, dev_data, test_data, n=n)
    names = [f"{prefix}_{kmer}" for kmer in _VOCABS[n]]
    arrays, names = _select_and_densify(X_train, y_train, X_dev, X_test, k, names)
    return (arrays, names) if return_names else arrays


def get_aac_features(train_data, y_train, dev_data, test_data, k=None, return_names=False):
    return _build_kmer_features(train_data, y_train, dev_data, test_data, 1, k, "AAC", return_names)


def get_dipeptide_features(train_data, y_train, dev_data, test_data, k=200, return_names=False):
    return _build_kmer_features(train_data, y_train, dev_data, test_data, 2, k, "DIPEP", return_names)


def get_tripeptide_frequency(train_data, y_train, dev_data, test_data, k=200, return_names=False):
    return _build_kmer_features(train_data, y_train, dev_data, test_data, 3, k, "TRIPEP", return_names)


def get_svd_embedding_features(train_data, dev_data, test_data, n_components=128, random_state=42,
                               return_names=False):
    X_train, X_dev, X_test = _kmer_frequencies(train_data, dev_data, test_data, n=3)
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    arrays = (
        svd.fit_transform(X_train).astype(np.float32),
        svd.transform(X_dev).astype(np.float32),
        svd.transform(X_test).astype(np.float32),
    )
    names = [f"SVD_{i}" for i in range(n_components)]
    return (arrays, names) if return_names else arrays


def get_combined_features(train_data, y_train, dev_data, test_data, k_di=200, k_tri=200,
                          return_names=False):
    """AAC (20) + top-k dipeptides + top-k tripeptides, concatenated."""
    parts = [
        get_aac_features(train_data, y_train, dev_data, test_data, return_names=True),
        get_dipeptide_features(train_data, y_train, dev_data, test_data, k=k_di, return_names=True),
        get_tripeptide_frequency(train_data, y_train, dev_data, test_data, k=k_tri, return_names=True),
    ]
    arrays = tuple(np.hstack([arrs[i] for arrs, _ in parts]) for i in range(3))
    names = [n for _, part_names in parts for n in part_names]
    return (arrays, names) if return_names else arrays
