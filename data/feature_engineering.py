import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_classif
import scipy.sparse as sp


AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")


# get amino acid composition
def get_aac_features(train_data, y_train, dev_data, test_data, k=10):
    # vectorizer with 20 amino acids
    vec = CountVectorizer (
        vocabulary=AA_LIST,
        analyzer="char",
        lowercase=False,
    )
    
    # calculate counts
    X_train_counts = vec.fit_transform(train_data)
    X_dev_counts = vec.transform(dev_data)
    X_test_counts = vec.transform(test_data)
    
    # Use 1D array (.ravel()) and ensure minimum length of 1 to avoid division by zero
    train_lengths = np.maximum(np.array([len(data) for data in train_data]), 1)
    dev_lengths = np.maximum(np.array([len(data) for data in dev_data]), 1)
    test_lengths = np.maximum(np.array([len(data) for data in test_data]), 1)
    
    # get relative frequencies (composition)
    X_train_aac = sp.diags(1.0 / train_lengths) @ X_train_counts
    X_dev_aac = sp.diags(1.0 / dev_lengths) @ X_dev_counts
    X_test_aac = sp.diags(1.0 / test_lengths) @ X_test_counts
       
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train_aac, y_train) # Sparse -> K dimensions
    X_dev_selected = selector.transform(X_dev_aac)
    X_test_selected = selector.transform(X_test_aac)

    return (
        X_train_selected.toarray().astype(np.float32),
        X_dev_selected.toarray().astype(np.float32),
        X_test_selected.toarray().astype(np.float32),
    )

# get dipeptide frequency
def get_dipeptide_features(train_data, y_train, dev_data, test_data, k=200):
    vec_2mer = CountVectorizer (
        analyzer="char",
        ngram_range=(2, 2),
        lowercase=False,
    )
    
    # calculate counts
    X_train_counts = vec_2mer.fit_transform(train_data)
    X_dev_counts = vec_2mer.transform(dev_data)
    X_test_counts = vec_2mer.transform(test_data)
    
    train_lengths = np.array([max(len(data) - 1, 1) for data in train_data], dtype=np.float32)
    dev_lengths = np.array([max(len(data) - 1, 1) for data in dev_data], dtype=np.float32)
    test_lengths = np.array([max(len(data) - 1, 1) for data in test_data], dtype=np.float32)

    X_train = sp.diags(1.0 / train_lengths) @ X_train_counts
    X_dev = sp.diags(1.0 / dev_lengths) @ X_dev_counts
    X_test = sp.diags(1.0 / test_lengths) @ X_test_counts

    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)  # <-- y_train added here
    X_dev_selected = selector.transform(X_dev)
    X_test_selected = selector.transform(X_test)

    return (
        X_train_selected.toarray().astype(np.float32),
        X_dev_selected.toarray().astype(np.float32),
        X_test_selected.toarray().astype(np.float32),
    )


# get tripeptide frequency
def get_tripeptide_frequency(train_data, y_train, dev_data, test_data, k=200):
    vec_3mer = CountVectorizer (
        analyzer="char",
        ngram_range=(3, 3),
        lowercase=False,
    )
    
    # calculate counts
    X_train_counts = vec_3mer.fit_transform(train_data)
    X_dev_counts = vec_3mer.transform(dev_data)
    X_test_counts = vec_3mer.transform(test_data)
    
    train_lengths = np.array([max(len(data) - 2, 1) for data in train_data], dtype=np.float32)
    dev_lengths = np.array([max(len(data) - 2, 1) for data in dev_data], dtype=np.float32)
    test_lengths = np.array([max(len(data) - 2, 1) for data in test_data], dtype=np.float32)

    X_train = sp.diags(1.0 / train_lengths) @ X_train_counts
    X_dev = sp.diags(1.0 / dev_lengths) @ X_dev_counts
    X_test = sp.diags(1.0 / test_lengths) @ X_test_counts

    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)  # <-- y_train added here
    X_dev_selected = selector.transform(X_dev)
    X_test_selected = selector.transform(X_test)

    return (
        X_train_selected.toarray().astype(np.float32),
        X_dev_selected.toarray().astype(np.float32),
        X_test_selected.toarray().astype(np.float32),
    )


# embedding using 3-mer vecotrizer
def get_svd_embedding_features (train_data, dev_data, test_data, n_components=128, random_state=42):
    vec_3mer = CountVectorizer (
        analyzer="char",
        ngram_range=(3, 3),
        lowercase=False,
    )
    
    # vectorization (counting)
    X_train = vec_3mer.fit_transform(train_data)
    X_dev = vec_3mer.transform(dev_data)
    X_test = vec_3mer.transform(test_data)
    
    # dimensionality reduction
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    
    X_train_emb = svd.fit_transform(X_train)
    X_dev_emb = svd.transform(X_dev)
    X_test_emb = svd.transform(X_test)
    
    
    return X_train_emb, X_dev_emb, X_test_emb
    