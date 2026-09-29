import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD


AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")


# get amino acid composition
def get_aac_features(train_data, dev_data, test_data):
    # vectorizer with 20 amino acids
    vec = CountVectorizer (
        vocabulary=AA_LIST,
        analyzer="char",
        lowercase=False,
    )
    
    # calculate counts
    X_train_counts = vec.fit_transform(train_data).toarray()
    X_dev_counts = vec.transform(dev_data).toarray()
    X_test_counts = vec.transform(test_data).toarray()
    
    # get relative frequencies (composition)
    X_train_aac = X_train_counts / np.array([len(data) for data in train_data]).reshape(-1, 1)
    X_dev_aac = X_dev_counts / np.array([len(data) for data in dev_data]).reshape(-1, 1)
    X_test_aac = X_test_counts / np.array([len(data) for data in test_data]).reshape(-1, 1)
    
    
    return X_train_aac, X_dev_aac, X_test_aac


# get dipeptide frequency
def get_dipeptide_features(train_data, dev_data, test_data):
    vec_2mer = CountVectorizer (
        analyzer="char",
        ngram_range=(2, 2),
        lowercase=False,
    )
    
    # calculate counts
    X_train_counts = vec_2mer.fit_transform(train_data).toarray()
    X_dev_counts = vec_2mer.transform(dev_data).toarray()
    X_test_counts = vec_2mer.transform(test_data).toarray()
    
    # number of possible dipeptides
    train_lenghts = np.array([max(len(data) - 1, 1) for data in train_data]).reshape(-1, 1)
    dev_lengths = np.array([max(len(data) - 1, 1) for data in dev_data]).reshape(-1, 1)
    test_lengths = np.array([max(len(data) - 1, 1) for data in test_data]).reshape(-1, 1)
    
    # normalize by total number of dipeptides per sequence (length - 1)
    X_train_dipep = X_train_counts / train_lenghts
    X_dev_dipep = X_dev_counts / dev_lengths
    X_test_dipep = X_test_counts / test_lengths
    
    return X_train_dipep, X_dev_dipep, X_test_dipep


# get tripeptide frequency
def get_tripeptide_frequency(train_data, dev_data, test_data):
    vec_3mer = CountVectorizer (
        analyzer="char",
        ngram_range=(3, 3),
        lowercase=False,
    )
    
    # calculate counts
    X_train_counts = vec_3mer.fit_transform(train_data).toarray()
    X_dev_counts = vec_3mer.transform(dev_data).toarray()
    X_test_counts = vec_3mer.transform(test_data).toarray()
    
    # number of possible tripeptides
    train_lenths = np.array([max(len(data) - 2, 1) for data in train_data]).reshape(-1, 1)
    dev_lengths = np.array([max(len(data) - 2, 1) for data in dev_data]).reshape(-1, 1)
    test_lengths = np.array([max(len(data) - 2, 1) for data in test_data]).reshape(-1, 1)
    
    # normalize by total number of tripeptides per sequence (lenght - 3)
    X_train_tripep = X_train_counts / train_lenths
    X_dev_tripep = X_dev_counts / dev_lengths
    X_test_tripep = X_test_counts / test_lengths
    
    return X_train_tripep, X_dev_tripep, X_test_tripep


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
    