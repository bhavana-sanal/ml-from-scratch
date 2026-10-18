import numpy as np
from mlscratch import *

rng = np.random.default_rng(0)


def blobs(n=100):
    X = np.r_[rng.normal(-2, 0.7, (n, 2)), rng.normal(2, 0.7, (n, 2))]
    return X, np.r_[np.zeros(n, int), np.ones(n, int)]


def test_linear_regression_recovers_weights():
    X = rng.normal(size=(200, 3)); w = np.array([1.5, -2.0, 0.5])
    m = LinearRegression().fit(X, X @ w + 4)
    assert np.allclose(m.w, w) and np.isclose(m.b, 4)


def test_classifiers_separate_blobs():
    X, y = blobs()
    for m in (LogisticRegression(), KNN(3), DecisionTree(3), GaussianNB()):
        assert (m.fit(X, y).predict(X) == y).mean() > 0.97, type(m).__name__


def test_kmeans_finds_two_clusters():
    X, y = blobs()
    lab = KMeans(2).fit(X).predict(X)
    assert max((lab == y).mean(), (lab != y).mean()) > 0.97


def test_pca_roundtrip_and_variance():
    X = rng.normal(size=(100, 2)) @ np.array([[3, 0, 0], [0, 1, 0.1]])
    p = PCA(2).fit(X)
    assert np.allclose(p.inverse_transform(p.transform(X)), X)
    assert p.explained_variance_ratio[0] > 0.8
