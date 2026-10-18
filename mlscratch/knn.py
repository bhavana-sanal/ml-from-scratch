import numpy as np


class KNN:
    """k-nearest-neighbours classifier (Euclidean distance, majority vote)."""

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X, self.y = np.asarray(X), np.asarray(y)
        return self

    def predict(self, X):
        d = ((np.asarray(X)[:, None, :] - self.X[None, :, :]) ** 2).sum(-1)
        idx = np.argsort(d, axis=1)[:, : self.k]
        return np.array([np.bincount(row).argmax() for row in self.y[idx]])
