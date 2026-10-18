import numpy as np


class PCA:
    """Principal component analysis via SVD of the centred data."""

    def __init__(self, n_components=2):
        self.n = n_components

    def fit(self, X):
        self.mean = X.mean(0)
        _, s, vt = np.linalg.svd(X - self.mean, full_matrices=False)
        self.components = vt[: self.n]
        self.explained_variance_ratio = (s ** 2 / (s ** 2).sum())[: self.n]
        return self

    def transform(self, X):
        return (X - self.mean) @ self.components.T

    def inverse_transform(self, Z):
        return Z @ self.components + self.mean
