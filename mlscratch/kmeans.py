import numpy as np


class KMeans:
    """Lloyd's algorithm with k-means++ initialisation."""

    def __init__(self, k=3, max_iter=100, seed=0):
        self.k, self.max_iter, self.rng = k, max_iter, np.random.default_rng(seed)

    def _init(self, X):
        c = [X[self.rng.integers(len(X))]]
        for _ in range(self.k - 1):
            d = np.min([((X - ci) ** 2).sum(1) for ci in c], axis=0)
            c.append(X[self.rng.choice(len(X), p=d / d.sum())])
        return np.array(c)

    def fit(self, X):
        self.centers = self._init(X)
        for _ in range(self.max_iter):
            labels = self.predict(X)
            new = np.array([X[labels == j].mean(0) if (labels == j).any() else self.centers[j] for j in range(self.k)])
            if np.allclose(new, self.centers):
                break
            self.centers = new
        self.inertia = ((X - self.centers[self.predict(X)]) ** 2).sum()
        return self

    def predict(self, X):
        return ((X[:, None, :] - self.centers[None]) ** 2).sum(-1).argmin(1)
