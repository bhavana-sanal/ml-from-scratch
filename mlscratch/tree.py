import numpy as np


def _gini(y):
    p = np.bincount(y) / len(y)
    return 1 - (p ** 2).sum()


class DecisionTree:
    """CART classifier using Gini impurity."""

    def __init__(self, max_depth=5, min_samples=2):
        self.max_depth, self.min_samples = max_depth, min_samples

    def fit(self, X, y):
        self.root = self._grow(np.asarray(X), np.asarray(y), 0)
        return self

    def _best_split(self, X, y):
        best, base = (None, None, 0.0), _gini(y)
        for j in range(X.shape[1]):
            for t in np.unique(X[:, j])[:-1]:
                m = X[:, j] <= t
                gain = base - (m.mean() * _gini(y[m]) + (1 - m.mean()) * _gini(y[~m]))
                if gain > best[2]:
                    best = (j, t, gain)
        return best

    def _grow(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples or _gini(y) == 0:
            return int(np.bincount(y).argmax())
        j, t, gain = self._best_split(X, y)
        if j is None:
            return int(np.bincount(y).argmax())
        m = X[:, j] <= t
        return (j, t, self._grow(X[m], y[m], depth + 1), self._grow(X[~m], y[~m], depth + 1))

    def _one(self, x, node):
        while isinstance(node, tuple):
            j, t, l, r = node
            node = l if x[j] <= t else r
        return node

    def predict(self, X):
        return np.array([self._one(x, self.root) for x in X])
