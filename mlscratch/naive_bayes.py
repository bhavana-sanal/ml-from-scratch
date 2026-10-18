import numpy as np


class GaussianNB:
    """Gaussian Naive Bayes classifier."""

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mu = np.array([X[y == c].mean(0) for c in self.classes])
        self.var = np.array([X[y == c].var(0) for c in self.classes]) + 1e-9
        self.prior = np.array([(y == c).mean() for c in self.classes])
        return self

    def predict(self, X):
        ll = -0.5 * (np.log(2 * np.pi * self.var)[None] + (X[:, None, :] - self.mu[None]) ** 2 / self.var[None]).sum(-1)
        return self.classes[(ll + np.log(self.prior)).argmax(1)]
