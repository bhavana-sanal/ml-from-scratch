import numpy as np


class LinearRegression:
    """Ordinary least squares via the normal equation (pseudo-inverse, so rank-deficient X is fine)."""

    def fit(self, X, y):
        A = np.c_[np.ones(len(X)), X]
        w = np.linalg.pinv(A) @ y
        self.b, self.w = w[0], w[1:]
        return self

    def predict(self, X):
        return X @ self.w + self.b


class LogisticRegression:
    """Binary logistic regression trained with batch gradient descent + L2 penalty."""

    def __init__(self, lr=0.1, epochs=1000, l2=0.0):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        n, d = X.shape
        self.w, self.b = np.zeros(d), 0.0
        for _ in range(self.epochs):
            err = self._sigmoid(X @ self.w + self.b) - y
            self.w -= self.lr * (X.T @ err / n + self.l2 * self.w)
            self.b -= self.lr * err.mean()
        return self

    def predict_proba(self, X):
        return self._sigmoid(X @ self.w + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
