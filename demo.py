"""Compare every classifier on a synthetic two-class dataset."""
import numpy as np
from mlscratch import *

rng = np.random.default_rng(42)
n = 300
X = np.r_[rng.normal([-1, -1], 1.2, (n, 2)), rng.normal([1, 1], 1.2, (n, 2))]
y = np.r_[np.zeros(n, int), np.ones(n, int)]
idx = rng.permutation(len(X)); tr, te = idx[:400], idx[400:]

for m in (LogisticRegression(), KNN(5), DecisionTree(4), GaussianNB()):
    acc = (m.fit(X[tr], y[tr]).predict(X[te]) == y[te]).mean()
    print(f"{type(m).__name__:20s} test accuracy: {acc:.1%}")
km = KMeans(2).fit(X)
print(f"{'KMeans':20s} inertia: {km.inertia:.1f}")
print(f"{'PCA':20s} variance explained: {PCA(1).fit(X).explained_variance_ratio[0]:.1%}")
