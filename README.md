# ml-from-scratch

[![CI](https://github.com/bhavana-sanal/ml-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/bhavana-sanal/ml-from-scratch/actions) ![License](https://img.shields.io/badge/license-MIT-blue)

Classic machine-learning algorithms implemented from first principles with **NumPy only** — no scikit-learn.
Each model is a single readable file with a `fit` / `predict` API.

| Algorithm | File | Notes |
|---|---|---|
| Linear regression | `linear.py` | normal equation via pseudo-inverse |
| Logistic regression | `linear.py` | gradient descent, L2 regularisation |
| k-Nearest Neighbours | `knn.py` | vectorised pairwise distances |
| K-Means | `kmeans.py` | k-means++ initialisation |
| Decision tree (CART) | `tree.py` | Gini impurity, recursive splitting |
| Gaussian Naive Bayes | `naive_bayes.py` | log-space likelihoods |
| PCA | `pca.py` | SVD, explained-variance ratio |

```bash
pip install -r requirements.txt
python demo.py     # accuracy comparison on synthetic data
pytest             # unit tests
```

```python
from mlscratch import DecisionTree
clf = DecisionTree(max_depth=4).fit(X_train, y_train)
print((clf.predict(X_test) == y_test).mean())
```
