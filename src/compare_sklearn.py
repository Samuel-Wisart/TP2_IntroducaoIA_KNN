"""Comparison helpers using scikit-learn."""

from __future__ import annotations

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier


def build_sklearn_knn(k: int):
    """Create a scikit-learn KNN classifier with the requested k."""
    return KNeighborsClassifier(n_neighbors=k)


def build_sklearn_kmeans(k: int, random_state: int = 42):
    """Create a scikit-learn KMeans model."""
    return KMeans(n_clusters=k, random_state=random_state, n_init="auto")
