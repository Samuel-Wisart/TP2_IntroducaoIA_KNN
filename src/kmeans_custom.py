"""Custom k-Means clustering skeleton."""

from __future__ import annotations

import numpy as np


class KMeansClusterer:
    """Simple k-Means implementation to be completed with the project logic."""

    def __init__(self, k: int = 2, max_iter: int = 100, random_state: int | None = 42):
        self.k = k
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None

    def fit(self, x):
        """Fit the model to the feature matrix."""
        x = np.asarray(x, dtype=float)
        rng = np.random.default_rng(self.random_state)
        initial_indices = rng.choice(len(x), size=self.k, replace=False)
        self.centroids = x[initial_indices]

        for _ in range(self.max_iter):
            distances = np.linalg.norm(x[:, None, :] - self.centroids[None, :, :], axis=2)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([
                x[labels == cluster_id].mean(axis=0) if np.any(labels == cluster_id) else self.centroids[cluster_id]
                for cluster_id in range(self.k)
            ])
            if np.allclose(new_centroids, self.centroids):
                self.centroids = new_centroids
                self.labels_ = labels
                break
            self.centroids = new_centroids
            self.labels_ = labels

        return self

    def predict(self, x):
        """Assign each sample to the nearest centroid."""
        x = np.asarray(x, dtype=float)
        distances = np.linalg.norm(x[:, None, :] - self.centroids[None, :, :], axis=2)
        return np.argmin(distances, axis=1)
