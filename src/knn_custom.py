"""Custom k-Nearest Neighbors classifier skeleton."""

from __future__ import annotations

import numpy as np


class KNNClassifier:
    """Simple kNN classifier to be completed with the project logic."""

    def __init__(self, k: int = 3):
        self.k = k
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train):
        """Store the training data."""
        self.x_train = np.asarray(x_train)
        self.y_train = np.asarray(y_train)
        return self

    def predict(self, x_test):
        """Predict labels for the test set."""
        x_test = np.asarray(x_test)
        predictions = [self._predict_one(row) for row in x_test]
        return np.asarray(predictions)

    def _predict_one(self, row):
        """Predict a single sample using Euclidean distance and majority vote."""
        distances = np.linalg.norm(self.x_train - row, axis=1)
        nearest_indices = np.argsort(distances)[: self.k]
        nearest_labels = self.y_train[nearest_indices]
        values, counts = np.unique(nearest_labels, return_counts=True)
        return values[np.argmax(counts)]
