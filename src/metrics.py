"""Metric helpers for the custom kNN implementation."""

from __future__ import annotations

import numpy as np


def confusion_matrix_binary(y_true, y_pred):
    """Build a 2x2 confusion matrix for labels 0 and 1."""
    matrix = np.zeros((2, 2), dtype=int)
    for true_value, pred_value in zip(y_true, y_pred):
        matrix[int(true_value), int(pred_value)] += 1
    return matrix


def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def precision_score(y_true, y_pred):
    matrix = confusion_matrix_binary(y_true, y_pred)
    tp = matrix[1, 1]
    fp = matrix[0, 1]
    return float(tp / (tp + fp)) if (tp + fp) else 0.0


def recall_score(y_true, y_pred):
    matrix = confusion_matrix_binary(y_true, y_pred)
    tp = matrix[1, 1]
    fn = matrix[1, 0]
    return float(tp / (tp + fn)) if (tp + fn) else 0.0


def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0
