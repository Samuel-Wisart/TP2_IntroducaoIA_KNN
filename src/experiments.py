"""End-to-end experiment helpers for kNN and k-Means."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .compare_sklearn import build_sklearn_kmeans, build_sklearn_knn
from .data_loader import TARGET_COLUMN, load_full_dataset, load_train_test, split_features_target
from .kmeans_custom import KMeansClusterer
from .knn_custom import KNNClassifier
from .metrics import accuracy_score, confusion_matrix_binary, f1_score, precision_score, recall_score
from .preprocessing import StandardScaler


@dataclass
class ExperimentPaths:
    root: Path
    results_dir: Path


def ensure_results_dir(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)


def _scale_train_test(x_train, x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled, scaler


def _scale_full_dataframe(features: pd.DataFrame):
    scaler = StandardScaler()
    return scaler.fit_transform(features), scaler


def run_knn_experiment(train_path: str | Path, test_path: str | Path, k_values: list[int], results_dir: str | Path | None = None) -> list[dict[str, Any]]:
    x_train, x_test, y_train, y_test = load_train_test(train_path, test_path)
    x_train_scaled, x_test_scaled, _ = _scale_train_test(x_train, x_test)

    results = []
    for k in k_values:
        model = KNNClassifier(k=k).fit(x_train_scaled, y_train.to_numpy())
        predictions = model.predict(x_test_scaled)
        matrix = confusion_matrix_binary(y_test, predictions)
        result = {
            "algorithm": "custom_knn",
            "k": k,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "f1": f1_score(y_test, predictions),
            "confusion_matrix": matrix.tolist(),
            "predictions": predictions.tolist(),
            "y_true": y_test.to_list(),
        }
        results.append(result)

        if results_dir is not None:
            path = Path(results_dir)
            ensure_results_dir(path)
            pd.DataFrame(matrix, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(path / f"knn_confusion_matrix_k{k}.csv")
            pd.DataFrame([result]).drop(columns=["predictions", "y_true"]).to_json(path / f"knn_metrics_k{k}.json", orient="records", indent=2)
    return results


def run_knn_sklearn_comparison(train_path: str | Path, test_path: str | Path, k_values: list[int], results_dir: str | Path | None = None) -> list[dict[str, Any]]:
    x_train, x_test, y_train, y_test = load_train_test(train_path, test_path)
    x_train_scaled, x_test_scaled, _ = _scale_train_test(x_train, x_test)

    results = []
    for k in k_values:
        model = build_sklearn_knn(k)
        model.fit(x_train_scaled, y_train)
        predictions = model.predict(x_test_scaled)
        matrix = confusion_matrix_binary(y_test, predictions)
        result = {
            "algorithm": "sklearn_knn",
            "k": k,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "f1": f1_score(y_test, predictions),
            "confusion_matrix": matrix.tolist(),
            "predictions": predictions.tolist(),
            "y_true": y_test.to_list(),
        }
        results.append(result)

        if results_dir is not None:
            path = Path(results_dir)
            ensure_results_dir(path)
            pd.DataFrame(matrix, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(path / f"sklearn_knn_confusion_matrix_k{k}.csv")
            pd.DataFrame([result]).drop(columns=["predictions", "y_true"]).to_json(path / f"sklearn_knn_metrics_k{k}.json", orient="records", indent=2)
    return results


def run_kmeans_experiment(train_path: str | Path, test_path: str | Path, k_values: list[int], results_dir: str | Path | None = None) -> list[dict[str, Any]]:
    full_df = load_full_dataset(train_path, test_path)
    features, target = split_features_target(full_df)
    scaled_features, _ = _scale_full_dataframe(features)

    results = []
    for k in k_values:
        model = KMeansClusterer(k=k)
        model.fit(scaled_features)
        labels = model.labels_
        centroids_scaled = model.centroids

        centroid_frame = pd.DataFrame(centroids_scaled, columns=features.columns)
        cluster_summary = pd.crosstab(labels, target, rownames=["cluster"], colnames=[TARGET_COLUMN])

        result = {
            "algorithm": "custom_kmeans",
            "k": k,
            "centroids_scaled": centroid_frame.to_dict(orient="records"),
            "cluster_summary": cluster_summary.to_dict(),
            "labels": labels.tolist(),
        }
        results.append(result)

        if results_dir is not None:
            path = Path(results_dir)
            ensure_results_dir(path)
            centroid_frame.to_csv(path / f"kmeans_centroids_k{k}.csv", index=False)
            cluster_summary.to_csv(path / f"kmeans_cluster_summary_k{k}.csv")
            with open(path / f"kmeans_result_k{k}.json", "w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)
    return results


def run_kmeans_sklearn_comparison(train_path: str | Path, test_path: str | Path, k_values: list[int], results_dir: str | Path | None = None) -> list[dict[str, Any]]:
    full_df = load_full_dataset(train_path, test_path)
    features, target = split_features_target(full_df)
    scaled_features, _ = _scale_full_dataframe(features)

    results = []
    for k in k_values:
        model = build_sklearn_kmeans(k)
        labels = model.fit_predict(scaled_features)
        centroid_frame = pd.DataFrame(model.cluster_centers_, columns=features.columns)
        cluster_summary = pd.crosstab(labels, target, rownames=["cluster"], colnames=[TARGET_COLUMN])
        result = {
            "algorithm": "sklearn_kmeans",
            "k": k,
            "centroids_scaled": centroid_frame.to_dict(orient="records"),
            "cluster_summary": cluster_summary.to_dict(),
            "labels": labels.tolist(),
        }
        results.append(result)

        if results_dir is not None:
            path = Path(results_dir)
            ensure_results_dir(path)
            centroid_frame.to_csv(path / f"sklearn_kmeans_centroids_k{k}.csv", index=False)
            cluster_summary.to_csv(path / f"sklearn_kmeans_cluster_summary_k{k}.csv")
            with open(path / f"sklearn_kmeans_result_k{k}.json", "w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)
    return results
