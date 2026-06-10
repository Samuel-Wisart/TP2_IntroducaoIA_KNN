"""Utilities for loading and preparing the NBA rookie dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


TARGET_COLUMN = "TARGET_5Yrs"


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset into a pandas DataFrame."""
    return pd.read_csv(path)


def split_features_target(dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into feature matrix and target vector."""
    features = dataframe.drop(columns=[TARGET_COLUMN])
    target = dataframe[TARGET_COLUMN]
    return features, target


def load_train_test(train_path: str | Path, test_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the training and test files and return X/y parts for both."""
    train_df = load_dataset(train_path)
    test_df = load_dataset(test_path)

    x_train, y_train = split_features_target(train_df)
    x_test, y_test = split_features_target(test_df)
    return x_train, x_test, y_train, y_test


def load_full_dataset(train_path: str | Path, test_path: str | Path) -> pd.DataFrame:
    """Join the two provided files into a single dataframe."""
    train_df = load_dataset(train_path)
    test_df = load_dataset(test_path)
    return pd.concat([train_df, test_df], ignore_index=True)
