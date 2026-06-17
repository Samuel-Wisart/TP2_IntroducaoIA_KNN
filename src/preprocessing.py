"""Feature preprocessing helpers for distance-based algorithms."""

from __future__ import annotations

import numpy as np
import pandas as pd


class StandardScaler:
    """Minimal standard scaler implemented with NumPy."""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, x):
        x = self._to_array(x)
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        return self

    def transform(self, x):
        x = self._to_array(x)
        return (x - self.mean_) / self.scale_

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    @staticmethod
    def _to_array(x):
        if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            return x.to_numpy(dtype=float)
        return np.asarray(x, dtype=float)


def as_dataframe(features, columns=None):
    """Return a DataFrame copy when possible for nicer reporting."""
    if isinstance(features, pd.DataFrame):
        return features.copy()
    return pd.DataFrame(features, columns=columns)
