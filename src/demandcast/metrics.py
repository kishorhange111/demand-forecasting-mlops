"""WRMSSE, the official M5 Accuracy metric, over its 12 aggregation levels and 42,840 series.

Series are scaled by the in-sample squared error of a one-step naive forecast, counted from their first
non-zero sale, and weighted by dollar sales in the last 28 training days. Level 12 (item-store) alone would
reward predicting near zero for sparse items, which is why the metric also scores every aggregate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .data import HORIZON

LEVELS: list[list[str]] = [
    [],
    ["state_id"],
    ["store_id"],
    ["cat_id"],
    ["dept_id"],
    ["state_id", "cat_id"],
    ["state_id", "dept_id"],
    ["store_id", "cat_id"],
    ["store_id", "dept_id"],
    ["item_id"],
    ["item_id", "state_id"],
    ["item_id", "store_id"],
]


def _level_matrix(ids: pd.DataFrame, keys: list[str]) -> sp.csr_matrix:
    """Sparse matrix mapping bottom-level series to the aggregates of one level."""
    if keys:
        groups = ids[keys].astype(str).agg("|".join, axis=1)
        codes = pd.factorize(groups)[0]
    else:
        codes = np.zeros(len(ids), dtype=np.int64)
    n = len(ids)
    return sp.csr_matrix((np.ones(n, dtype=np.float64), (codes, np.arange(n))), shape=(codes.max() + 1, n))


class WRMSSE:
    """Scorer bound to one forecast origin. Everything it learns comes from before that origin."""

    def __init__(self, ids: pd.DataFrame, train_sales: np.ndarray, train_prices: np.ndarray):
        self.levels = [_level_matrix(ids, keys) for keys in LEVELS]
        dollars = (train_sales[:, -HORIZON:] * np.nan_to_num(train_prices[:, -HORIZON:])).sum(axis=1)
        self.weights: list[np.ndarray] = []
        self.scales: list[np.ndarray] = []
        for matrix in self.levels:
            level_dollars = matrix @ dollars
            self.weights.append(level_dollars / level_dollars.sum() / len(LEVELS))
            self.scales.append(_naive_scale(matrix @ train_sales))

    @property
    def n_series(self) -> int:
        return sum(m.shape[0] for m in self.levels)

    def per_level(self, forecast: np.ndarray, actual: np.ndarray) -> list[float]:
        """Each level's contribution scaled to a full weight, so 1.0 means as bad as the naive scale."""
        scores = []
        for matrix, weight, scale in zip(self.levels, self.weights, self.scales):
            error = matrix @ (forecast - actual)
            rmsse = np.sqrt((error**2).mean(axis=1) / scale)
            scores.append(float((weight * rmsse).sum() * len(LEVELS)))
        return scores

    def score(self, forecast: np.ndarray, actual: np.ndarray) -> float:
        return float(np.mean(self.per_level(forecast, actual)))


def _naive_scale(series: np.ndarray) -> np.ndarray:
    """Mean squared one-step difference, starting at each series' first non-zero observation."""
    started = np.maximum.accumulate(series != 0, axis=1)
    diffs = np.diff(series, axis=1) ** 2
    counted = started[:, :-1]  # a step counts if the previous day was already active
    totals = (diffs * counted).sum(axis=1)
    counts = np.maximum(counted.sum(axis=1), 1)
    # A series with no variation has no scale, and M5 gives such a series no weight. Avoid dividing by zero.
    return np.maximum(totals / counts, 1e-12)
