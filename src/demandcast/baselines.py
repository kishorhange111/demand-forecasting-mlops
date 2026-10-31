"""Simple forecasts that a learned model has to beat. Each maps past sales to a (series, 28) forecast."""

from __future__ import annotations

import numpy as np

from .data import HORIZON


def naive(train_sales: np.ndarray) -> np.ndarray:
    """Repeat the last observed day."""
    return np.repeat(train_sales[:, -1:], HORIZON, axis=1).astype(np.float64)


def seasonal_naive(train_sales: np.ndarray) -> np.ndarray:
    """Repeat the last observed week. This is M5's official sNaive benchmark."""
    last_week = train_sales[:, -7:]
    return np.tile(last_week, (1, HORIZON // 7)).astype(np.float64)


def moving_average(train_sales: np.ndarray, window: int = 28) -> np.ndarray:
    """A flat forecast at the recent mean."""
    return np.repeat(train_sales[:, -window:].mean(axis=1, keepdims=True), HORIZON, axis=1)


def weekday_average(train_sales: np.ndarray, weeks: int = 4) -> np.ndarray:
    """Mean of the same weekday over the last few weeks."""
    recent = train_sales[:, -7 * weeks :].astype(np.float64)
    profile = recent.reshape(recent.shape[0], weeks, 7).mean(axis=1)
    return np.tile(profile, (1, HORIZON // 7))


BASELINES = {
    "naive": naive,
    "seasonal_naive": seasonal_naive,
    "moving_average_28": moving_average,
    "weekday_average_4w": weekday_average,
}
