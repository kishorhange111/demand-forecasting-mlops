"""Prediction intervals by split conformal prediction, against a Poisson baseline.

A replenishment system turns forecasts into safety stock, so what matters is whether "95 %" is really 95 %.
Point-forecast residuals from an earlier fold calibrate the width, separately for each band of forecast
magnitude, because a fast seller's errors are far larger than a slow seller's.

Coverage from this is approximate rather than the textbook guarantee: consecutive days of demand are not
exchangeable. That's why calibration runs on an earlier window and coverage is measured on a later one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

LEVELS = (0.50, 0.80, 0.95)


@dataclass
class ConformalIntervals:
    edges: np.ndarray
    offsets: dict[float, tuple[np.ndarray, np.ndarray]]  # level -> (below, above) per band

    @classmethod
    def calibrate(cls, forecast: np.ndarray, actual: np.ndarray, bands: int = 10, levels: tuple[float, ...] = LEVELS) -> ConformalIntervals:
        flat_forecast, residual = forecast.ravel(), (actual - forecast).ravel()
        edges = np.unique(np.quantile(flat_forecast, np.linspace(0, 1, bands + 1)[1:-1]))
        band = np.searchsorted(edges, flat_forecast, side="right")
        offsets = {}
        for level in levels:
            tail = (1 - level) / 2
            below, above = np.zeros(len(edges) + 1), np.zeros(len(edges) + 1)
            for index in range(len(edges) + 1):
                members = residual[band == index]
                if len(members) == 0:
                    continue
                # The (n + 1) / n inflation is the finite-sample correction of split conformal.
                n = len(members)
                below[index] = np.quantile(members, max(tail * (n + 1) / n - 1 / n, 0.0))
                above[index] = np.quantile(members, min((1 - tail) * (n + 1) / n, 1.0))
            offsets[level] = (below, above)
        return cls(edges, offsets)

    def predict(self, forecast: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
        below, above = self.offsets[level]
        band = np.searchsorted(self.edges, forecast, side="right")
        return np.maximum(forecast + below[band], 0.0), forecast + above[band]


def poisson_interval(forecast: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """The interval implied by assuming demand is Poisson around the forecast."""
    tail = (1 - level) / 2
    rate = np.maximum(forecast, 1e-9)
    return stats.poisson.ppf(tail, rate), stats.poisson.ppf(1 - tail, rate)


def coverage_report(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray, forecast: np.ndarray) -> dict:
    """Empirical coverage overall and by demand volume, with the mean width that coverage costs."""
    inside = (actual >= lower) & (actual <= upper)
    report = {"coverage": float(inside.mean()), "mean_width": float((upper - lower).mean())}
    high = forecast >= np.quantile(forecast, 0.9)
    report["coverage_top_decile_volume"] = float(inside[high].mean())
    report["coverage_other_volume"] = float(inside[~high].mean()) if (~high).any() else None
    return report
