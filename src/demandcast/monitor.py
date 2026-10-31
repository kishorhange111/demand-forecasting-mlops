"""Drift signals and the decision they feed: is it time to retrain?"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PSI_MODERATE = 0.1
PSI_MAJOR = 0.2


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population stability index, with bins taken from the reference's quantiles. NaNs get their own bin."""
    reference_valid = reference[~np.isnan(reference)]
    if len(reference_valid) == 0:
        return 0.0
    edges = np.unique(np.quantile(reference_valid, np.linspace(0, 1, bins + 1)[1:-1]))

    def shares(values: np.ndarray) -> np.ndarray:
        missing = np.isnan(values)
        counts = np.bincount(np.searchsorted(edges, values[~missing], side="right"), minlength=len(edges) + 1).astype(np.float64)
        counts = np.append(counts, missing.sum())
        # A small floor keeps an empty bin from making the index infinite.
        return np.maximum(counts / max(len(values), 1), 1e-4)

    expected, actual = shares(reference), shares(current)
    return float(((actual - expected) * np.log(actual / expected)).sum())


def feature_psi(reference: np.ndarray, current: np.ndarray, names: list[str]) -> dict[str, float]:
    return {name: psi(reference[:, i], current[:, i]) for i, name in enumerate(names)}


@dataclass
class RetrainPolicy:
    """Retrain when the error has worsened against the last good reference, or inputs have shifted a lot."""

    error_increase: float = 0.10
    psi_threshold: float = PSI_MAJOR
    min_features_over: int = 3

    def decide(self, reference_error: float | None, latest_error: float | None, psi_by_feature: dict[str, float]) -> dict:
        over = sorted(name for name, value in psi_by_feature.items() if value > self.psi_threshold)
        error_ratio = None if not reference_error or latest_error is None else latest_error / reference_error
        reasons = []
        if error_ratio is not None and error_ratio > 1 + self.error_increase:
            reasons.append(f"error up {error_ratio - 1:.1%} against the reference")
        if len(over) >= self.min_features_over:
            reasons.append(f"{len(over)} features over PSI {self.psi_threshold}")
        return {"retrain": bool(reasons), "reasons": reasons, "error_ratio": error_ratio, "features_over_psi": over}
