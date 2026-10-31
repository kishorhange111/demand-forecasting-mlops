"""When should a model be retrained? Replay real history and measure the answer.

Walks forward through 28-day windows, as production would. At each window a strategy decides whether to retrain, using only
what it could know by then: sales up to that day, and the error of its last forecast, once those sales have come in.

- ``static``: trained once and never again. Lag features still refresh, so only the learned weights go stale.
- ``scheduled``: retrained at every window, the expensive upper bound.
- ``triggered``: retrained only when ``RetrainPolicy`` fires on drift or a rise in error.

Every strategy is scored against the same actual sales.

    demandcast-replay --out results
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import lightgbm as lgb
import numpy as np

from .backtest import scorer_at
from .data import HORIZON, M5, N_OBSERVED_DAYS, load
from .features import FEATURE_NAMES, Encodings, build_features, on_sale_mask
from .model import ModelConfig, fit, predict
from .monitor import RetrainPolicy, feature_psi

# Drift is watched only on inputs that can really shift. An item's ID or the day of the month changes by construction.
MONITORED = [
    "lag_28", "rmean_7", "rmean_28", "rmean_56", "rstd_28", "nonzero_28", "price", "price_relative", "price_change_28",
]  # fmt: skip
PSI_SAMPLE_SERIES = 5000


def monitored_features(m5: M5, encodings: Encodings, days: np.ndarray, origin: int, sample: np.ndarray) -> np.ndarray:
    matrix = build_features(m5, encodings, days, origin, sample)
    keep = on_sale_mask(m5, days, origin, sample)
    return matrix[keep][:, [FEATURE_NAMES.index(name) for name in MONITORED]]


class ModelCache:
    """Fits each origin at most once, since several strategies need the same model."""

    def __init__(self, m5: M5, encodings: Encodings, cfg: ModelConfig, log):
        self.m5, self.encodings, self.cfg, self.log = m5, encodings, cfg, log
        self.models: dict[int, lgb.Booster] = {}
        self.fit_seconds: dict[int, float] = {}

    def get(self, origin: int) -> lgb.Booster:
        if origin not in self.models:
            started = time.perf_counter()
            self.models[origin], _ = fit(self.m5, self.encodings, origin, self.cfg)
            self.fit_seconds[origin] = time.perf_counter() - started
            self.log(f"  fitted model at origin {origin} in {self.fit_seconds[origin]:.0f}s")
        return self.models[origin]


def replay(m5: M5, encodings: Encodings, origins: list[int], cfg: ModelConfig, policy: RetrainPolicy, log) -> dict:
    cache = ModelCache(m5, encodings, cfg, log)
    rng = np.random.default_rng(0)
    sample = np.sort(rng.choice(m5.n_series, size=min(PSI_SAMPLE_SERIES, m5.n_series), replace=False))

    def score(origin: int, trained_at: int) -> float:
        forecast = predict(cache.get(trained_at), m5, encodings, origin)
        actual = m5.sales[:, origin : origin + HORIZON].astype(np.float64)
        return scorer_at(m5, origin).score(forecast, actual)

    scores: dict[str, dict[str, float]] = {"static": {}, "scheduled": {}, "triggered": {}}
    trained_at: dict[str, dict[str, int]] = {"static": {}, "scheduled": {}, "triggered": {}}
    decisions = []

    current = origins[0]
    reference_days = np.arange(current - HORIZON, current)
    reference = monitored_features(m5, encodings, reference_days, current, sample)
    reference_error: float | None = None
    last_error: float | None = None

    for index, origin in enumerate(origins):
        log(f"window starting at day {origin}")
        scores["static"][str(origin)] = score(origin, origins[0])
        trained_at["static"][str(origin)] = origins[0]
        scores["scheduled"][str(origin)] = score(origin, origin)
        trained_at["scheduled"][str(origin)] = origin

        if index > 0:
            # Decide using inputs for the coming window, and the error on the window that has just finished.
            window_days = np.arange(origin, origin + HORIZON)
            drift = feature_psi(reference, monitored_features(m5, encodings, window_days, origin, sample), MONITORED)
            decision = policy.decide(reference_error, last_error, drift)
            decisions.append({"origin": origin, "psi": {k: round(v, 4) for k, v in drift.items()}, **decision})
            log(f"  decision: retrain={decision['retrain']} {decision['reasons']}")
            if decision["retrain"]:
                current = origin
                reference = monitored_features(m5, encodings, np.arange(origin - HORIZON, origin), origin, sample)
                reference_error = None

        error = score(origin, current)
        scores["triggered"][str(origin)] = error
        trained_at["triggered"][str(origin)] = current
        last_error = error
        if reference_error is None:
            reference_error = error  # the first full window after a retrain sets the bar for later windows

    def summary(name: str) -> dict:
        retrains = len(set(trained_at[name].values()))
        values = list(scores[name].values())
        return {
            "mean_wrmsse": float(np.mean(values)),
            "worst_window": float(np.max(values)),
            "fits": retrains,
            "fit_seconds": float(sum(cache.fit_seconds[o] for o in set(trained_at[name].values()))),
            "by_window": scores[name],
            "trained_at": trained_at[name],
        }

    return {
        "origins": origins,
        "config": asdict(cfg),
        "policy": asdict(policy),
        "monitored_features": MONITORED,
        "strategies": {name: summary(name) for name in scores},
        "decisions": decisions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results")
    parser.add_argument("--windows", type=int, default=12)
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    def log(message: str) -> None:
        print(message, flush=True)

    m5 = load(args.data)
    encodings = Encodings.from_data(m5)
    last = N_OBSERVED_DAYS - HORIZON
    origins = [last - HORIZON * k for k in range(args.windows - 1, -1, -1)]
    cfg = ModelConfig(train_days=args.train_days, trees=args.trees, threads=args.threads)
    report = replay(m5, encodings, origins, cfg, RetrainPolicy(), log)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "replay.json").write_text(json.dumps(report, indent=2))
    log(
        json.dumps(
            {k: {m: v[m] for m in ("mean_wrmsse", "worst_window", "fits", "fit_seconds")} for k, v in report["strategies"].items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
