"""Rolling-origin backtest and the checks that come before trusting its numbers.

The metric is checked first, against two published reference scores. Model choices are made on the earlier
folds. The last fold, M5's official evaluation window, is only scored for the final configuration.

    demandcast-backtest --out results
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import data as dataio
from .baselines import BASELINES
from .data import HORIZON, N_OBSERVED_DAYS, load, load_winner_forecast
from .features import Encodings
from .intervals import LEVELS, ConformalIntervals, coverage_report, poisson_interval
from .metrics import WRMSSE
from .model import ModelConfig, fit, predict

# Origins in days, oldest first. The final one is M5's private leaderboard window (d_1942 to d_1969).
FINAL_ORIGIN = N_OBSERVED_DAYS - HORIZON
TUNING_ORIGINS = [FINAL_ORIGIN - HORIZON * k for k in (3, 2, 1)]

ABLATIONS = {
    "no_price": ("price", "price_relative", "price_change_28"),
    "no_calendar": ("dow", "dom", "month", "week", "event_name", "event_type", "snap"),
}


def scorer_at(m5: dataio.M5, origin: int) -> WRMSSE:
    return WRMSSE(m5.ids, m5.sales[:, :origin], m5.prices[:, :origin])


def check_metric(m5: dataio.M5, data_dir: str, log) -> dict:
    """Reproduce two published scores on the official window, or nothing downstream can be trusted."""
    origin = FINAL_ORIGIN
    actual = m5.sales[:, origin : origin + HORIZON].astype(np.float64)
    scorer = scorer_at(m5, origin)
    result = {"n_series": scorer.n_series}
    result["seasonal_naive"] = scorer.score(BASELINES["seasonal_naive"](m5.sales[:, :origin]), actual)
    winner = load_winner_forecast(data_dir, m5.ids)
    result["winner_yj_stu"] = scorer.score(winner, actual)
    log(f"metric check: {result}")
    return result


def evaluate_baselines(m5: dataio.M5, origins: list[int]) -> dict:
    results: dict[str, dict] = {name: {} for name in BASELINES}
    for origin in origins:
        scorer = scorer_at(m5, origin)
        actual = m5.sales[:, origin : origin + HORIZON].astype(np.float64)
        for name, fn in BASELINES.items():
            results[name][str(origin)] = scorer.score(fn(m5.sales[:, :origin]), actual)
    return results


def run_model(m5, encodings, origin: int, cfg: ModelConfig, log) -> tuple[np.ndarray, dict]:
    booster, info = fit(m5, encodings, origin, cfg)
    start = time.perf_counter()
    forecast = predict(booster, m5, encodings, origin)
    info["predict_seconds"] = time.perf_counter() - start
    actual = m5.sales[:, origin : origin + HORIZON].astype(np.float64)
    scorer = scorer_at(m5, origin)
    info["wrmsse"] = scorer.score(forecast, actual)
    info["per_level"] = scorer.per_level(forecast, actual)
    info["importance_gain"] = dict(zip(booster.feature_name(), booster.feature_importance("gain").round(0).tolist()))
    log(f"  origin {origin}: WRMSSE {info['wrmsse']:.4f} rows {info['train_rows']:,} fit {info['fit_seconds']:.0f}s")
    return forecast, info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--skip-ablation", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        print(message, flush=True)

    m5 = load(args.data)
    encodings = Encodings.from_data(m5)
    summary: dict = {"metric_check": check_metric(m5, args.data, log)}

    def save() -> None:
        (out / "backtest.json").write_text(json.dumps(summary, indent=2))

    save()
    summary["baselines"] = evaluate_baselines(m5, [*TUNING_ORIGINS, FINAL_ORIGIN])
    save()

    # Choose on the tuning folds only.
    base = ModelConfig(threads=args.threads)
    candidates = {
        "two_years": base,
        "one_year": replace(base, train_days=365),
    }
    summary["tuning"] = {}
    best_name, best_score = None, np.inf
    for name, cfg in candidates.items():
        log(f"candidate {name}")
        folds = {}
        for origin in TUNING_ORIGINS:
            _, info = run_model(m5, encodings, origin, cfg, log)
            folds[str(origin)] = {k: info[k] for k in ("wrmsse", "train_rows", "fit_seconds")}
        mean = float(np.mean([fold["wrmsse"] for fold in folds.values()]))
        summary["tuning"][name] = {"folds": folds, "mean_wrmsse": mean}
        log(f"  {name}: mean WRMSSE {mean:.4f}")
        if mean < best_score:
            best_name, best_score = name, mean
        save()
    summary["selected"] = best_name
    selected = candidates[best_name]

    if not args.skip_ablation:
        # One window each: this asks which feature groups matter, not for a precise score.
        origin = TUNING_ORIGINS[-1]
        summary["ablation"] = {"origin": origin, "all_features": summary["tuning"][best_name]["folds"][str(origin)]["wrmsse"]}
        for name, excluded in ABLATIONS.items():
            log(f"ablation {name}")
            _, info = run_model(m5, encodings, origin, replace(selected, excluded_features=excluded), log)
            summary["ablation"][name] = info["wrmsse"]
            save()

    # Calibrate intervals on the fold just before the one they're tested on.
    log("final configuration")
    calibration_origin = FINAL_ORIGIN - HORIZON
    calibration_forecast, calibration_info = run_model(m5, encodings, calibration_origin, selected, log)
    forecast, info = run_model(m5, encodings, FINAL_ORIGIN, selected, log)
    summary["final"] = info
    summary["final_calibration_fold"] = calibration_info["wrmsse"]
    actual = m5.sales[:, FINAL_ORIGIN : FINAL_ORIGIN + HORIZON].astype(np.float64)
    calibration_actual = m5.sales[:, calibration_origin : calibration_origin + HORIZON].astype(np.float64)

    # Rows that can't sell are forced to zero, and including them would inflate coverage.
    can_sell = ~np.isnan(m5.prices[:, FINAL_ORIGIN : FINAL_ORIGIN + HORIZON])
    can_sell_calibration = ~np.isnan(m5.prices[:, calibration_origin : calibration_origin + HORIZON])
    intervals = ConformalIntervals.calibrate(calibration_forecast[can_sell_calibration], calibration_actual[can_sell_calibration])
    summary["intervals"] = {}
    for level in LEVELS:
        low, high = intervals.predict(forecast[can_sell], level)
        poisson_low, poisson_high = poisson_interval(forecast[can_sell], level)
        summary["intervals"][str(level)] = {
            "conformal": coverage_report(actual[can_sell], low, high, forecast[can_sell]),
            "poisson": coverage_report(actual[can_sell], poisson_low, poisson_high, forecast[can_sell]),
        }
    save()
    log("done")


if __name__ == "__main__":
    main()
