"""The production training job: backtest a candidate, put it through the gate, and package what the service needs.

    demandcast-train --artifacts artifacts --tracking-uri sqlite:///mlflow.db

Only the service's own artifacts are overwritten, and only when the candidate is promoted.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .backtest import scorer_at
from .data import HORIZON, M5, N_OBSERVED_DAYS, load
from .features import Encodings
from .intervals import ConformalIntervals
from .model import ModelConfig, fit, predict
from .registry import GatePolicy, register_if_better


def backtest_scores(
    m5: M5, encodings: Encodings, cfg: ModelConfig, origins: list[int], log=print
) -> tuple[dict[str, float], np.ndarray, int]:
    """Score a configuration on each window. Also returns the last window's forecast, which calibrates the intervals."""
    scores: dict[str, float] = {}
    forecast, origin = None, None
    for origin in origins:
        booster, _ = fit(m5, encodings, origin, cfg)
        forecast = predict(booster, m5, encodings, origin)
        actual = m5.sales[:, origin : origin + HORIZON].astype(np.float64)
        scores[str(origin)] = scorer_at(m5, origin).score(forecast, actual)
        log(f"  origin {origin}: WRMSSE {scores[str(origin)]:.4f}")
    return scores, forecast, origin


def run_training(
    m5: M5, cfg: ModelConfig, artifacts: Path, tracking_uri: str, artifact_root: str, origins: list[int], policy: GatePolicy, log=print
) -> dict:
    encodings = Encodings.from_data(m5)
    scores, forecast, last_origin = backtest_scores(m5, encodings, cfg, origins, log)

    # The final model sees everything observed. Its quality is inferred from the windows above.
    final_origin = m5.sales.shape[1]
    booster, info = fit(m5, encodings, final_origin, cfg)
    decision = register_if_better(booster, scores, asdict(cfg), tracking_uri=tracking_uri, artifact_root=artifact_root, policy=policy)
    log(f"gate: {decision}")

    if decision["promote"]:
        actual = m5.sales[:, last_origin : last_origin + HORIZON].astype(np.float64)
        can_sell = ~np.isnan(m5.prices[:, last_origin : last_origin + HORIZON])
        intervals = ConformalIntervals.calibrate(forecast[can_sell], actual[can_sell])
        artifacts.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(artifacts / "model.txt"))
        (artifacts / "intervals.json").write_text(
            json.dumps(
                {
                    "edges": intervals.edges.tolist(),
                    "offsets": {str(k): [v[0].tolist(), v[1].tolist()] for k, v in intervals.offsets.items()},
                }
            )
        )
        (artifacts / "manifest.json").write_text(
            json.dumps(
                {
                    "version": decision["version"],
                    "origin": final_origin,
                    "scores": scores,
                    "train_rows": info["train_rows"],
                    "config": asdict(cfg),
                },
                indent=2,
            )
        )
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--artifact-root", default="mlruns")
    parser.add_argument("--train-days", type=int, default=730)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--clean", action="store_true", help="delete existing artifacts first")
    args = parser.parse_args()
    if args.clean:
        shutil.rmtree(args.artifacts, ignore_errors=True)
    m5 = load(args.data)
    last = N_OBSERVED_DAYS - HORIZON
    origins = [last - HORIZON * k for k in range(args.windows - 1, -1, -1)]
    cfg = ModelConfig(train_days=args.train_days, trees=args.trees, threads=args.threads)
    run_training(m5, cfg, Path(args.artifacts), args.tracking_uri, args.artifact_root, origins, GatePolicy())


if __name__ == "__main__":
    main()
