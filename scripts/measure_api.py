"""Exercise the service on real M5 and the trained artifacts, then record request latency.

Requests run in process through FastAPI's test client, one at a time, so the numbers cover routing, features, model and
intervals, but not a network hop or concurrent load.

    python scripts/measure_api.py --artifacts artifacts --out results/api.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from demandcast.api import build_state, create_app
from demandcast.data import HORIZON, load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--out", default="results/api.json")
    parser.add_argument("--requests", type=int, default=300)
    args = parser.parse_args()

    started = time.perf_counter()
    m5 = load(args.data)
    state = build_state(m5, Path(args.artifacts))
    load_seconds = time.perf_counter() - started
    rng = np.random.default_rng(0)
    picks = rng.choice(m5.n_series, size=args.requests, replace=False)
    actual = m5.sales[:, state.origin : state.origin + HORIZON]

    latencies = []
    inside = total = 0
    with TestClient(create_app(state)) as client:
        for series in picks:
            item, store = m5.ids.iloc[int(series)][["item_id", "store_id"]]
            begin = time.perf_counter()
            response = client.post("/forecast", json={"item_id": item, "store_id": store, "horizon": HORIZON, "interval": 0.8})
            latencies.append((time.perf_counter() - begin) * 1000)
            if response.status_code != 200:
                raise SystemExit(f"{item} at {store}: {response.status_code} {response.text}")
            for point, sold in zip(response.json()["points"], actual[int(series)]):
                inside += point["lower"] <= sold <= point["upper"]
                total += 1
        health = client.get("/health").json()
        metrics = client.get("/metrics").text

    report = {
        "requests": args.requests,
        "horizon": HORIZON,
        "startup_seconds": round(load_seconds, 1),
        "p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "p99_ms": round(float(np.percentile(latencies, 99)), 2),
        "health": health,
        "coverage_of_80_percent_interval_on_sampled_series": round(inside / total, 4),
        "metrics_exposed": all(name in metrics for name in ("forecast_requests_total", "forecast_request_seconds", "forecast_model_info")),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
