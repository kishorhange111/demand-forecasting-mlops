"""Forecast service. Features come from the latest sales snapshot, so a request needs only an item and a store.

DEMAND_ARTIFACTS=artifacts DEMAND_DATA=data uvicorn demandcast.api:app --port 8080
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from .data import HORIZON, M5, load
from .features import Encodings
from .intervals import ConformalIntervals
from .model import predict


class ForecastRequest(BaseModel):
    item_id: str = Field(min_length=1, max_length=64)
    store_id: str = Field(min_length=1, max_length=16)
    horizon: int = Field(default=HORIZON, ge=1, le=HORIZON)
    interval: float = Field(default=0.8)


class ForecastPoint(BaseModel):
    day_index: int
    date: str | None
    forecast: float
    lower: float
    upper: float


class ForecastResponse(BaseModel):
    item_id: str
    store_id: str
    model_version: str
    interval: float
    points: list[ForecastPoint]


@dataclass
class ServiceState:
    m5: M5
    encodings: Encodings
    booster: lgb.Booster
    intervals: ConformalIntervals
    version: str
    origin: int
    series_index: dict[tuple[str, str], int]


def load_intervals(path: Path) -> ConformalIntervals:
    raw = json.loads(path.read_text())
    offsets = {float(level): (np.array(pair[0]), np.array(pair[1])) for level, pair in raw["offsets"].items()}
    return ConformalIntervals(np.array(raw["edges"]), offsets)


def build_state(m5: M5, artifacts: Path) -> ServiceState:
    manifest = json.loads((artifacts / "manifest.json").read_text())
    return ServiceState(
        m5=m5,
        encodings=Encodings.from_data(m5),
        booster=lgb.Booster(model_file=str(artifacts / "model.txt")),
        intervals=load_intervals(artifacts / "intervals.json"),
        version=str(manifest["version"]),
        origin=int(manifest["origin"]),
        series_index={(item, store): i for i, (item, store) in enumerate(zip(m5.ids["item_id"], m5.ids["store_id"]))},
    )


def create_app(state: ServiceState | None = None) -> FastAPI:
    registry = CollectorRegistry()
    requests = Counter("forecast_requests_total", "Requests by path and status", ["path", "status"], registry=registry)
    latency = Histogram(
        "forecast_request_seconds",
        "Request latency",
        ["path"],
        registry=registry,
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
    )
    forecast_mean = Gauge("forecast_mean_units", "Running mean of forecast values, a cheap signal of output drift", registry=registry)
    forecast_count = Counter("forecast_points_total", "Forecast points served", registry=registry)
    model_info = Gauge("forecast_model_info", "Serving model", ["version", "origin"], registry=registry)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state is not None:
            app.state.service = state
        else:
            app.state.service = build_state(
                load(os.environ.get("DEMAND_DATA", "data")), Path(os.environ.get("DEMAND_ARTIFACTS", "artifacts"))
            )
        service = app.state.service
        model_info.labels(service.version, str(service.origin)).set(1)
        yield

    app = FastAPI(title="Demand forecast", version="0.1.0", lifespan=lifespan)
    running = {"sum": 0.0, "count": 0}

    @app.middleware("http")
    async def observe(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        # Known routes only, so a scanner hitting random URLs can't grow the label set without limit.
        path = request.url.path if request.url.path in {"/health", "/forecast", "/metrics"} else "other"
        requests.labels(path, str(response.status_code)).inc()
        latency.labels(path).observe(time.perf_counter() - started)
        return response

    @app.get("/health")
    def health(request: Request) -> dict:
        service = request.app.state.service
        return {"status": "ok", "model_version": service.version, "origin": service.origin, "series": service.m5.n_series}

    @app.post("/forecast", response_model=ForecastResponse)
    def forecast(body: ForecastRequest, request: Request) -> ForecastResponse:
        service: ServiceState = request.app.state.service
        if body.interval not in service.intervals.offsets:
            raise HTTPException(422, f"interval must be one of {sorted(service.intervals.offsets)}")
        index = service.series_index.get((body.item_id, body.store_id))
        if index is None:
            raise HTTPException(404, f"unknown item and store: {body.item_id} at {body.store_id}")
        values = predict(service.booster, service.m5, service.encodings, service.origin, np.array([index]), body.horizon)[0]
        lower, upper = service.intervals.predict(values, body.interval)
        running["sum"] += float(values.sum())
        running["count"] += len(values)
        forecast_count.inc(len(values))
        forecast_mean.set(running["sum"] / running["count"])
        dates = service.m5.calendar["date"].to_numpy()
        points = [
            ForecastPoint(
                day_index=service.origin + h,
                date=str(dates[service.origin + h]) if service.origin + h < len(dates) else None,
                forecast=round(float(values[h]), 4),
                lower=round(float(lower[h]), 4),
                upper=round(float(upper[h]), 4),
            )
            for h in range(body.horizon)
        ]
        return ForecastResponse(
            item_id=body.item_id, store_id=body.store_id, model_version=service.version, interval=body.interval, points=points
        )

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app


# Built at import for uvicorn. Loading waits for start-up, so importing this module stays cheap.
app = create_app()
