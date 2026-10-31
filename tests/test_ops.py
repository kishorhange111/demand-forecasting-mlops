"""The operations around the model: the promotion gate, the replay and the service."""

from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from demandcast.api import ServiceState, create_app, load_intervals
from demandcast.data import HORIZON
from demandcast.features import Encodings
from demandcast.intervals import ConformalIntervals
from demandcast.model import ModelConfig, fit
from demandcast.monitor import RetrainPolicy
from demandcast.registry import GatePolicy, evaluate_gate, load_champion, register_if_better
from demandcast.replay import replay
from demandcast.train import run_training

SMALL = ModelConfig(train_days=200, trees=10, num_leaves=7, min_data_in_leaf=5, threads=1, bagging_fraction=1.0, feature_fraction=1.0)


def test_gate_promotes_the_first_model():
    assert evaluate_gate({"1": 0.6}, None)["promote"]


def test_gate_promotes_a_clear_win():
    assert evaluate_gate({"1": 0.58, "2": 0.59}, {"1": 0.60, "2": 0.60})["promote"]


def test_gate_rejects_a_change_within_noise():
    decision = evaluate_gate({"1": 0.5999}, {"1": 0.60})
    assert not decision["promote"]
    assert "needs" in decision["reasons"][0]


def test_gate_rejects_a_better_average_that_hides_a_bad_window():
    decision = evaluate_gate({"1": 0.40, "2": 0.70}, {"1": 0.60, "2": 0.60})
    assert decision["mean_challenger"] < decision["mean_champion"]
    assert not decision["promote"]
    assert "window 2" in decision["reasons"][0]


def test_gate_rejects_scores_from_different_windows():
    assert not evaluate_gate({"1": 0.1}, {"2": 0.9})["promote"]


def test_gate_thresholds_are_configurable():
    assert evaluate_gate({"1": 0.5999}, {"1": 0.60}, GatePolicy(min_improvement=0.0))["promote"]


@pytest.fixture
def tracking(tmp_path):
    return {"tracking_uri": f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}", "artifact_root": tmp_path / "mlruns"}


def test_registry_moves_the_alias_only_when_the_gate_passes(tiny_m5, tracking):
    encodings = Encodings.from_data(tiny_m5)
    booster, _ = fit(tiny_m5, encodings, 372, SMALL)
    first = register_if_better(booster, {"372": 0.60}, {"trees": 10}, **tracking)
    assert first["promote"] and first["version"] == "1"
    worse = register_if_better(booster, {"372": 0.70}, {"trees": 10}, **tracking)
    assert not worse["promote"]
    assert worse["previous_champion"] == "1"
    _, serving = load_champion(tracking_uri=tracking["tracking_uri"])
    assert serving == "1"
    better = register_if_better(booster, {"372": 0.50}, {"trees": 10}, **tracking)
    assert better["promote"]
    _, serving = load_champion(tracking_uri=tracking["tracking_uri"])
    assert serving == "3"


def test_training_job_only_writes_artifacts_when_promoted(tiny_m5, tracking, tmp_path):
    artifacts = tmp_path / "artifacts"
    arguments = {
        "m5": tiny_m5,
        "artifacts": artifacts,
        "tracking_uri": tracking["tracking_uri"],
        "artifact_root": str(tracking["artifact_root"]),
        "origins": [344, 372],
        "log": lambda _: None,
    }
    assert run_training(cfg=SMALL, policy=GatePolicy(), **arguments)["promote"]
    assert (artifacts / "model.txt").exists()
    before = (artifacts / "manifest.json").read_text()
    # An identical retrain can't clear the improvement bar, so the service keeps the same model.
    assert not run_training(cfg=SMALL, policy=GatePolicy(), **arguments)["promote"]
    assert (artifacts / "manifest.json").read_text() == before


def test_saved_intervals_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    forecast = rng.uniform(0, 10, 2000)
    intervals = ConformalIntervals.calibrate(forecast, forecast + rng.normal(size=2000))
    import json

    path = tmp_path / "i.json"
    path.write_text(
        json.dumps(
            {"edges": intervals.edges.tolist(), "offsets": {str(k): [v[0].tolist(), v[1].tolist()] for k, v in intervals.offsets.items()}}
        )
    )
    loaded = load_intervals(path)
    low_a, high_a = intervals.predict(forecast[:5], 0.8)
    low_b, high_b = loaded.predict(forecast[:5], 0.8)
    assert np.allclose(low_a, low_b) and np.allclose(high_a, high_b)


def test_replay_scores_every_strategy_on_every_window(tiny_m5):
    encodings = Encodings.from_data(tiny_m5)
    origins = [316, 344, 372]
    report = replay(tiny_m5, encodings, origins, replace(SMALL, train_days=150), RetrainPolicy(), lambda _: None)
    assert report["origins"] == origins
    assert report["strategies"]["static"]["fits"] == 1
    assert report["strategies"]["scheduled"]["fits"] == 3
    assert 1 <= report["strategies"]["triggered"]["fits"] <= 3
    assert len(report["decisions"]) == 2
    for strategy in report["strategies"].values():
        assert set(strategy["by_window"]) == {"316", "344", "372"}


def test_replay_never_uses_a_model_from_the_future(tiny_m5):
    encodings = Encodings.from_data(tiny_m5)
    report = replay(tiny_m5, encodings, [316, 344, 372], replace(SMALL, train_days=150), RetrainPolicy(), lambda _: None)
    for strategy in report["strategies"].values():
        for window, trained_at in strategy["trained_at"].items():
            assert trained_at <= int(window)


@pytest.fixture
def client(tiny_m5):
    encodings = Encodings.from_data(tiny_m5)
    origin = tiny_m5.sales.shape[1]
    booster, _ = fit(tiny_m5, encodings, origin, SMALL)
    rng = np.random.default_rng(0)
    forecast = rng.uniform(0, 10, 2000)
    state = ServiceState(
        m5=tiny_m5,
        encodings=encodings,
        booster=booster,
        intervals=ConformalIntervals.calibrate(forecast, forecast + rng.normal(size=2000)),
        version="7",
        origin=origin,
        series_index={(i, s): n for n, (i, s) in enumerate(zip(tiny_m5.ids["item_id"], tiny_m5.ids["store_id"]))},
    )
    with TestClient(create_app(state)) as test_client:
        yield test_client


def test_forecast_returns_ordered_days_with_valid_intervals(client, tiny_m5):
    item, store = tiny_m5.ids.iloc[0][["item_id", "store_id"]]
    body = client.post("/forecast", json={"item_id": item, "store_id": store, "horizon": HORIZON}).json()
    assert body["model_version"] == "7"
    assert len(body["points"]) == HORIZON
    assert [p["day_index"] for p in body["points"]] == list(range(400, 428))
    for point in body["points"]:
        assert 0 <= point["lower"] <= point["upper"]
        assert point["date"] is not None


def test_forecast_rejects_bad_requests(client, tiny_m5):
    item, store = tiny_m5.ids.iloc[0][["item_id", "store_id"]]
    assert client.post("/forecast", json={"item_id": "nope", "store_id": store}).status_code == 404
    assert client.post("/forecast", json={"item_id": item, "store_id": store, "horizon": 29}).status_code == 422
    assert client.post("/forecast", json={"item_id": item, "store_id": store, "horizon": 0}).status_code == 422
    assert client.post("/forecast", json={"item_id": item, "store_id": store, "interval": 0.33}).status_code == 422


def test_health_and_metrics(client, tiny_m5):
    assert client.get("/health").json()["model_version"] == "7"
    item, store = tiny_m5.ids.iloc[0][["item_id", "store_id"]]
    client.post("/forecast", json={"item_id": item, "store_id": store, "horizon": 3})
    text = client.get("/metrics").text
    assert 'forecast_requests_total{path="/forecast",status="200"}' in text
    assert "forecast_points_total 3.0" in text
    assert 'forecast_model_info{origin="400",version="7"} 1.0' in text


def test_unknown_paths_do_not_grow_the_metric_labels(client):
    client.get("/random-scanner-path")
    assert "random-scanner-path" not in client.get("/metrics").text
