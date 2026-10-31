"""A promotion gate in front of the MLflow model registry.

A new model replaces the champion only if it wins on the same backtest windows. A better average isn't enough: it
must not be materially worse on any single window, because that's how a model that fails badly in a particular
season gets into production. The decision and its reasons are written to the run, so it can be audited later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import mlflow
from mlflow import MlflowClient

CHAMPION = "champion"


@dataclass(frozen=True)
class GatePolicy:
    min_improvement: float = 0.005  # relative improvement in mean WRMSSE, so noise can't trigger a release
    max_fold_regression: float = 0.02  # how much worse any one window may get, relative to the champion


DEFAULT_GATE = GatePolicy()


def evaluate_gate(challenger: dict[str, float], champion: dict[str, float] | None, policy: GatePolicy = DEFAULT_GATE) -> dict:
    """Compare per-window WRMSSE (lower is better). Windows must match, or the comparison means nothing."""
    if champion is None:
        return {"promote": True, "reasons": ["no champion yet"], "mean_challenger": _mean(challenger), "mean_champion": None}
    if set(challenger) != set(champion):
        return {
            "promote": False,
            "reasons": [f"different windows: {sorted(challenger)} vs {sorted(champion)}"],
            "mean_challenger": _mean(challenger),
            "mean_champion": _mean(champion),
        }
    reasons = []
    new, old = _mean(challenger), _mean(champion)
    improvement = (old - new) / old
    if improvement < policy.min_improvement:
        reasons.append(f"mean improved by {improvement:.2%}, needs {policy.min_improvement:.2%}")
    for window in sorted(challenger):
        regression = (challenger[window] - champion[window]) / champion[window]
        if regression > policy.max_fold_regression:
            reasons.append(f"window {window} is {regression:.2%} worse, limit is {policy.max_fold_regression:.2%}")
    return {
        "promote": not reasons,
        "reasons": reasons or [f"mean improved by {improvement:.2%}"],
        "mean_challenger": new,
        "mean_champion": old,
    }


def _mean(scores: dict[str, float]) -> float:
    return sum(scores.values()) / len(scores)


def champion_scores(client: MlflowClient, name: str) -> tuple[dict[str, float] | None, str | None]:
    """The current champion's backtest scores, read back from the run that produced it."""
    try:
        version = client.get_model_version_by_alias(name, CHAMPION)
    except mlflow.exceptions.MlflowException:
        return None, None
    metrics = client.get_run(version.run_id).data.metrics
    scores = {key.removeprefix("wrmsse_origin_"): value for key, value in metrics.items() if key.startswith("wrmsse_origin_")}
    return scores, str(version.version)


def register_if_better(
    booster: lgb.Booster,
    scores: dict[str, float],
    params: dict,
    name: str = "demand-forecast",
    tracking_uri: str = "sqlite:///mlflow.db",
    artifact_root: str | Path = "mlruns",
    policy: GatePolicy = DEFAULT_GATE,
    experiment: str = "demand-forecasting",
) -> dict:
    """Log the run and model, then move the champion alias only if the gate passes."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)
    if client.get_experiment_by_name(experiment) is None:
        client.create_experiment(experiment, artifact_location=Path(artifact_root).resolve().as_uri())
    mlflow.set_experiment(experiment)
    try:
        client.get_registered_model(name)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(name)

    previous, previous_version = champion_scores(client, name)
    decision = evaluate_gate(scores, previous, policy)
    with mlflow.start_run() as run:
        mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        mlflow.log_metrics({f"wrmsse_origin_{window}": value for window, value in scores.items()})
        mlflow.log_metric("wrmsse_mean", _mean(scores))
        mlflow.set_tags({"gate.promote": str(decision["promote"]), "gate.reasons": "; ".join(decision["reasons"])[:500]})
        info = mlflow.lightgbm.log_model(booster, name="model")
    version = client.create_model_version(name, info.model_uri, run_id=run.info.run_id)
    if decision["promote"]:
        client.set_registered_model_alias(name, CHAMPION, version.version)
    return {**decision, "version": str(version.version), "previous_champion": previous_version}


def load_champion(name: str = "demand-forecast", tracking_uri: str = "sqlite:///mlflow.db") -> tuple[lgb.Booster, str]:
    mlflow.set_tracking_uri(tracking_uri)
    version = MlflowClient(tracking_uri).get_model_version_by_alias(name, CHAMPION)
    return mlflow.lightgbm.load_model(f"models:/{name}@{CHAMPION}"), str(version.version)
