"""A global LightGBM model across all series, with a Tweedie objective for zero-heavy counts."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import lightgbm as lgb
import numpy as np

from .data import HORIZON, M5
from .features import CATEGORICAL, FEATURE_NAMES, HISTORY_NEEDED, Encodings, build_features, on_sale_mask


@dataclass
class ModelConfig:
    train_days: int = 730
    trees: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 127
    min_data_in_leaf: int = 100
    feature_fraction: float = 0.5
    bagging_fraction: float = 0.3
    max_bin: int = 127
    tweedie_power: float = 1.1
    threads: int = 4
    seed: int = 0
    excluded_features: tuple[str, ...] = ()


def fit(m5: M5, encodings: Encodings, origin: int, cfg: ModelConfig) -> tuple[lgb.Booster, dict]:
    """Train on target days before ``origin``. Features for those days read no later than ``day - 28``."""
    start_day = max(origin - cfg.train_days, HISTORY_NEEDED)
    days = np.arange(start_day, origin)
    started = time.perf_counter()
    mask = on_sale_mask(m5, days, origin)
    features = build_features(m5, encodings, days, origin)[mask]
    labels = m5.sales[:, days].ravel()[mask].astype(np.float32)
    feature_seconds = time.perf_counter() - started

    used = [name for name in FEATURE_NAMES if name not in cfg.excluded_features]
    if cfg.excluded_features:
        keep = [FEATURE_NAMES.index(name) for name in used]
        features = features[:, keep]
    dataset = lgb.Dataset(
        features,
        labels,
        feature_name=used,
        categorical_feature=[c for c in CATEGORICAL if c in used],
        free_raw_data=True,
        params={"max_bin": cfg.max_bin},
    )
    params = {
        "objective": "tweedie",
        "tweedie_variance_power": cfg.tweedie_power,
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "min_data_in_leaf": cfg.min_data_in_leaf,
        "feature_fraction": cfg.feature_fraction,
        "bagging_fraction": cfg.bagging_fraction,
        "bagging_freq": 1,
        "max_bin": cfg.max_bin,
        "num_threads": cfg.threads,
        "seed": cfg.seed,
        "verbosity": -1,
    }
    started = time.perf_counter()
    booster = lgb.train(params, dataset, num_boost_round=cfg.trees)
    info = {
        "config": asdict(cfg),
        "origin": origin,
        "train_rows": int(mask.sum()),
        "feature_seconds": feature_seconds,
        "fit_seconds": time.perf_counter() - started,
    }
    return booster, info


def predict(
    booster: lgb.Booster, m5: M5, encodings: Encodings, origin: int, series: np.ndarray | None = None, horizon: int = HORIZON
) -> np.ndarray:
    """Forecast of shape (series, horizon). A row that cannot sell (no price, not yet launched) is exactly 0."""
    days = np.arange(origin, origin + horizon)
    picked_count = m5.n_series if series is None else len(series)
    features = build_features(m5, encodings, days, origin, series)
    if booster.feature_name() != FEATURE_NAMES:
        features = features[:, [FEATURE_NAMES.index(name) for name in booster.feature_name()]]
    forecast = booster.predict(features).reshape(picked_count, horizon)
    return np.where(on_sale_mask(m5, days, origin, series).reshape(picked_count, horizon), forecast, 0.0)
