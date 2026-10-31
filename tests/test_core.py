import numpy as np
import pandas as pd
import pytest

from demandcast.baselines import moving_average, naive, seasonal_naive, weekday_average
from demandcast.features import FEATURE_NAMES, HISTORY_NEEDED, Encodings, build_features, first_sale_day, on_sale_mask
from demandcast.intervals import ConformalIntervals, coverage_report, poisson_interval
from demandcast.metrics import WRMSSE, _naive_scale
from demandcast.model import ModelConfig, fit, predict
from demandcast.monitor import RetrainPolicy, feature_psi, psi


def test_features_do_not_change_when_the_future_changes(tiny_m5):
    """The central leakage test: rewriting everything at or after the origin must not move any feature."""
    enc = Encodings.from_data(tiny_m5)
    origin = 300
    days = np.arange(origin, origin + 28)
    before = build_features(tiny_m5, enc, days, origin)
    sales = tiny_m5.sales.copy()
    tiny_m5.sales[:, origin:] = 9999
    try:
        after = build_features(tiny_m5, enc, days, origin)
    finally:
        tiny_m5.sales[:] = sales
    assert np.array_equal(before, after, equal_nan=True)


def test_features_read_no_sale_after_day_minus_28(tiny_m5):
    enc = Encodings.from_data(tiny_m5)
    origin, day = 300, np.array([290])
    before = build_features(tiny_m5, enc, day, origin)
    sales = tiny_m5.sales.copy()
    tiny_m5.sales[:, 290 - 27 :] = 9999  # everything newer than lag 28 of day 290
    try:
        after = build_features(tiny_m5, enc, day, origin)
    finally:
        tiny_m5.sales[:] = sales
    assert np.array_equal(before, after, equal_nan=True)


def test_lag_and_rolling_values_are_correct(tiny_m5):
    enc = Encodings.from_data(tiny_m5)
    day = 200
    row = build_features(tiny_m5, enc, np.array([day]), 300, np.array([0]))[0]
    sales = tiny_m5.sales[0]
    assert row[FEATURE_NAMES.index("lag_28")] == sales[day - 28]
    assert row[FEATURE_NAMES.index("lag_56")] == sales[day - 56]
    assert row[FEATURE_NAMES.index("rmean_7")] == pytest.approx(sales[day - 34 : day - 27].mean())
    assert row[FEATURE_NAMES.index("rmean_28")] == pytest.approx(sales[day - 55 : day - 27].mean())
    assert row[FEATURE_NAMES.index("rstd_28")] == pytest.approx(sales[day - 55 : day - 27].std(), rel=1e-5)


def test_features_reject_a_day_that_would_read_the_future(tiny_m5):
    enc = Encodings.from_data(tiny_m5)
    with pytest.raises(ValueError, match="past origin"):
        build_features(tiny_m5, enc, np.array([328]), 300)
    with pytest.raises(ValueError, match="history"):
        build_features(tiny_m5, enc, np.array([HISTORY_NEEDED - 1]), 300)


def test_feature_rows_are_ordered_by_series_then_day(tiny_m5):
    enc = Encodings.from_data(tiny_m5)
    days = np.array([200, 201])
    matrix = build_features(tiny_m5, enc, days, 300)
    assert matrix.shape == (tiny_m5.n_series * 2, len(FEATURE_NAMES))
    lag = FEATURE_NAMES.index("lag_28")
    assert matrix[3, lag] == tiny_m5.sales[1, 201 - 28]


def test_first_sale_day_handles_never_sold():
    sales = np.array([[0, 0, 3], [1, 0, 0], [0, 0, 0]])
    assert first_sale_day(sales).tolist() == [2, 0, 3]


def test_on_sale_mask_excludes_unlaunched_rows(tiny_m5):
    days = np.array([90, 200])
    mask = on_sale_mask(tiny_m5, days, 300).reshape(tiny_m5.n_series, 2)
    assert mask[0].all()
    assert mask[-1].tolist() == [False, True]


def test_wrmsse_is_zero_for_a_perfect_forecast(tiny_m5):
    origin = 372
    scorer = WRMSSE(tiny_m5.ids, tiny_m5.sales[:, :origin], tiny_m5.prices[:, :origin])
    actual = tiny_m5.sales[:, origin : origin + 28].astype(float)
    assert scorer.score(actual, actual) == 0.0
    assert scorer.score(actual + 1, actual) > 0


def test_wrmsse_weights_sum_to_one_across_levels(tiny_m5):
    scorer = WRMSSE(tiny_m5.ids, tiny_m5.sales[:, :372], tiny_m5.prices[:, :372])
    assert sum(w.sum() for w in scorer.weights) == pytest.approx(1.0)
    assert len(scorer.levels) == 12


def test_wrmsse_matches_a_hand_computed_case():
    # A single series: every level reduces to it, so WRMSSE equals its own RMSSE.
    ids = pd.DataFrame([{"id": "a", "item_id": "i", "dept_id": "d", "cat_id": "c", "store_id": "s", "state_id": "t"}])
    history = np.tile([0, 2], 20)[None, :].astype(float)  # one-step differences are always +-2, so the scale is 4
    prices = np.ones_like(history)
    scorer = WRMSSE(ids, history, prices)
    forecast = np.full((1, 28), 3.0)
    actual = np.full((1, 28), 1.0)
    # RMSSE = sqrt(mean((3 - 1)^2) / 4) = 1
    assert scorer.score(forecast, actual) == pytest.approx(1.0)


def test_scale_starts_at_the_first_nonzero_sale():
    # Leading zeros are the time before launch. They must not dilute the scale.
    series = np.array([[0.0, 0.0, 0.0, 1.0, 3.0, 1.0]])
    assert _naive_scale(series)[0] == pytest.approx(4.0)


def test_aggregate_levels_count_cancelling_errors_as_correct():
    ids = pd.DataFrame(
        [{"id": str(i), "item_id": f"i{i}", "dept_id": "d", "cat_id": "c", "store_id": "s", "state_id": "t"} for i in range(2)]
    )
    history = np.tile([1, 3], 20).reshape(1, -1).repeat(2, axis=0).astype(float)
    scorer = WRMSSE(ids, history, np.ones_like(history))
    actual = np.ones((2, 28))
    forecast = actual + np.array([[1.0], [-1.0]])
    per_level = scorer.per_level(forecast, actual)
    assert per_level[0] == pytest.approx(0.0)
    assert per_level[-1] > 0


def test_baseline_shapes_and_values():
    train = np.arange(1, 29)[None, :]
    assert naive(train)[0].tolist() == [28.0] * 28
    assert seasonal_naive(train)[0, :7].tolist() == list(range(22, 29))
    assert seasonal_naive(train)[0, 7] == 22
    assert moving_average(train)[0, 0] == pytest.approx(14.5)
    assert weekday_average(train).shape == (1, 28)
    assert weekday_average(train)[0, 0] == pytest.approx(np.mean([1, 8, 15, 22]))


def test_model_beats_seasonal_naive_on_seasonal_data(tiny_m5):
    enc = Encodings.from_data(tiny_m5)
    origin = 372
    cfg = ModelConfig(train_days=250, trees=80, num_leaves=7, min_data_in_leaf=5, threads=1, bagging_fraction=1.0, feature_fraction=1.0)
    booster, info = fit(tiny_m5, enc, origin, cfg)
    forecast = predict(booster, tiny_m5, enc, origin)
    actual = tiny_m5.sales[:, origin : origin + 28].astype(float)
    scorer = WRMSSE(tiny_m5.ids, tiny_m5.sales[:, :origin], tiny_m5.prices[:, :origin])
    assert scorer.score(forecast, actual) < scorer.score(seasonal_naive(tiny_m5.sales[:, :origin]), actual)
    assert info["train_rows"] > 0
    assert (forecast >= 0).all()


def test_model_predicts_zero_when_an_item_has_no_price(tiny_m5):
    enc = Encodings.from_data(tiny_m5)
    cfg = ModelConfig(train_days=200, trees=5, num_leaves=7, min_data_in_leaf=5, threads=1)
    booster, _ = fit(tiny_m5, enc, 372, cfg)
    original = tiny_m5.prices.copy()
    tiny_m5.prices[0, 375:] = np.nan
    try:
        forecast = predict(booster, tiny_m5, enc, 372)
    finally:
        tiny_m5.prices[:] = original
    assert (forecast[0, 3:] == 0).all()
    assert forecast[1].sum() > 0


def test_excluded_features_are_not_used(tiny_m5):
    enc = Encodings.from_data(tiny_m5)
    cfg = ModelConfig(train_days=200, trees=5, num_leaves=7, min_data_in_leaf=5, threads=1, excluded_features=("price", "price_relative"))
    booster, _ = fit(tiny_m5, enc, 372, cfg)
    assert "price" not in booster.feature_name()
    assert predict(booster, tiny_m5, enc, 372).shape == (tiny_m5.n_series, 28)


def test_conformal_intervals_reach_nominal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(0)
    forecast = rng.uniform(1, 50, 40000)
    actual = forecast + rng.normal(0, 1 + forecast / 5)
    half = 20000
    intervals = ConformalIntervals.calibrate(forecast[:half], actual[:half])
    for level in (0.5, 0.8, 0.95):
        low, high = intervals.predict(forecast[half:], level)
        assert coverage_report(actual[half:], low, high, forecast[half:])["coverage"] == pytest.approx(level, abs=0.02)


def test_conformal_widths_grow_with_forecast_volume():
    rng = np.random.default_rng(0)
    forecast = rng.uniform(1, 50, 20000)
    actual = forecast + rng.normal(0, forecast / 5)
    intervals = ConformalIntervals.calibrate(forecast, actual)
    low, high = intervals.predict(np.array([2.0, 45.0]), 0.8)
    assert (high - low)[1] > 3 * (high - low)[0]


def test_lower_bound_never_goes_below_zero():
    forecast = np.array([0.1, 0.2, 0.3] * 100)
    actual = forecast + np.random.default_rng(0).normal(0, 5, 300)
    low, _ = ConformalIntervals.calibrate(forecast, actual, bands=2).predict(forecast, 0.95)
    assert (low >= 0).all()


def test_poisson_intervals_undercover_overdispersed_demand():
    rng = np.random.default_rng(0)
    forecast = np.full(50000, 20.0)
    actual = rng.negative_binomial(2, 2 / 22, 50000)  # mean 20, far larger variance than Poisson's
    low, high = poisson_interval(forecast, 0.95)
    assert coverage_report(actual, low, high, forecast)["coverage"] < 0.7


def test_psi_is_near_zero_for_the_same_distribution_and_large_for_a_shift():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=50000)
    assert psi(reference, rng.normal(size=50000)) < 0.01
    assert psi(reference, rng.normal(loc=1.0, size=50000)) > 0.5


def test_psi_notices_new_missing_values():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=10000)
    current = rng.normal(size=10000)
    current[:3000] = np.nan
    assert psi(reference, current) > 0.2


def test_feature_psi_names_each_column():
    x = np.random.default_rng(0).normal(size=(1000, 2))
    assert set(feature_psi(x, x, ["a", "b"])) == {"a", "b"}


@pytest.mark.parametrize(
    ("reference", "latest", "drift", "expected"),
    [
        (0.60, 0.61, {}, False),
        (0.60, 0.70, {}, True),
        (0.60, 0.61, {"a": 0.5, "b": 0.5}, False),
        (0.60, 0.61, {"a": 0.5, "b": 0.5, "c": 0.5}, True),
        (None, None, {}, False),
    ],
)
def test_retrain_policy(reference, latest, drift, expected):
    assert RetrainPolicy().decide(reference, latest, drift)["retrain"] is expected
