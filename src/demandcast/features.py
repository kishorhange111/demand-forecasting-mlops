"""Features for a 28-day-ahead forecast, built so they cannot see the future.

Every sales feature for a target day ``d`` reads days up to ``d - 28`` at the latest. That lets one model serve
all 28 horizons without recursion, since a lag-28 value is observed for every day in the forecast window, and
errors don't compound as they do when a model feeds on its own predictions. Prices and the calendar are
treated as known in advance, as M5 specifies: retailers plan promotions ahead of time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import M5

MIN_LAG = 28
LAGS = (28, 35, 42, 49, 56)
ROLLING_WINDOWS = (7, 14, 28, 56)
HISTORY_NEEDED = MIN_LAG + max(max(LAGS), max(ROLLING_WINDOWS))

ID_FEATURES = ["item", "dept", "cat", "store", "state"]
CALENDAR_FEATURES = ["dow", "dom", "month", "week", "event_name", "event_type", "snap"]
CATEGORICAL = [*ID_FEATURES, "dow", "month", "event_name", "event_type", "snap"]
FEATURE_NAMES = [
    *ID_FEATURES,
    *[f"lag_{lag}" for lag in LAGS],
    *[f"rmean_{w}" for w in ROLLING_WINDOWS],
    "rstd_28",
    "nonzero_28",
    "age",
    "price",
    "price_relative",
    "price_change_28",
    *CALENDAR_FEATURES,
]


@dataclass
class Encodings:
    """Integer codes shared across every fold, so a category means the same thing in each model."""

    codes: dict[str, np.ndarray]
    calendar: pd.DataFrame

    @classmethod
    def from_data(cls, m5: M5) -> Encodings:
        ids = m5.ids
        codes = {
            "item": pd.factorize(ids["item_id"], sort=True)[0],
            "dept": pd.factorize(ids["dept_id"], sort=True)[0],
            "cat": pd.factorize(ids["cat_id"], sort=True)[0],
            "store": pd.factorize(ids["store_id"], sort=True)[0],
            "state": pd.factorize(ids["state_id"], sort=True)[0],
        }
        cal = m5.calendar
        date = pd.to_datetime(cal["date"])
        frame = pd.DataFrame(
            {
                "dow": date.dt.dayofweek.to_numpy(),
                "dom": date.dt.day.to_numpy(),
                "month": date.dt.month.to_numpy(),
                "week": date.dt.isocalendar().week.to_numpy().astype(int),
                "event_name": pd.factorize(cal["event_name_1"], sort=True)[0] + 1,
                "event_type": pd.factorize(cal["event_type_1"], sort=True)[0] + 1,
            }
        )
        for state in ("CA", "TX", "WI"):
            frame[f"snap_{state}"] = cal[f"snap_{state}"].to_numpy()
        return cls(codes, frame)


def first_sale_day(sales: np.ndarray) -> np.ndarray:
    """Index of each series' first non-zero day, or the array's length if it has never sold."""
    active = sales > 0
    first = active.argmax(axis=1)
    first[~active.any(axis=1)] = sales.shape[1]
    return first


def build_features(m5: M5, encodings: Encodings, days: np.ndarray, origin: int, series: np.ndarray | None = None) -> np.ndarray:
    """Feature matrix with one row per (series, day), ordered by series and then day.

    Only ``sales[:, :origin]`` is read. Any day whose lag-28 would cross the origin is a bug in the caller, so
    it raises rather than silently using the future.
    """
    days = np.asarray(days)
    if days.max() - MIN_LAG >= origin:
        raise ValueError(f"day {days.max()} needs sales from day {days.max() - MIN_LAG}, past origin {origin}")
    if days.min() < HISTORY_NEEDED:
        raise ValueError(f"day {days.min()} has fewer than {HISTORY_NEEDED} days of history")

    picked = np.arange(m5.n_series) if series is None else np.asarray(series)
    sales = m5.sales[picked, :origin].astype(np.float64)
    prices = m5.prices[picked]
    n_series, n_days = len(picked), len(days)

    # Prefix sums make every rolling window O(1). Index i holds the sum of the first i days.
    zero = np.zeros((n_series, 1))
    cum = np.concatenate([zero, sales.cumsum(axis=1)], axis=1)
    cum_sq = np.concatenate([zero, (sales**2).cumsum(axis=1)], axis=1)
    cum_nonzero = np.concatenate([zero, (sales > 0).cumsum(axis=1)], axis=1)
    end = days - MIN_LAG + 1  # one past the last usable day, for each target day

    def rolling(prefix: np.ndarray, window: int) -> np.ndarray:
        return (prefix[:, end] - prefix[:, end - window]) / window

    columns: list[np.ndarray] = []

    def repeat_static(values: np.ndarray) -> np.ndarray:
        return np.repeat(values, n_days)

    for name in ID_FEATURES:
        columns.append(repeat_static(encodings.codes[name][picked]))
    for lag in LAGS:
        columns.append(sales[:, days - lag].ravel())
    for window in ROLLING_WINDOWS:
        columns.append(rolling(cum, window).ravel())
    mean_28 = rolling(cum, 28)
    columns.append(np.sqrt(np.maximum(rolling(cum_sq, 28) - mean_28**2, 0.0)).ravel())
    columns.append(rolling(cum_nonzero, 28).ravel())

    first_sale = first_sale_day(sales)
    columns.append((days[None, :] - first_sale[:, None]).ravel())

    price = prices[:, days]
    known = prices[:, :origin]
    counts = (~np.isnan(known)).sum(axis=1)
    usual_price = np.where(counts > 0, np.nansum(known, axis=1) / np.maximum(counts, 1), np.nan)
    columns.append(price.ravel())
    columns.append((price / usual_price[:, None]).ravel())
    columns.append((price / prices[:, days - MIN_LAG]).ravel())

    calendar = encodings.calendar
    for name in ("dow", "dom", "month", "week", "event_name", "event_type"):
        columns.append(np.tile(calendar[name].to_numpy()[days], n_series))
    state_names = m5.ids["state_id"].to_numpy()[picked]
    snap = np.empty((n_series, n_days), dtype=np.float64)
    for state in ("CA", "TX", "WI"):
        snap[state_names == state] = calendar[f"snap_{state}"].to_numpy()[days]
    columns.append(snap.ravel())

    return np.column_stack(columns).astype(np.float32)


def on_sale_mask(m5: M5, days: np.ndarray, origin: int, series: np.ndarray | None = None) -> np.ndarray:
    """True where a row is forecastable: the item has started selling and has a price on that day."""
    picked = np.arange(m5.n_series) if series is None else np.asarray(series)
    first_sale = first_sale_day(m5.sales[picked, :origin])
    started = days[None, :] >= first_sale[:, None]
    priced = ~np.isnan(m5.prices[picked][:, days])
    return (started & priced).ravel()
