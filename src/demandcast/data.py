"""M5 (Walmart, 30,490 item-store series, 1,969 days): download and load into arrays.

Days are 0-based in this code: M5's ``d_1`` is index 0. An ``origin`` is the number of observed days, so a
forecast from origin ``T`` may read ``sales[:, :T]`` and predicts indices ``T ... T + 27``.

M5's private leaderboard scored d_1942 to d_1969. Those sales live in ``sales_test_evaluation.csv``, which many
mirrors leave out, and d_1914 to d_1941 was only the public window. Scoring on the wrong one still produces
plausible numbers, so ``backtest.check_metric`` compares against published scores before anything else runs.

The files are downloaded at run time and never committed: redistribution is not ours to grant.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np
import pandas as pd

DATA_URL = "https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip"
WINNER_URL = "https://github.com/Nixtla/m5-forecasts/raw/main/forecasts/" + quote("0001 YJ_STU.zip")
HORIZON = 28
N_OBSERVED_DAYS = 1969
ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


@dataclass
class M5:
    ids: pd.DataFrame  # one row per series, columns ID_COLUMNS
    sales: np.ndarray  # (n_series, n_observed_days)
    prices: np.ndarray  # (n_series, n_calendar_days), NaN where the item is not on sale
    calendar: pd.DataFrame  # one row per calendar day, including days after the observed range

    @property
    def n_series(self) -> int:
        return self.sales.shape[0]


def download(data_dir: str | Path) -> Path:
    root = Path(data_dir) / "m5"
    root.mkdir(parents=True, exist_ok=True)
    if (root / "sales_test_evaluation.csv").exists():
        return root
    archive_path = root / "m5.zip"
    if not archive_path.exists():
        partial = archive_path.with_suffix(".partial")
        with urlopen(DATA_URL, timeout=300) as response, open(partial, "wb") as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)
        partial.rename(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(root)
    return root


def load(data_dir: str | Path) -> M5:
    """Parse the CSVs once, then reuse a compact cache: this takes parsing from minutes to seconds."""
    root = download(data_dir)
    cache = root / "arrays.npz"
    calendar = pd.read_csv(root / "calendar.csv")
    ids_path = root / "ids.parquet"
    if cache.exists() and ids_path.exists():
        arrays = np.load(cache)
        return M5(pd.read_parquet(ids_path), arrays["sales"], arrays["prices"], calendar)

    keys = ["item_id", "store_id"]
    table = pd.read_csv(root / "sales_train_evaluation.csv")
    held_out = pd.read_csv(root / "sales_test_evaluation.csv")
    table = table.merge(
        held_out.drop(columns=["dept_id", "cat_id", "state_id", "id"], errors="ignore"), on=keys, how="left", validate="one_to_one"
    )
    table["id"] = table["item_id"] + "_" + table["store_id"] + "_evaluation"
    ids = table[ID_COLUMNS].copy()
    day_columns = [f"d_{i}" for i in range(1, N_OBSERVED_DAYS + 1)]
    sales = table[day_columns].to_numpy(dtype=np.int32)
    prices = _daily_prices(pd.read_csv(root / "sell_prices.csv"), ids, calendar)
    np.savez_compressed(cache, sales=sales, prices=prices)
    ids.to_parquet(ids_path)
    return M5(ids, sales, prices, calendar)


def _daily_prices(sell_prices: pd.DataFrame, ids: pd.DataFrame, calendar: pd.DataFrame) -> np.ndarray:
    """Turn the long weekly price table into a (series, day) matrix."""
    weeks = np.sort(calendar["wm_yr_wk"].unique())
    week_index = {week: i for i, week in enumerate(weeks)}
    series_index = {(item, store): i for i, (item, store) in enumerate(zip(ids["item_id"], ids["store_id"]))}
    weekly = np.full((len(ids), len(weeks)), np.nan, dtype=np.float32)
    rows = [series_index[pair] for pair in zip(sell_prices["item_id"], sell_prices["store_id"])]
    columns = sell_prices["wm_yr_wk"].map(week_index).to_numpy()
    weekly[rows, columns] = sell_prices["sell_price"].to_numpy(dtype=np.float32)
    return weekly[:, calendar["wm_yr_wk"].map(week_index).to_numpy()]


def load_winner_forecast(data_dir: str | Path, ids: pd.DataFrame) -> np.ndarray:
    """The M5 Accuracy winner's published evaluation-period forecast, aligned to ``ids`` (series, 28)."""
    root = Path(data_dir) / "m5"
    target = root / "winner.zip"
    if not target.exists():
        with urlopen(WINNER_URL, timeout=300) as response:
            target.write_bytes(response.read())
    with zipfile.ZipFile(target) as archive, archive.open(archive.namelist()[0]) as handle:
        forecast = pd.read_csv(io.BytesIO(handle.read()))
    columns = [f"F{i}" for i in range(1, HORIZON + 1)]
    return forecast.set_index("id").loc[ids["id"], columns].to_numpy(dtype=np.float64)
