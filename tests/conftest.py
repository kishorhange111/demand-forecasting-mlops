import numpy as np
import pandas as pd
import pytest

from demandcast.data import M5


@pytest.fixture(scope="session")
def tiny_m5() -> M5:
    """A small dataset with M5's shape and real structure: weekly seasonality, a price effect, a late launch."""
    rng = np.random.default_rng(0)
    n_days, n_calendar = 400, 428
    rows = []
    for store, state in (("CA_1", "CA"), ("TX_1", "TX")):
        for item in range(4):
            cat = "FOODS" if item < 2 else "HOBBIES"
            item_id = f"{cat}_1_{item:03d}"
            rows.append(
                {
                    "id": f"{item_id}_{store}_evaluation",
                    "item_id": item_id,
                    "dept_id": f"{cat}_1",
                    "cat_id": cat,
                    "store_id": store,
                    "state_id": state,
                }
            )
    ids = pd.DataFrame(rows)
    dow = np.arange(n_calendar) % 7
    base = 1 + np.arange(len(ids))[:, None] * 0.5
    rate = base * (1 + 0.8 * (dow >= 5))
    prices = np.full((len(ids), n_calendar), 2.0, dtype=np.float32)
    prices[:, 300:320] = 1.5
    rate = rate * np.where(prices < 2.0, 1.5, 1.0)
    sales = rng.poisson(rate[:, :n_days]).astype(np.int32)
    # The last series launches on day 100, so it has no sales or price before that.
    sales[-1, :100] = 0
    prices[-1, :100] = np.nan
    dates = pd.date_range("2015-01-01", periods=n_calendar)
    calendar = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "wm_yr_wk": np.arange(n_calendar) // 7,
            "event_name_1": [None] * n_calendar,
            "event_type_1": [None] * n_calendar,
            "snap_CA": (dates.day <= 10).astype(int),
            "snap_TX": (dates.day <= 15).astype(int),
            "snap_WI": (dates.day <= 5).astype(int),
        }
    )
    return M5(ids, sales, prices, calendar)
