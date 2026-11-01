# demand-forecasting-mlops: M5 retail forecasting and the operations around it

A global LightGBM model on M5 (30,490 series), with rolling-origin backtests, conformal intervals, a gated MLflow registry, drift-triggered retraining and a monitored FastAPI service.

**Author:** Kishor Hange

## Results

WRMSSE on M5's private-leaderboard window (d_1942 to d_1969, lower is better), scored once after the configuration was chosen on earlier windows:

| Forecast | WRMSSE |
|---|---|
| M5 winner (YJ_STU), for reference | 0.5204 |
| **This model** (one LightGBM, 300 trees, no ensemble) | **0.5536** |
| Weekday average, 4 weeks | 0.7343 |
| Seasonal naive | 0.8470 |
| Moving average, 28 days | 0.9462 |
| Naive | 1.7520 |

That's 35 % lower error than seasonal naive and 6 % higher than the winner. The final fit took about 10 minutes on a laptop CPU, on 21.8M rows. No leaderboard rank is claimed: the window's sales are public now, so this can't be compared with live competition entries as an equal.

Windows vary a lot, which is why one score isn't enough. The selected model scored 0.660, 0.524 and 0.693 on the three tuning windows (mean 0.626).

**Prediction intervals** (conformal, calibrated on the previous 28-day window, against assuming Poisson demand):

| Nominal | Conformal coverage | Conformal, top-decile volume | Poisson coverage | Poisson, top-decile volume |
|---|---|---|---|---|
| 50 % | 49.7 % | 45.3 % | 75.3 % | 39.1 % |
| 80 % | 79.7 % | 75.1 % | 88.2 % | 59.6 % |
| 95 % | 95.0 % | 92.6 % | 94.8 % | 76.4 % |

Poisson's 95 % overall looks fine, but it covers only 76 % on fast sellers, the items where a stockout costs most. Conformal holds up better there, at the cost of wider intervals (mean width 5.1 vs 3.4 units at 95 %), and still under-covers them a little.

**Feature ablation** (one window, origin 1913): removing calendar features worsened WRMSSE from 0.693 to 0.932. Removing price features *improved* it to 0.674. That's a single window, so it's a question to investigate rather than evidence that price features hurt.

## When to retrain

`demandcast-replay` walks through 12 real 28-day windows (d_1634 to d_1969). At each one a strategy can use only what it would know by then. The rule was fixed before the run and not tuned: retrain if error rises 10 % against a reference window, or if 3 or more features exceed PSI 0.2. This uses a lighter model (one year of history, 200 trees), so its scores aren't comparable with the table above.

| Strategy | Mean WRMSSE | Worst window | Models fitted |
|---|---|---|---|
| Never retrain | 0.729 | 0.986 | 1 |
| **Retrain when triggered** | **0.709** | 0.986 | **4** |
| Retrain every window | 0.694 | 0.986 | 12 |

Triggered retraining got 58 % of the available improvement for 3 extra fits instead of 11, and beat never retraining on 8 windows, lost on 2 and tied on 2. Each fit took 3 to 5 minutes. One fit's logged time spans the laptop sleeping overnight, so `fit_seconds` in `results/replay.json` overstates that strategy's cost.

- **Input drift never fired.** The highest PSI on any feature was 0.064, far below the 0.2 threshold, so all 3 retrains came from rising error. On this data the PSI monitor added nothing, and error tracking did the work. That depends on actual sales arriving quickly, which real replenishment data may not do.
- **Retraining doesn't fix everything.** The worst window scored about 0.986 under all three strategies, so whatever happened there wasn't model staleness.
- **12 windows is a small sample**, and no significance test was run. Treat 58 % as an indication, not a precise figure.

## Operations on real data

- **Promotion gate:** the production model (mean WRMSSE 0.626 on 3 windows) became version 1 in the MLflow registry. A deliberately weak challenger (40 trees, mean 0.903) was registered as version 2 but rejected. The champion alias stayed on version 1 and the served files were unchanged.
- **Reproducibility:** the training job reproduced the backtest's window scores to within 0.0001.
- **API:** 300 random item-store requests for 28-day forecasts took **8.4 ms at p50 and 15.3 ms at p99**, in process, one at a time, on a laptop CPU. That excludes any network hop and concurrent load, which weren't tested. The 80 % interval covered 78.5 % of actual sales on those series.

**Not done:** no container, no scheduler, and no deployment. The steps run as command-line jobs, and retraining isn't automated.

**A bug real data caught:** the first version of the training job trained up to the last day of M5, where the calendar and prices also end, so the service had nothing to build forecast features from. The small test fixture had spare calendar days, which hid it. The service now forecasts from day 1,941, and a regression test covers this. Those 28 days are historical, so this demonstrates the service rather than forecasting a real future.

## Verifying the metric

The WRMSSE implementation reproduces published M5 figures on the private-leaderboard window (d_1942 to d_1969):

| Check | This code | Published |
|---|---|---|
| Seasonal naive | 0.8470 | 0.847 |
| Winner (YJ_STU) | 0.5204 | 0.5204 |
| Official weights | max difference 8e-11 | `weights_evaluation.csv` |

This check caught a real mistake: scoring on d_1914 to d_1941, the public window, gave plausible but wrong numbers (winner 0.699).

## Run it

```bash
pip install -e ".[dev]"
pytest
demandcast-backtest --out results
demandcast-replay --out results
demandcast-train --artifacts artifacts
DEMAND_ARTIFACTS=artifacts uvicorn demandcast.api:app --port 8080
python scripts/measure_api.py --artifacts artifacts --out results/api.json
```

M5 is downloaded at run time and not redistributed.
