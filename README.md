# demand-forecasting-mlops: M5 retail forecasting and the operations around it

A global LightGBM model on M5 (30,490 series), with rolling-origin backtests, conformal intervals, a gated MLflow registry, drift-triggered retraining and a monitored FastAPI service.

**Author:** Kishor Hange

## Status

The backtest below is complete. **The retraining replay and the training job haven't been run on M5**, and the registry and API have only been tested on small synthetic data, so this repository makes no claim about when to retrain or about serving latency. Run `demandcast-replay` and `demandcast-train` to produce those.

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
```

M5 is downloaded at run time and not redistributed.
