# demand-forecasting-mlops: M5 retail forecasting and the operations around it

A global LightGBM model on M5 (30,490 series), with rolling-origin backtests, conformal intervals, a gated MLflow registry, drift-triggered retraining and a monitored FastAPI service.

**Author:** Kishor Hange

## Status: evaluation not finished

The code and tests are complete, but this repository doesn't yet claim a final score. **The final M5 window hasn't been scored for the model, and the retraining replay and the training job haven't been run on M5.** Run `demandcast-backtest`, `demandcast-replay` and `demandcast-train` to produce them.

## Verified so far

The WRMSSE implementation reproduces published M5 figures on the private-leaderboard window (d_1942 to d_1969):

| Check | This code | Published |
|---|---|---|
| Seasonal naive | 0.8470 | 0.847 |
| Winner (YJ_STU) | 0.5204 | 0.5204 |
| Official weights | max difference 8e-11 | `weights_evaluation.csv` |

This check caught a real mistake: scoring on d_1914 to d_1941, the public window, gave plausible but wrong numbers (winner 0.699).

Partial backtest, tuning windows only (lower is better):

| Window origin | LightGBM, two years of history | Best baseline (weekday average) |
|---|---|---|
| 1857 | 0.660 | 0.718 |
| 1885 | 0.524 | 0.655 |

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
