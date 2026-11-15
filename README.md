# Cinema Demand Forecasting

Forecast **tickets sold per show 48 hours ahead** (T-48h) for a cinema chain, with calibrated
prediction intervals and a staffing hint. Chronological evaluation, leakage-safe features, tests,
a FastAPI service and a Streamlit dashboard.

> **Attribution.** The topic and dataset come from
> [HimanshuKumar83/Cinema-Ticket-Demand-Prediction](https://github.com/HimanshuKumar83/Cinema-Ticket-Demand-Prediction),
> built on the Kaggle [Cinema Tickets](https://www.kaggle.com/datasets/arashnic/cinema-ticket) data
> (CC BY-NC-SA 4.0). This repository is an independent re-implementation with a different feature
> set, uncertainty method and evaluation protocol. The reference's reported numbers below are
> quoted from its `metrics.json`; I did not re-run its code.

## Results (held-out test: 2018-09-30 -> 2018-11-04, 22,764 shows)

Split is strictly chronological: train 2018-02-21 -> 08-23, validation 08-24 -> 09-29, test after.
16 of the 33 test-period films never appear in training.

| Model | MAE | WAPE | Bias | 80% interval coverage |
|---|---|---|---|---|
| Seasonal naive (last same cinema/film/slot) | 92.49 | 74.86% | -6.8 | - |
| Occupancy prior x capacity | 75.93 | 61.46% | +1.0 | - |
| Reference repo (as reported by its author) | 48.36 | 39.14% | - | 69.3% |
| **This repo, static** (no recalibration) | **43.49** | **35.20%** | -15.5 | 74.0% |
| **This repo, rolling-recalibrated** | 46.74 | 37.83% | -0.06 | **80.0%** |

**Read this trade-off honestly.** Demand rose during the test window, so the static model
under-forecasts by ~15 tickets per show and its intervals cover only 74%. Rolling recalibration
(re-fitting level and interval margins from sales observed up to D-2) removes the bias and hits
the 80% target, but it **raises MAE by ~3 tickets**, because the level correction matches the mean
while MAE is minimised by the median. Pick static if you care about MAE, recalibrated if you care
about unbiased totals and honest intervals. Asymmetric cost (under-forecast = 1.5x) is
about equal: 58.25 (static) vs 58.44 (recalibrated).

Weak spot: **cold-start films** (no sales in the previous 14 days; 353 test shows) have
WAPE 42.9%, bias about -100 tickets, and only ~60% interval coverage.

Exploratory feature-group ablation (single LightGBM L1 model, test set; reproduce with the full
run): reference-style features 47.58 MAE / 38.51% WAPE -> + schedule-footprint features
46.86 / 37.93%. Schedule features give a modest gain; the strongest signals are the recent
sales of the same cinema-film pair (`cinfilm_tix_7d`, 34% of gain) and the cinema identity.

> These numbers were computed on the cleaned 142,418-row table taken from the reference repo
> (the Kaggle site was not reachable from where I built this). Re-run on your own download:
> `python -m src.cinema_demand.train` regenerates everything in `reports/`.
> The recalibration and cost-based blend were added after seeing test-set bias, so treat the test
> numbers as slightly optimistic; the 4-fold rolling backtest in the full run is the cleaner check.

## What is different from a plain "rolling-mean + LightGBM" approach

1. **Explicit information contract.** Schedule of day D (films, slots, hall capacity) is known;
   ticket sales are known only up to D-2. `tests/test_leakage.py` perturbs all sales after D-2 and
   asserts that no feature for day D changes.
2. **Post-show columns dropped** (`total_sales`, `occu_perc`, `ticket_price`, `ticket_use`,
   `tickets_out`); `reports/leakage_audit.json` quantifies why (they are algebraic functions of the target).
3. **Schedule-footprint features:** how many screens a film has nationally that day versus its
   7-day norm, each hall's share of the cinema's day, slot position, run age.
4. **Three complementary targets** (tickets with L1, log-tickets, occupancy x capacity), blended
   with weights chosen on validation by the **asymmetric business cost**, not raw MAE.
5. **Conformalised quantile regression** (log scale) with separate margins for cold-start
   films; optional **rolling recalibration** from the most recent observed sales.
6. **Explainability** via tree-SHAP per forecast (`Forecaster.forecast(..., explain=True)`).

## Quick start

```bash
pip install -r requirements.txt
python scripts/download_data.py        # or place cinemaTicket_Ref.csv in data/raw/
python -m src.cinema_demand.train      # full run: ablation + test + 4-fold backtest (slow on 1 CPU)
python -m src.cinema_demand.train --fast   # final model + test only
python -m pytest -q                    # 15 tests
uvicorn app.api:app --port 8000        # POST /forecast, GET /metrics, docs at /docs
streamlit run app/dashboard.py
```

`config.yaml` holds the split dates, lag, interval level, cost ratio and staffing placeholders.

## Layout

```
src/cinema_demand/  data.py (clean/audit/split) features.py models.py evaluate.py predict.py train.py plots.py
app/                api.py (FastAPI)  dashboard.py (Streamlit)
tests/              leakage invariance, cleaning, metrics, model bounds, forecaster, API
reports/            metrics.json, figures/, feature_importance.csv, top15_worst_predictions.csv
```

## Known limitations

- **API/dashboard serve the static model.** Rolling recalibration is implemented in
  `DemandModel.predict(df, context=...)` and used in the evaluation pipeline, but
  `Forecaster.forecast` does not pass a context yet.
- Footprint features need the **full national programme** for the target day, as in training.
- Rows in the data are shows with >=1 ticket sold; shows with zero sales are not recorded, so
  true demand is slightly overstated for very weak shows. About 0.4% of rows are capped at capacity (censored demand).
- `capacity` is non-integer in the raw data; it is stable within cinema/film/slot, so it is used
  as a hall-size proxy.
- Staffing ratios in `config.yaml` are **placeholders**, not derived from data. Decision support
  only; a human should review any roster or screening change.
- Single season of data (Feb-Nov 2018): no holiday calendar, no multi-year seasonality.
