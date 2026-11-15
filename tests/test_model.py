import numpy as np
import pandas as pd
import pytest

from src.cinema_demand.data import chronological_split
from src.cinema_demand.features import ALL_FEATURES, build_features
from src.cinema_demand.models import DemandModel, POINT_SPECS
from src.cinema_demand.predict import Forecaster


@pytest.fixture(scope="module")
def fitted(synth):
    f = build_features(synth)
    tr, va, te = chronological_split(f, "2018-04-20", "2018-05-10")
    m = DemandModel.fit(tr, va, ALL_FEATURES, alpha=0.2, seed=0)
    return m, f, tr, va, te


def test_predictions_respect_physical_bounds(fitted):
    m, f, tr, va, te = fitted
    p = m.predict(te)
    cap = te["capacity"].to_numpy()
    assert (p["pred"] >= 0).all() and (p["pred"] <= cap + 1e-9).all()
    assert (p["lo"] <= p["pred"] + 1e-9).all() and (p["pred"] <= p["hi"] + 1e-9).all()
    assert (p["hi"] <= cap + 1e-9).all()
    assert abs(sum(m.blend_weights.values()) - 1) < 1e-9


def test_intervals_reach_reasonable_coverage_on_stationary_data(fitted):
    m, f, tr, va, te = fitted
    p = m.predict(te)
    cov = np.mean((te["tickets_sold"] >= p["lo"]) & (te["tickets_sold"] <= p["hi"]))
    assert cov > 0.70


def test_recalibration_only_uses_sales_up_to_t_minus_lag(fitted):
    m, f, tr, va, te = fitted
    ctx = f[f["date"] > pd.Timestamp("2018-04-20")]
    D = te["date"].min() + pd.Timedelta(days=10)
    day = te[te["date"] == D]
    a = m.predict(day, context=ctx, window=14, lag=2, min_rows=50)
    ctx2 = ctx.copy()
    fut = ctx2["date"] > D - pd.Timedelta(days=2)
    ctx2.loc[fut, "tickets_sold"] = 9999.0                      # corrupt everything after D-2
    b = m.predict(day, context=ctx2, window=14, lag=2, min_rows=50)
    pd.testing.assert_frame_equal(a, b)
    assert a["recalibrated"].all()


def test_forecaster_ignores_history_on_or_after_target_date(fitted, synth):
    m, f, tr, va, te = fitted
    D = pd.Timestamp("2018-05-20")
    sched = synth[synth["date"] == D][["date", "cinema_code", "film_code", "show_time"]]
    cfg = {"forecast": {"lag_days": 2}, "staffing": {"tickets_per_staff": 100, "min_staff": 2}}
    a = Forecaster(m, synth[synth["date"] < D], cfg).forecast(sched)
    b = Forecaster(m, synth[synth["date"] <= D + pd.Timedelta(days=5)], cfg).forecast(sched)
    pd.testing.assert_frame_equal(a, b)
    assert {"pred", "lo", "hi", "cold_start_film", "staff_recommended"} <= set(a.columns)
    assert (a["staff_recommended"] >= 2).all()


def test_explanations_have_requested_length(fitted):
    m, f, tr, va, te = fitted
    ex = m.explain(te.head(3), top_k=4)
    assert len(ex) == 3 and all(len(r) == 4 for r in ex)
