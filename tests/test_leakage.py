"""The core guarantee: nothing observed after (D - lag) can influence features for date D."""
import numpy as np
import pandas as pd
import pytest

from src.cinema_demand.features import ALL_FEATURES, build_features


@pytest.mark.parametrize("lag", [2, 3])
def test_features_invariant_to_future_sales(synth, lag):
    base = build_features(synth, lag=lag)
    rng = np.random.default_rng(1)
    for D in pd.to_datetime(["2018-04-10", "2018-05-02", "2018-05-25"]):
        perturbed = synth.copy()
        future = perturbed["date"] > D - pd.Timedelta(days=lag)  # includes day D itself
        perturbed.loc[future, "tickets_sold"] = rng.integers(0, 5000, future.sum()).astype(float)
        pert = build_features(perturbed, lag=lag)
        a = base[base["date"] == D].reset_index(drop=True)[ALL_FEATURES]
        b = pert[pert["date"] == D].reset_index(drop=True)[ALL_FEATURES]
        pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_features_do_change_when_allowed_history_changes(synth):
    """Sanity check that the test above has teeth: perturbing *past* sales does change features."""
    D = pd.Timestamp("2018-05-02")
    base = build_features(synth)
    p = synth.copy()
    past = p["date"] <= D - pd.Timedelta(days=2)
    p.loc[past, "tickets_sold"] = p.loc[past, "tickets_sold"] * 2
    a = base[base["date"] == D]["film_tix_7d"].to_numpy()
    b = build_features(p).query("date == @D")["film_tix_7d"].to_numpy()
    assert not np.allclose(a, b, equal_nan=True)


def test_forecast_rows_with_nan_sales_do_not_leak_into_history(synth):
    """Rows to be forecast (NaN sales) must not contribute to any history feature."""
    D = pd.Timestamp("2018-05-02")
    hist = synth[synth["date"] < D]
    sched = synth[synth["date"] == D].assign(tickets_sold=np.nan)
    f = build_features(pd.concat([hist, sched], ignore_index=True))
    g = build_features(synth[synth["date"] <= D])
    cols = [c for c in ALL_FEATURES if c != "cinema_code"]
    pd.testing.assert_frame_equal(f[f["date"] == D].reset_index(drop=True)[cols],
                                  g[g["date"] == D].reset_index(drop=True)[cols], check_dtype=False)
