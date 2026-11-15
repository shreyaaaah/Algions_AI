"""Metrics, sliced evaluation and interval diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def point_metrics(y, p, cap, cu=1.5, co=1.0) -> dict:
    y, p, cap = map(np.asarray, (y, p, cap))
    err = p - y
    cost = np.where(err < 0, cu * (-err), co * err)
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "WAPE_pct": float(100 * np.abs(err).sum() / y.sum()),
        "Occupancy_MAE": float(np.mean(np.abs(err) / cap)),
        "Asymmetric_Cost": float(cost.mean()),
        "Bias": float(err.mean()),
        "N": int(len(y)),
    }


def interval_metrics(y, lo, hi, cap) -> dict:
    y, lo, hi, cap = map(np.asarray, (y, lo, hi, cap))
    return {
        "coverage_pct": float(100 * np.mean((y >= lo) & (y <= hi))),
        "mean_width": float(np.mean(hi - lo)),
        "mean_width_occupancy": float(np.mean((hi - lo) / cap)),
        "below_lo_pct": float(100 * np.mean(y < lo)),
        "above_hi_pct": float(100 * np.mean(y > hi)),
    }


def slices(df: pd.DataFrame, pred_col="pred", lo_col="lo", hi_col="hi", cu=1.5, co=1.0) -> dict:
    """df needs tickets_sold, capacity, pred (+ optional lo/hi) and the feature columns."""
    out = {}
    groups = {
        "film_status": np.where(df["film_n_14d"].fillna(0) == 0, "new_or_cold_film", "established_film"),
        "capacity_bucket": pd.cut(df["capacity"], [0, 150, 300, 600, 1e9],
                                  labels=["<=150", "151-300", "301-600", ">600"]).astype(str),
        "weekday": df["weekday"].map(dict(enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]))),
        "slot": np.where(df["show_time"] <= 4, df["show_time"].astype(str), "5+"),
    }
    for name, g in groups.items():
        res = {}
        for val, idx in df.groupby(pd.Series(g, index=df.index)).groups.items():
            s = df.loc[idx]
            r = point_metrics(s["tickets_sold"], s[pred_col], s["capacity"], cu, co)
            if lo_col in s:
                r.update(interval_metrics(s["tickets_sold"], s[lo_col], s[hi_col], s["capacity"]))
            res[str(val)] = {k: round(v, 4) for k, v in r.items()}
        out[name] = res
    return out
