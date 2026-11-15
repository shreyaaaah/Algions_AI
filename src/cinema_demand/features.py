"""Leakage-safe feature engineering for show-level demand at a T-`lag` day horizon.

Information contract (what is known when forecasting show s on date D):
  * SCHEDULE of day D is known: which films play in which cinema/slot, and each
    hall's capacity (programmes are published in advance).
  * TICKET SALES are known only for dates <= D - lag (default lag = 2 -> T-48h).
Every feature is built from one of those two sources. `tests/test_leakage.py`
enforces the second rule by perturbing sales after D-lag and asserting that
nothing changes.

Rows whose `tickets_sold` is NaN are treated as "to be forecast": they
contribute to schedule features but never to history features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAG_DEFAULT = 2
CAT_COLS = ["cinema_code"]

FEATURE_GROUPS = {
    "calendar": ["weekday", "month", "day_of_month", "week_of_year", "show_time"],
    "capacity": ["capacity", "log_capacity", "capacity_imputed"],
    "history_core": [
        "film_tix_7d", "film_occ_7d", "film_tix_14d", "film_occ_14d",
        "cinema_tix_7d", "cinema_occ_7d", "cinema_tix_14d", "cinema_occ_14d",
        "cinslot_tix_14d", "cinslot_occ_14d",
        "cinfilm_tix_14d", "cinfilm_occ_14d",
        "tix_lag7", "film_age_days", "film_n_14d", "cinema_n_14d",
    ],
    "schedule": [
        "film_day_shows", "film_day_cinemas", "film_day_cap", "film_screens_vs_7d",
        "cin_day_shows", "cin_day_films", "cin_day_cap", "cap_share_cinday",
        "cf_day_shows", "slot_rank", "slot_is_first", "slot_is_last", "cin_day_slots",
    ],
    "history_extended": [
        "film_tix_28d", "film_occ_28d", "cinema_tix_28d", "cinema_occ_28d",
        "cinfilm_tix_7d", "cinfilm_occ_7d", "cinfilm_tix_28d", "cinfilm_occ_28d",
        "cinslot_tix_7d", "cinslot_occ_7d", "cinslot_tix_28d", "cinslot_occ_28d",
        "cinwd_tix_28d", "cinwd_occ_28d", "filmwd_tix_28d", "filmwd_occ_28d",
        "film_trend_7_28", "cinfilm_trend_7_28", "cinema_cap_28d",
        "last_seen_tix", "last_seen_occ", "last_seen_age", "tix_lag14",
        "cinfilm_age_days", "cinfilm_n_28d", "cinema_code",
    ],
}
ALL_FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]


# --------------------------------------------------------------------------- #
def _window_block(df, t, T, keys, windows, lag, name, count_windows=()):
    """Trailing-window stats over calendar days [D-lag-w+1, D-lag] per entity."""
    gid = df.groupby(keys, sort=False).ngroup().to_numpy()
    G = int(gid.max()) + 1
    obs = df["tickets_sold"].notna().to_numpy()
    flat = gid * T + t

    def cum(w):
        c = np.zeros((G, T + 1))
        c[:, 1:] = np.bincount(flat, weights=w, minlength=G * T).reshape(G, T).cumsum(1)
        return c

    S = cum(np.where(obs, df["tickets_sold"].fillna(0).to_numpy(), 0.0))
    C = cum(np.where(obs, df["capacity"].to_numpy(), 0.0))
    N = cum(obs.astype(float))
    hi1 = np.clip(t - lag + 1, 0, None)
    out = {}
    for w in windows:
        lo = np.minimum(np.clip(t - lag - w + 1, 0, None), hi1)
        s, c, n = S[gid, hi1] - S[gid, lo], C[gid, hi1] - C[gid, lo], N[gid, hi1] - N[gid, lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"{name}_tix_{w}d"] = np.where(n > 0, s / n, np.nan)
            out[f"{name}_occ_{w}d"] = np.where(c > 0, s / c, np.nan)
            if name == "cinema" and w == 28:
                out["cinema_cap_28d"] = np.where(n > 0, c / n, np.nan)
        if w in count_windows:
            out[f"{name}_n_{w}d"] = n
    return out


def build_features(df: pd.DataFrame, lag: int = LAG_DEFAULT, censor_date=None) -> pd.DataFrame:
    """Return df (sorted) with model features appended. Input needs columns
    date, cinema_code, film_code, show_time, capacity, tickets_sold(NaN = forecast row)
    and optionally capacity_imputed."""
    df = df.copy()
    if "capacity_imputed" not in df:
        df["capacity_imputed"] = 0
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "cinema_code", "film_code", "show_time"]).reset_index(drop=True)
    d0 = df["date"].min()
    censor = pd.Timestamp(censor_date) if censor_date is not None else d0
    t = (df["date"] - d0).dt.days.to_numpy()
    T = int(t.max()) + 1

    # calendar / capacity
    df["weekday"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["log_capacity"] = np.log1p(df["capacity"])

    # trailing-window history (sales only up to D - lag)
    blocks = {}
    blocks.update(_window_block(df, t, T, ["film_code"], (7, 14, 28), lag, "film", (14,)))
    blocks.update(_window_block(df, t, T, ["cinema_code"], (7, 14, 28), lag, "cinema", (14, 28)))
    blocks.update(_window_block(df, t, T, ["cinema_code", "film_code"], (7, 14, 28), lag, "cinfilm", (28,)))
    blocks.update(_window_block(df, t, T, ["cinema_code", "show_time"], (7, 14, 28), lag, "cinslot"))
    blocks.update(_window_block(df, t, T, ["cinema_code", "weekday"], (28,), lag, "cinwd"))
    blocks.update(_window_block(df, t, T, ["film_code", "weekday"], (28,), lag, "filmwd"))
    # film-level n over 7d is needed for the screen-footprint ratio
    film7 = _window_block(df, t, T, ["film_code"], (7,), lag, "film", (7,))
    blocks["_film_n_7d"] = film7["film_n_7d"]
    for k, v in blocks.items():
        df[k] = v
    with np.errstate(invalid="ignore", divide="ignore"):
        df["film_trend_7_28"] = df["film_tix_7d"] / df["film_tix_28d"]
        df["cinfilm_trend_7_28"] = df["cinfilm_tix_7d"] / df["cinfilm_tix_28d"]

    # exact lags and last-seen value of the same cinema/film/slot
    key = ["cinema_code", "film_code", "show_time"]
    obs_rows = df.loc[df["tickets_sold"].notna(), key + ["date", "tickets_sold", "capacity"]]
    for k in (7, 14):
        r = obs_rows[key + ["date", "tickets_sold"]].copy()
        r["date"] = r["date"] + pd.Timedelta(days=k)
        r = r.rename(columns={"tickets_sold": f"tix_lag{k}"})
        df = df.merge(r, on=key + ["date"], how="left")
    q = df[key + ["date"]].copy()
    q["_i"] = np.arange(len(q))
    q["_qd"] = q["date"] - pd.Timedelta(days=lag)
    q = q.sort_values("_qd")
    right = (obs_rows.rename(columns={"date": "_last_date", "tickets_sold": "last_seen_tix",
                                      "capacity": "_last_cap"})
                     .sort_values("_last_date"))
    m = pd.merge_asof(q, right, left_on="_qd", right_on="_last_date", by=key, direction="backward")
    m = m.sort_values("_i")
    df["last_seen_tix"] = m["last_seen_tix"].to_numpy()
    df["last_seen_occ"] = (m["last_seen_tix"] / m["_last_cap"]).to_numpy()
    df["last_seen_age"] = (m["date"] - m["_last_date"]).dt.days.to_numpy()

    # schedule-known features of day D (footprint of the film / cinema)
    g = df.groupby(["film_code", "date"])
    df["film_day_shows"] = g["cinema_code"].transform("size")
    df["film_day_cinemas"] = g["cinema_code"].transform("nunique")
    df["film_day_cap"] = g["capacity"].transform("sum")
    with np.errstate(invalid="ignore", divide="ignore"):
        daily7 = df["_film_n_7d"] / 7.0
        df["film_screens_vs_7d"] = np.where(daily7 > 0, df["film_day_shows"] / daily7, np.nan)
    g = df.groupby(["cinema_code", "date"])
    df["cin_day_shows"] = g["film_code"].transform("size")
    df["cin_day_films"] = g["film_code"].transform("nunique")
    df["cin_day_cap"] = g["capacity"].transform("sum")
    df["cin_day_slots"] = g["show_time"].transform("nunique")
    df["cap_share_cinday"] = df["capacity"] / df["cin_day_cap"]
    df["cf_day_shows"] = df.groupby(["cinema_code", "film_code", "date"])["show_time"].transform("size")
    df["slot_rank"] = g["show_time"].rank(method="dense")
    df["slot_is_first"] = (df["show_time"] == g["show_time"].transform("min")).astype(int)
    df["slot_is_last"] = (df["show_time"] == g["show_time"].transform("max")).astype(int)

    # release / run age. First appearance is the schedule-known release date; when a
    # film is already running on the first day of the data its age is unknown (NaN).
    for name, cols in (("film", ["film_code"]), ("cinfilm", ["cinema_code", "film_code"])):
        first = df.groupby(cols)["date"].transform("min")
        age = (df["date"] - first).dt.days.astype(float)
        df[f"{name}_age_days"] = np.where(first <= censor, np.nan, age)

    df["capacity_imputed"] = df["capacity_imputed"].astype(int)
    return df.drop(columns=["_film_n_7d"])


def feature_matrix(df: pd.DataFrame, feature_list, cinema_categories=None) -> pd.DataFrame:
    X = df[list(feature_list)].copy()
    if "cinema_code" in X:
        cats = cinema_categories if cinema_categories is not None else sorted(df["cinema_code"].unique())
        X["cinema_code"] = pd.Categorical(X["cinema_code"], categories=list(cats))
    return X
