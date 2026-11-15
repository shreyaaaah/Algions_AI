"""Inference: forecast a published schedule using only history up to T-lag."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import load_config
from .features import build_features
from .models import DemandModel

SCHEDULE_COLS = ["date", "cinema_code", "film_code", "show_time"]


class Forecaster:
    def __init__(self, model: DemandModel, history: pd.DataFrame, cfg: dict | None = None):
        self.model = model
        self.cfg = cfg or load_config()
        self.lag = self.cfg["forecast"]["lag_days"]
        self.history = history.copy()
        self.history["date"] = pd.to_datetime(self.history["date"])
        if "capacity_imputed" not in self.history:
            self.history["capacity_imputed"] = 0
        self.censor_date = self.history["date"].min()

    def _fill_capacity(self, sched: pd.DataFrame) -> pd.DataFrame:
        s = sched.copy()
        if "capacity" not in s:
            s["capacity"] = np.nan
        s["capacity"] = pd.to_numeric(s["capacity"], errors="coerce")
        inferred = s["capacity"].isna()
        h = self.history
        last = (h.sort_values("date").groupby(["cinema_code", "film_code", "show_time"])["capacity"].last())
        cin = h.groupby("cinema_code")["capacity"].median()
        k = list(zip(s["cinema_code"], s["film_code"], s["show_time"]))
        s["capacity"] = s["capacity"].fillna(pd.Series(last.reindex(k).to_numpy(), index=s.index))
        s["capacity"] = s["capacity"].fillna(s["cinema_code"].map(cin)).fillna(h["capacity"].median())
        s["capacity_imputed"] = inferred.astype(int)
        return s

    def forecast(self, schedule: pd.DataFrame, explain: bool = False) -> pd.DataFrame:
        """`schedule`: full programme for the target date(s) across cinemas (film footprint
        features need the whole day's schedule, as in training). Needs date, cinema_code,
        film_code, show_time; capacity optional."""
        missing = [c for c in SCHEDULE_COLS if c not in schedule]
        if missing:
            raise ValueError(f"schedule is missing columns: {missing}")
        sched = self._fill_capacity(schedule)
        sched["date"] = pd.to_datetime(sched["date"])
        sched["tickets_sold"] = np.nan
        keep = ["date", "cinema_code", "film_code", "show_time", "tickets_sold",
                "capacity", "capacity_imputed"]
        hist = self.history[self.history["date"] < sched["date"].min()]
        combo = pd.concat([hist[keep], sched[keep]], ignore_index=True)
        feats = build_features(combo, lag=self.lag, censor_date=self.censor_date)
        target = feats[feats["tickets_sold"].isna()].copy()
        pred = self.model.predict(target)
        out = target[SCHEDULE_COLS + ["capacity"]].join(pred)
        out["cold_start_film"] = (target["film_n_14d"].fillna(0) == 0).to_numpy()
        out["low_history"] = (target["cinfilm_n_28d"].fillna(0) < 3).to_numpy()
        out["staff_recommended"] = [self.staffing(h) for h in out["hi"]]
        if explain:
            out["top_drivers"] = self.model.explain(target)
        return out.reset_index(drop=True)

    def staffing(self, upper_bound: float) -> int:
        s = self.cfg["staffing"]
        return int(max(s["min_staff"], math.ceil(upper_bound / s["tickets_per_staff"])))
