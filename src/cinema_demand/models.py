"""Point forecasters, cost-aware blend, and conformalised quantile intervals
with optional rolling recalibration from recently observed sales."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import feature_matrix

BASE_PARAMS = dict(
    learning_rate=0.04, num_leaves=63, min_child_samples=40, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, cat_smooth=20, min_data_per_group=50,
    verbose=-1, n_jobs=-1,
)
POINT_SPECS = {
    # name: (LightGBM objective, target transform)
    "tickets_l1": ("l1", "raw"),
    "log_l2": ("l2", "log"),
    "occupancy_xent": ("cross_entropy", "occ"),
}
FACTOR_BOUNDS = (0.8, 1.25)


def _target(df, kind):
    if kind == "raw":
        return df["tickets_sold"].to_numpy(float)
    if kind == "log":
        return np.log1p(df["tickets_sold"].to_numpy(float))
    return np.clip(df["occupancy"].to_numpy(float), 0.0, 1.0)


def _inverse(pred, kind, cap):
    if kind == "raw":
        return pred
    if kind == "log":
        return np.expm1(pred)
    return pred * cap


def fit_booster(Xtr, ytr, Xva, yva, objective, seed=42, alpha=None, max_rounds=4000):
    params = dict(BASE_PARAMS, objective=objective, seed=seed)
    if alpha is not None:
        params["alpha"] = alpha
    dtr = lgb.Dataset(Xtr, ytr, categorical_feature="auto")
    dva = lgb.Dataset(Xva, yva, reference=dtr)
    return lgb.train(params, dtr, num_boost_round=max_rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def _groups(df):
    return np.where(df["film_n_14d"].fillna(0).to_numpy() == 0, "cold_film", "established_film")


def _cqr_margin(scores, alpha):
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def asym_cost(y, p, cu, co):
    e = p - y
    return float(np.mean(np.where(e < 0, -cu * e, co * e)))


@dataclass
class DemandModel:
    feature_list: list
    cinema_categories: list
    boosters: dict = field(default_factory=dict)
    blend_weights: dict = field(default_factory=dict)
    conformal: dict = field(default_factory=dict)  # static group -> log-space margin
    alpha: float = 0.20
    cu: float = 1.5
    co: float = 1.0
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(cls, train, val, feature_list, alpha=0.20, seed=42, specs=POINT_SPECS, cu=1.5, co=1.0):
        cats = sorted(train["cinema_code"].unique())
        Xtr, Xva = (feature_matrix(d, feature_list, cats) for d in (train, val))
        m = cls(list(feature_list), cats, alpha=alpha, cu=cu, co=co)
        m.meta = {"train_end": str(train["date"].max().date()), "oos_start": str(val["date"].min().date())}
        raw_val = {}
        cap_v = val["capacity"].to_numpy(float)
        for name, (obj, kind) in specs.items():
            b = fit_booster(Xtr, _target(train, kind), Xva, _target(val, kind), obj, seed)
            m.boosters[name] = (b, kind)
            raw_val[name] = np.clip(_inverse(b.predict(Xva, num_iteration=b.best_iteration), kind, cap_v), 0, cap_v)
        m.blend_weights = _fit_blend(raw_val, val["tickets_sold"].to_numpy(), cu, co)
        ltr, lva = np.log1p(train["tickets_sold"]), np.log1p(val["tickets_sold"])
        for tag, a in (("q_lo", alpha / 2), ("q_hi", 1 - alpha / 2)):
            m.boosters[tag] = (fit_booster(Xtr, ltr, Xva, lva, "quantile", seed, alpha=a), "logq")
        m._calibrate_static(val)
        m.meta["best_iterations"] = {k: int(v[0].best_iteration) for k, v in m.boosters.items()}
        return m

    def _raw_log_interval(self, X):
        lo = self.boosters["q_lo"][0].predict(X, num_iteration=self.boosters["q_lo"][0].best_iteration)
        hi = self.boosters["q_hi"][0].predict(X, num_iteration=self.boosters["q_hi"][0].best_iteration)
        return np.minimum(lo, hi), np.maximum(lo, hi)

    def _calibrate_static(self, val):
        X = feature_matrix(val, self.feature_list, self.cinema_categories)
        lo, hi = self._raw_log_interval(X)
        y = np.log1p(val["tickets_sold"].to_numpy())
        score, g = np.maximum(lo - y, y - hi), _groups(val)
        self.conformal = {"all": _cqr_margin(score, self.alpha)}
        for k in np.unique(g):
            self.conformal[str(k)] = _cqr_margin(score[g == k], self.alpha) if (g == k).sum() >= 300 else self.conformal["all"]

    # -------------------------------------------------------------- predict
    def _static(self, df):
        X = feature_matrix(df, self.feature_list, self.cinema_categories)
        cap = df["capacity"].to_numpy(float)
        comps = {n: np.clip(_inverse(b.predict(X, num_iteration=b.best_iteration), k, cap), 0, cap)
                 for n, (b, k) in self.boosters.items() if n not in ("q_lo", "q_hi")}
        point = sum(self.blend_weights[k] * v for k, v in comps.items())
        qlo, qhi = self._raw_log_interval(X)
        return point, comps, qlo, qhi, cap

    def predict(self, df, context: pd.DataFrame | None = None, window: int = 28, lag: int = 2,
                min_rows: int = 500, min_group_rows: int = 300) -> pd.DataFrame:
        """Forecast `df`. If `context` (feature rows with *observed* sales, out-of-sample for
        the model, i.e. dated after the training period) is supplied, the level and the
        interval margins are recalibrated per target date D from context rows dated in
        [D-lag-window+1, D-lag] - exactly the sales that are known at T-48h."""
        point, comps, qlo, qhi, cap = self._static(df)
        n = len(df)
        factor = np.ones(n)
        marg = np.array([self.conformal.get(str(g), self.conformal["all"]) for g in _groups(df)])
        used = np.zeros(n, dtype=bool)
        if context is not None and len(context):
            cp, _, cqlo, cqhi, _ = self._static(context)
            cy = context["tickets_sold"].to_numpy(float)
            cly, cdate, cg = np.log1p(cy), context["date"].to_numpy(), _groups(context)
            ddate, dg = df["date"].to_numpy(), _groups(df)
            for D in np.unique(ddate):
                w = (cdate >= D - np.timedelta64(lag + window - 1, "D")) & (cdate <= D - np.timedelta64(lag, "D"))
                if w.sum() < min_rows:
                    continue
                f = float(np.clip(cy[w].sum() / max(cp[w].sum(), 1e-9), *FACTOR_BOUNDS))
                lf = np.log(f)
                sc = np.maximum(cqlo[w] + lf - cly[w], cly[w] - cqhi[w] - lf)
                allm = _cqr_margin(sc, self.alpha)
                gm = {k: (_cqr_margin(sc[cg[w] == k], self.alpha) if (cg[w] == k).sum() >= min_group_rows else allm)
                      for k in np.unique(cg[w])}
                rows = ddate == D
                factor[rows] = f
                marg[rows] = [gm.get(g, allm) for g in dg[rows]]
                used[rows] = True
        lf = np.log(factor)
        point_f = np.clip(point * factor, 0, cap)
        lo = np.clip(np.expm1(qlo + lf - marg), 0, cap)
        hi = np.clip(np.expm1(qhi + lf + marg), 0, cap)
        lo, hi = np.minimum(lo, point_f), np.maximum(hi, point_f)
        out = pd.DataFrame({"pred": point_f, "lo": lo, "hi": hi, "pred_occupancy": point_f / cap,
                            "recal_factor": factor, "recalibrated": used}, index=df.index)
        for k, v in comps.items():
            out[f"pred_{k}"] = v
        return out

    # -------------------------------------------------------------- explain
    def explain(self, df, top_k=5):
        """Tree-SHAP contributions of the tickets_l1 model (raw-ticket units; + raises forecast)."""
        b, _ = self.boosters["tickets_l1"]
        X = feature_matrix(df, self.feature_list, self.cinema_categories)
        contrib = b.predict(X, num_iteration=b.best_iteration, pred_contrib=True)[:, :-1]
        names = np.array(self.feature_list)
        return [[(names[i], float(r[i])) for i in np.argsort(-np.abs(r))[:top_k]] for r in contrib]

    def importance(self) -> pd.DataFrame:
        b, _ = self.boosters["tickets_l1"]
        return (pd.DataFrame({"feature": self.feature_list, "gain": b.feature_importance("gain")})
                .assign(gain_pct=lambda d: 100 * d["gain"] / d["gain"].sum())
                .sort_values("gain", ascending=False).reset_index(drop=True))

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "DemandModel":
        return joblib.load(path)


def _fit_blend(preds: dict, y, cu, co, step=0.1) -> dict:
    """Blend weights minimise the *asymmetric operational cost* on validation."""
    names = list(preds)
    best, best_w = np.inf, None
    grid = np.arange(0, 1 + 1e-9, step)
    for w in itertools.product(grid, repeat=len(names) - 1):
        if sum(w) > 1 + 1e-9:
            continue
        ws = list(w) + [1 - sum(w)]
        c = asym_cost(y, sum(wi * preds[n] for wi, n in zip(ws, names)), cu, co)
        if c < best:
            best, best_w = c, ws
    return dict(zip(names, map(float, best_w)))
