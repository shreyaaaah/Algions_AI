"""End-to-end pipeline:  raw CSV -> clean -> features -> ablation -> final model ->
held-out test -> rolling-origin backtest -> reports & artifacts.

    python -m src.cinema_demand.train            # full run
    python -m src.cinema_demand.train --fast     # skip ablation + backtest (smoke run)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, load_config
from .data import audit_leakage, chronological_split, clean, load_raw
from .evaluate import interval_metrics, point_metrics, slices
from .features import ALL_FEATURES, FEATURE_GROUPS, build_features
from .models import POINT_SPECS, DemandModel
from . import plots

STEPS = [
    ("A. calendar + capacity + core history (reference-style set)", ["calendar", "capacity", "history_core"]),
    ("B. + schedule footprint (film/cinema/slot, known at T-48h)", ["calendar", "capacity", "history_core", "schedule"]),
    ("C. + extended history, trends, last-seen, cinema id (full)", list(FEATURE_GROUPS)),
]


def _rnd(d, n=4):
    return {k: (round(v, n) if isinstance(v, float) else v) for k, v in d.items()}


def baselines(df):
    """Naive references that need no learning (all T-48h safe)."""
    cap = df["capacity"].to_numpy()
    last = (df["last_seen_tix"].fillna(df["cinslot_tix_28d"]).fillna(df["cinema_tix_28d"])
              .fillna(df["film_tix_28d"]).fillna(0))
    occ = (df["cinfilm_occ_28d"].fillna(df["cinslot_occ_28d"]).fillna(df["cinema_occ_28d"])
             .fillna(df["film_occ_28d"]).fillna(0.15))
    return {
        "seasonal_naive (last same cinema/film/slot)": np.clip(last, 0, cap),
        "hierarchical occupancy prior x capacity": np.clip(occ * cap, 0, cap),
    }


def run(cfg=None, fast=False):
    cfg = cfg or load_config()
    out_art, out_rep = ROOT / "artifacts", ROOT / "reports"
    out_art.mkdir(exist_ok=True), out_rep.mkdir(exist_ok=True)
    raw_path = ROOT / cfg["data"]["raw_path"]
    raw = load_raw(raw_path)
    df, clean_rep = clean(raw)
    (out_rep / "cleaning_report.json").write_text(json.dumps(clean_rep, indent=2))
    (out_rep / "leakage_audit.json").write_text(json.dumps(audit_leakage(raw), indent=2))
    proc = ROOT / cfg["data"]["processed_path"]
    proc.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(proc)

    lag, alpha = cfg["forecast"]["lag_days"], 1 - cfg["forecast"]["interval_coverage"]
    cu, co = cfg["cost"]["under_forecast"], cfg["cost"]["over_forecast"]
    feats = build_features(df, lag=lag)
    train, val, test = chronological_split(feats, cfg["split"]["train_end"], cfg["split"]["val_end"])
    split_info = {k: {"rows": int(len(d)), "start": str(d["date"].min().date()), "end": str(d["date"].max().date()),
                      "films": int(d["film_code"].nunique())} for k, d in
                  (("train", train), ("val", val), ("test", test))}
    split_info["test_films_unseen_in_train"] = int(len(set(test["film_code"]) - set(train["film_code"])))
    (out_art / "split_info.json").write_text(json.dumps(split_info, indent=2))
    results = {"config": {"lag_days": lag, "interval_target": 1 - alpha}, "split": split_info}

    # --- baselines on test
    results["baselines"] = {n: _rnd(point_metrics(test["tickets_sold"], p, test["capacity"], cu, co))
                            for n, p in baselines(test).items()}

    # --- ablation (point model = LightGBM L1 on tickets, early-stopped on validation)
    if not fast:
        abl = []
        for label, groups in STEPS[:-1]:
            fl = [f for g in groups for f in FEATURE_GROUPS[g]]
            m = DemandModel.fit(train, val, fl, alpha, cfg["random_seed"],
                                specs={"tickets_l1": POINT_SPECS["tickets_l1"]}, cu=cu, co=co)
            r = point_metrics(test["tickets_sold"], m.predict(test)["pred"], test["capacity"], cu, co)
            abl.append({"step": label, **_rnd(r)})
        results["ablation_single_l1_model"] = abl

    # --- final model: full feature set + blend + conformal intervals
    model = DemandModel.fit(train, val, ALL_FEATURES, alpha, cfg["random_seed"], cu=cu, co=co)
    ctx = feats[feats["date"] > pd.Timestamp(cfg["split"]["train_end"])]   # out-of-sample, observed sales
    pred_static = model.predict(test)
    pred = model.predict(test, context=ctx, window=cfg["forecast"]["recal_window_days"], lag=lag)
    tst = test.join(pred)
    pm = point_metrics(tst["tickets_sold"], pred["pred"], tst["capacity"], cu, co)
    im = interval_metrics(tst["tickets_sold"], pred["lo"], pred["hi"], tst["capacity"])
    comp = {c: _rnd(point_metrics(tst["tickets_sold"], pred[c], tst["capacity"], cu, co))
            for c in pred.columns if c.startswith("pred_") and c != "pred_occupancy"}
    if not fast:
        results["ablation_single_l1_model"].append(
            {"step": STEPS[-1][0], **_rnd(point_metrics(tst["tickets_sold"], pred["pred_tickets_l1"], tst["capacity"], cu, co))})
    results.update({
        "final_model": {"point": _rnd(pm), "interval": _rnd(im), "components": comp,
                        "static_no_recalibration": {
                            "point": _rnd(point_metrics(test["tickets_sold"], pred_static["pred"], test["capacity"], cu, co)),
                            "interval": _rnd(interval_metrics(test["tickets_sold"], pred_static["lo"], pred_static["hi"], test["capacity"]))},
                        "blend_weights": model.blend_weights, "conformal_margins_log": model.conformal,
                        "best_iterations": model.meta["best_iterations"]},
        "slices": slices(tst, cu=cu, co=co),
    })
    # validation reference (what the calibration saw)
    pv = model.predict(val, context=ctx, window=cfg["forecast"]["recal_window_days"], lag=lag)
    results["validation"] = {"point": _rnd(point_metrics(val["tickets_sold"], pv["pred"], val["capacity"], cu, co)),
                             "interval": _rnd(interval_metrics(val["tickets_sold"], pv["lo"], pv["hi"], val["capacity"]))}

    # --- rolling-origin backtest (same recipe, 4 expanding folds)
    if not fast:
        results["backtest"] = backtest(feats, cfg, alpha, cu, co)

    # --- artifacts
    model.save(out_art / "model.joblib")
    imp = model.importance()
    imp.to_csv(out_rep / "feature_importance.csv", index=False)
    tst.assign(abs_err=(tst["pred"] - tst["tickets_sold"]).abs()).nlargest(15, "abs_err")[
        ["date", "cinema_code", "film_code", "show_time", "capacity", "tickets_sold", "pred", "lo", "hi"]
    ].to_csv(out_rep / "top15_worst_predictions.csv", index=False)
    (out_art / "feature_list.json").write_text(json.dumps(ALL_FEATURES, indent=2))
    (out_rep / "metrics.json").write_text(json.dumps(results, indent=2, default=float))
    plots.make_all(tst, imp, results, out_rep)
    return results


def backtest(feats, cfg, alpha, cu, co):
    """Expanding-window folds ending at the last date. Each fold: train < val window < test window."""
    w = pd.Timedelta(days=cfg["backtest"]["window_days"])
    end = feats["date"].max()
    folds = []
    for k in range(cfg["backtest"]["n_folds"] - 1, -1, -1):
        te_end = end - k * w
        te_start, va_start = te_end - w + pd.Timedelta(days=1), te_end - 2 * w + pd.Timedelta(days=1)
        tr = feats[feats["date"] < va_start]
        va = feats[(feats["date"] >= va_start) & (feats["date"] < te_start)]
        te = feats[(feats["date"] >= te_start) & (feats["date"] <= te_end)]
        m = DemandModel.fit(tr, va, ALL_FEATURES, alpha, cfg["random_seed"], cu=cu, co=co)
        ctx = feats[feats["date"] >= va_start]
        ps = m.predict(te)
        pr = m.predict(te, context=ctx, window=cfg["forecast"]["recal_window_days"], lag=cfg["forecast"]["lag_days"])
        base = baselines(te)["hierarchical occupancy prior x capacity"]
        pmx = lambda p: point_metrics(te["tickets_sold"], p, te["capacity"], cu, co)
        imx = lambda p: interval_metrics(te["tickets_sold"], p["lo"], p["hi"], te["capacity"])
        folds.append({"test_window": f"{te_start.date()} -> {te_end.date()}", "n_train": int(len(tr)),
                      "model_MAE": round(pmx(pr["pred"])["MAE"], 2), "model_WAPE_pct": round(pmx(pr["pred"])["WAPE_pct"], 2),
                      "static_WAPE_pct": round(pmx(ps["pred"])["WAPE_pct"], 2),
                      "baseline_WAPE_pct": round(pmx(base)["WAPE_pct"], 2),
                      "model_bias": round(pmx(pr["pred"])["Bias"], 2), "static_bias": round(pmx(ps["pred"])["Bias"], 2),
                      "interval_coverage_pct": round(imx(pr)["coverage_pct"], 2),
                      "static_interval_coverage_pct": round(imx(ps)["coverage_pct"], 2)})
    return folds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    res = run(fast=a.fast)
    print(json.dumps({k: res[k] for k in ("final_model",)}, indent=2, default=float)[:3000])
