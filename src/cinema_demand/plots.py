"""Report figures (matplotlib, no seaborn dependency)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_all(tst, imp, results, out_dir):
    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _pred_vs_actual(tst, fig_dir / "pred_vs_actual.png")
    _daily_tracking(tst, fig_dir / "daily_tracking.png")
    _importance(imp, fig_dir / "feature_importance.png")
    if "backtest" in results:
        _backtest(results["backtest"], fig_dir / "backtest.png")


def _pred_vs_actual(t, path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.hexbin(t["tickets_sold"] + 1, t["pred"] + 1, gridsize=60, xscale="log", yscale="log",
              bins="log", mincnt=1, cmap="viridis")
    lim = [1, max(t["tickets_sold"].max(), t["pred"].max()) * 1.2]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set(xlabel="actual tickets + 1", ylabel="forecast + 1", title="Held-out test: forecast vs actual")
    fig.tight_layout(), fig.savefig(path, dpi=130), plt.close(fig)


def _daily_tracking(t, path):
    d = t.groupby("date").agg(actual=("tickets_sold", "sum"), pred=("pred", "sum"),
                              lo=("lo", "sum"), hi=("hi", "sum"))
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(d.index, d["actual"], label="actual", lw=1.6)
    ax.plot(d.index, d["pred"], label="forecast", lw=1.6)
    ax.set(title="Held-out test: total daily tickets (sum over shows)", ylabel="tickets")
    ax.legend(), fig.autofmt_xdate(), fig.tight_layout(), fig.savefig(path, dpi=130), plt.close(fig)


def _importance(imp, path):
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.barh(top["feature"], top["gain_pct"])
    ax.set(xlabel="% of total split gain", title="Top features (LightGBM tickets model)")
    fig.tight_layout(), fig.savefig(path, dpi=130), plt.close(fig)


def _backtest(folds, path):
    x = np.arange(len(folds))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(x - 0.2, [f["baseline_WAPE_pct"] for f in folds], 0.4, label="baseline (occupancy prior)")
    ax.bar(x + 0.2, [f["model_WAPE_pct"] for f in folds], 0.4, label="model")
    ax.set_xticks(x, [f["test_window"].replace(" -> ", "\n") for f in folds], fontsize=7)
    ax.set(ylabel="WAPE %", title="Rolling-origin backtest")
    ax.legend(), fig.tight_layout(), fig.savefig(path, dpi=130), plt.close(fig)
