"""Operations dashboard.   streamlit run app/dashboard.py

Tab 1 replays any date in the data as a live T-48h forecast (only history up to D-2 is used)
and compares it with what actually happened. Tab 2 forecasts an uploaded programme.
Tab 3 shows the model report. Decision support only - human review required.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.cinema_demand.config import load_config
from src.cinema_demand.models import DemandModel
from src.cinema_demand.predict import Forecaster

st.set_page_config(page_title="Cinema demand forecast", layout="wide")


@st.cache_resource
def load():
    cfg = load_config()
    model = DemandModel.load(ROOT / "artifacts" / "model.joblib")
    hist = pd.read_parquet(ROOT / cfg["data"]["processed_path"])
    return Forecaster(model, hist, cfg), hist, cfg


try:
    fc, hist, cfg = load()
except FileNotFoundError:
    st.error("Model or processed data not found. Run `python -m src.cinema_demand.train` first.")
    st.stop()

st.title("Cinema ticket demand - 48h-ahead forecast")
st.caption("Decision support only: a manager reviews every screen change and roster. "
           f"Intervals target {int(100 * (1 - fc.model.alpha))}% nominal coverage.")
tab1, tab2, tab3 = st.tabs(["Replay a day", "Forecast a programme", "Model report"])

with tab1:
    oos = pd.Timestamp(fc.model.meta["oos_start"])
    dates = sorted(d for d in hist["date"].unique() if d > oos)
    day = st.selectbox("Show date", dates, index=len(dates) - 1, format_func=lambda d: str(pd.Timestamp(d).date()))
    actual = hist[hist["date"] == day]
    out = fc.forecast(actual[["date", "cinema_code", "film_code", "show_time", "capacity"]], explain=False)
    out = out.merge(actual[["cinema_code", "film_code", "show_time", "tickets_sold"]],
                    on=["cinema_code", "film_code", "show_time"], how="left")
    out["abs_err"] = (out["pred"] - out["tickets_sold"]).abs()
    c = st.columns(4)
    c[0].metric("Shows", f"{len(out):,}")
    c[1].metric("Forecast tickets", f"{out['pred'].sum():,.0f}")
    c[2].metric("Actual tickets", f"{out['tickets_sold'].sum():,.0f}")
    c[3].metric("Day WAPE", f"{100 * out['abs_err'].sum() / out['tickets_sold'].sum():.1f}%")
    cov = ((out["tickets_sold"] >= out["lo"]) & (out["tickets_sold"] <= out["hi"])).mean()
    st.write(f"Share of shows inside the forecast interval: **{100 * cov:.1f}%**; "
             f"cold-start films flagged: **{int(out['cold_start_film'].sum())}** shows.")
    cin = st.selectbox("Cinema drill-down", sorted(out["cinema_code"].unique()))
    sub = out[out["cinema_code"] == cin].sort_values(["show_time", "film_code"])
    st.dataframe(sub[["film_code", "show_time", "capacity", "pred", "lo", "hi", "tickets_sold",
                      "staff_recommended", "cold_start_film"]].round(1), use_container_width=True)
    st.bar_chart(out.groupby("cinema_code")[["pred", "tickets_sold"]].sum().nlargest(25, "tickets_sold"))

with tab2:
    st.write("Upload the **full programme** for the target date(s) (all cinemas): columns "
             "`date, cinema_code, film_code, show_time` and optional `capacity`.")
    up = st.file_uploader("Programme CSV", type="csv")
    if up:
        sched = pd.read_csv(up)
        try:
            res = fc.forecast(sched, explain=True)
            res["top_drivers"] = res["top_drivers"].map(lambda d: ", ".join(f"{k} ({v:+.0f})" for k, v in d[:3]))
            st.dataframe(res.round(1), use_container_width=True)
            st.download_button("Download forecasts", res.to_csv(index=False), "forecasts.csv")
        except ValueError as e:
            st.error(str(e))

with tab3:
    mp = ROOT / "reports" / "metrics.json"
    if mp.exists():
        m = json.loads(mp.read_text())
        st.subheader("Held-out test")
        st.json(m["final_model"]["point"])
        st.json(m["final_model"]["interval"])
        if "ablation_single_l1_model" in m:
            st.subheader("Feature-group ablation")
            st.dataframe(pd.DataFrame(m["ablation_single_l1_model"]))
        if "backtest" in m:
            st.subheader("Rolling-origin backtest")
            st.dataframe(pd.DataFrame(m["backtest"]))
    for img in sorted((ROOT / "reports" / "figures").glob("*.png")):
        st.image(str(img), caption=img.stem)
