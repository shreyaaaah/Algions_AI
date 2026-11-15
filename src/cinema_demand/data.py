"""Loading, auditing and cleaning the Kaggle 'Cinema Tickets' file.

Raw columns: film_code, cinema_code, total_sales, tickets_sold, tickets_out,
show_time, occu_perc, ticket_price, ticket_use, capacity, date, month, quarter, day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["cinema_code", "film_code", "date", "show_time"]
# Columns that are outcomes of the show (or algebraic functions of the target)
# and therefore can never be used as model inputs at T-48h.
LEAKY = ["total_sales", "tickets_out", "ticket_use", "occu_perc", "ticket_price"]
KEEP = ["date", "cinema_code", "film_code", "show_time", "tickets_sold", "capacity"]


def load_raw(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "tickets_sold" not in df.columns and "tickets_num" in df.columns:
        df = df.rename(columns={"tickets_num": "tickets_sold"})
    return df


def audit_leakage(raw: pd.DataFrame) -> dict:
    """Quantify why each post-show column is excluded (reported, not assumed)."""
    out = {}
    if {"occu_perc", "tickets_sold", "capacity"} <= set(raw.columns):
        ok = raw["capacity"] > 0
        ratio = raw.loc[ok, "tickets_sold"] / raw.loc[ok, "capacity"] * 100
        out["occu_perc_vs_tickets_over_capacity"] = {
            "corr": float(np.corrcoef(ratio, raw.loc[ok, "occu_perc"])[0, 1]),
            "mean_abs_diff": float((ratio - raw.loc[ok, "occu_perc"]).abs().mean()),
        }
    if {"ticket_price", "total_sales", "tickets_sold"} <= set(raw.columns):
        ok = raw["tickets_sold"] > 0
        ratio = raw.loc[ok, "total_sales"] / raw.loc[ok, "tickets_sold"]
        out["ticket_price_vs_sales_over_tickets"] = {
            "corr": float(np.corrcoef(ratio, raw.loc[ok, "ticket_price"])[0, 1]),
            "mean_abs_diff": float((ratio - raw.loc[ok, "ticket_price"]).abs().mean()),
        }
    out["dropped_columns"] = [c for c in LEAKY if c in raw.columns]
    return out


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return one row per (cinema, film, date, slot) plus a cleaning report."""
    rep: dict = {"raw_rows": int(len(raw))}
    df = raw[[c for c in KEEP if c in raw.columns]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["date", "cinema_code", "film_code", "show_time", "tickets_sold"])
    df["tickets_sold"] = df["tickets_sold"].astype(float).clip(lower=0)

    # 1) duplicate show keys -> sum tickets, max capacity
    dup = int(df.duplicated(KEY, keep=False).sum())
    df = (df.groupby(KEY, as_index=False)
            .agg(tickets_sold=("tickets_sold", "sum"), capacity=("capacity", "max")))
    rep["duplicate_key_rows_merged"] = dup

    # 2) invalid capacity -> median valid capacity of the cinema (global fallback)
    bad = ~(df["capacity"] > 0)
    rep["capacity_invalid_rows"] = int(bad.sum())
    good = df.loc[~bad]
    med = good.groupby("cinema_code")["capacity"].median()
    fill = df["cinema_code"].map(med).fillna(good["capacity"].median())
    df["capacity_imputed"] = bad.astype(int)
    df.loc[bad, "capacity"] = fill[bad]

    # 3) physical bound: cannot sell more than capacity (these shows are censored demand)
    over = df["tickets_sold"] > df["capacity"]
    rep["tickets_over_capacity_rows"] = int(over.sum())
    df.loc[over, "tickets_sold"] = df.loc[over, "capacity"]
    df["censored"] = (df["tickets_sold"] >= df["capacity"] - 1e-9).astype(int)
    rep["censored_at_capacity_rows"] = int(df["censored"].sum())

    df["occupancy"] = df["tickets_sold"] / df["capacity"]
    df = df.sort_values(["date", "cinema_code", "film_code", "show_time"]).reset_index(drop=True)
    rep["clean_rows"] = int(len(df))
    rep["date_min"], rep["date_max"] = str(df["date"].min().date()), str(df["date"].max().date())
    rep["n_films"], rep["n_cinemas"] = int(df["film_code"].nunique()), int(df["cinema_code"].nunique())
    return df, rep


def chronological_split(df: pd.DataFrame, train_end: str, val_end: str):
    """Strictly time-ordered split (no shuffling)."""
    te, ve = pd.Timestamp(train_end), pd.Timestamp(val_end)
    train = df[df["date"] <= te]
    val = df[(df["date"] > te) & (df["date"] <= ve)]
    test = df[df["date"] > ve]
    return train, val, test
