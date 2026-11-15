import numpy as np
import pandas as pd

from src.cinema_demand.data import LEAKY, chronological_split, clean


def _raw():
    return pd.DataFrame({
        "film_code": [1, 1, 1, 2, 2], "cinema_code": [9, 9, 9, 9, 9],
        "show_time": [1, 1, 2, 1, 2], "date": ["2018-01-01"] * 3 + ["2018-01-02"] * 2,
        "tickets_sold": [10, 5, 300, 20, 4], "capacity": [100.0, 120.0, 200.0, 0.0, 100.0],
        "total_sales": [1.0] * 5, "tickets_out": [0] * 5, "ticket_use": [1] * 5,
        "occu_perc": [1.0] * 5, "ticket_price": [1.0] * 5,
    })


def test_clean_merges_duplicates_caps_and_imputes():
    df, rep = clean(_raw())
    assert not (set(LEAKY) & set(df.columns))
    assert rep["duplicate_key_rows_merged"] == 2 and rep["clean_rows"] == 4
    merged = df[(df.film_code == 1) & (df.show_time == 1)].iloc[0]
    assert merged.tickets_sold == 15 and merged.capacity == 120.0
    over = df[(df.film_code == 1) & (df.show_time == 2)].iloc[0]
    assert over.tickets_sold == 200.0 and over.censored == 1          # capped at capacity
    imputed = df[(df.film_code == 2) & (df.show_time == 1)].iloc[0]
    assert imputed.capacity_imputed == 1 and imputed.capacity > 0
    assert (df["occupancy"] <= 1.0).all()


def test_split_is_strictly_chronological(synth):
    tr, va, te = chronological_split(synth, "2018-04-15", "2018-05-15")
    assert tr["date"].max() < va["date"].min() <= va["date"].max() < te["date"].min()
    assert len(tr) + len(va) + len(te) == len(synth)
