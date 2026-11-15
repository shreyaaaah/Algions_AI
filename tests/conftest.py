import numpy as np
import pandas as pd
import pytest

from src.cinema_demand import models as M


@pytest.fixture(scope="session")
def synth():
    """Small synthetic chain: 8 cinemas x 6 films x 3 slots over 90 days, with a weekly pattern."""
    rng = np.random.default_rng(0)
    days = pd.date_range("2018-03-01", periods=90)
    rows = []
    for d in days:
        for c in range(8):
            for f in range(6):
                if rng.random() < 0.35:
                    continue
                for s in (1, 2, 3):
                    cap = 100.0 * (1 + c % 3) + 50
                    base = 0.15 + 0.25 * (d.weekday() >= 4) + 0.05 * f + 0.03 * s
                    occ = np.clip(rng.normal(base, 0.05), 0.01, 1.0)
                    rows.append((d, 1000 + c, 500 + f, s, np.round(occ * cap), cap))
    df = pd.DataFrame(rows, columns=["date", "cinema_code", "film_code", "show_time", "tickets_sold", "capacity"])
    df["occupancy"] = df["tickets_sold"] / df["capacity"]
    return df


@pytest.fixture(autouse=True)
def fast_lgb(monkeypatch):
    monkeypatch.setitem(M.BASE_PARAMS, "num_leaves", 7)
    monkeypatch.setitem(M.BASE_PARAMS, "learning_rate", 0.2)
    monkeypatch.setitem(M.BASE_PARAMS, "min_child_samples", 10)
