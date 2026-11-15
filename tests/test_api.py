import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from src.cinema_demand.data import chronological_split
from src.cinema_demand.features import ALL_FEATURES, build_features
from src.cinema_demand.models import DemandModel
from src.cinema_demand.predict import Forecaster


@pytest.fixture(scope="module")
def client(synth):
    f = build_features(synth)
    tr, va, _ = chronological_split(f, "2018-04-20", "2018-05-10")
    m = DemandModel.fit(tr, va, ALL_FEATURES, alpha=0.2, seed=0)
    cfg = {"forecast": {"lag_days": 2}, "staffing": {"tickets_per_staff": 100, "min_staff": 2}}
    D = pd.Timestamp("2018-05-20")
    fc = Forecaster(m, synth[synth["date"] < D], cfg)
    return TestClient(create_app(fc)), synth, D


def test_health_and_forecast(client):
    c, synth, D = client
    assert c.get("/health").json()["model_loaded"] is True
    day = synth[synth["date"] == D].head(12)
    body = {"shows": [{"date": str(D.date()), "cinema_code": int(r.cinema_code), "film_code": int(r.film_code),
                       "show_time": int(r.show_time)} for r in day.itertuples()], "explain": True}
    r = c.post("/forecast", json=body)
    assert r.status_code == 200
    out = r.json()["forecasts"]
    assert len(out) == 12 and all(x["lo"] <= x["pred"] <= x["hi"] for x in out)
    assert all(len(x["top_drivers"]) == 5 for x in out)


def test_validation_errors(client):
    c, *_ = client
    assert c.post("/forecast", json={"shows": []}).status_code == 422
    bad = {"shows": [{"date": "2018-05-20", "cinema_code": 1000, "film_code": 500, "show_time": 0}]}
    assert c.post("/forecast", json=bad).status_code == 422
