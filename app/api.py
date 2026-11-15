"""FastAPI service.   uvicorn app.api:app --port 8000   (docs at /docs)

The service needs the trained model (artifacts/model.joblib) and the cleaned history
(data/processed/shows.parquet) - both produced by `python -m src.cinema_demand.train`.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import date as Date

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.cinema_demand.config import ROOT, load_config
from src.cinema_demand.models import DemandModel
from src.cinema_demand.predict import Forecaster


class Show(BaseModel):
    date: Date
    cinema_code: int
    film_code: int
    show_time: int = Field(ge=1, description="daily slot index")
    capacity: float | None = Field(default=None, gt=0, description="hall capacity; inferred from history if omitted")


class ForecastRequest(BaseModel):
    shows: list[Show] = Field(min_length=1, max_length=20000,
                              description="the FULL published programme for the target date(s) across cinemas")
    explain: bool = False


def load_forecaster() -> Forecaster:
    cfg = load_config()
    model = DemandModel.load(ROOT / "artifacts" / "model.joblib")
    history = pd.read_parquet(ROOT / cfg["data"]["processed_path"])
    return Forecaster(model, history, cfg)


def create_app(forecaster: Forecaster | None = None) -> FastAPI:
    state = {"fc": forecaster}

    @asynccontextmanager
    async def lifespan(app):
        if state["fc"] is None:
            try:
                state["fc"] = load_forecaster()
            except FileNotFoundError:
                state["fc"] = None
        yield

    app = FastAPI(title="Cinema Demand Forecast API", version="2.0.0", lifespan=lifespan,
                  description="Decision-support forecasts of tickets sold per show, 48h ahead. "
                              "Human review required before changing screens or rosters.")

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": state["fc"] is not None}

    @app.get("/metrics")
    def metrics():
        p = ROOT / "reports" / "metrics.json"
        if not p.exists():
            raise HTTPException(404, "metrics.json not found - run training first")
        return json.loads(p.read_text())

    @app.post("/forecast")
    def forecast(req: ForecastRequest):
        fc = state["fc"]
        if fc is None:
            raise HTTPException(503, "model/history not loaded - run `python -m src.cinema_demand.train`")
        sched = pd.DataFrame([s.model_dump() for s in req.shows])
        try:
            out = fc.forecast(sched, explain=req.explain)
        except ValueError as e:
            raise HTTPException(422, str(e))
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        recs = out.round(3).to_dict("records")
        for r in recs:
            r["disclaimer"] = "decision support only"
        return {"n": len(recs), "interval_nominal_coverage": 1 - fc.model.alpha, "forecasts": recs}

    return app


app = create_app()
