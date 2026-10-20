"""REST API for Backlog Exchange.

    uvicorn api:app --reload
    open http://127.0.0.1:8000/docs     (interactive API docs)
    open http://127.0.0.1:8000/         (web demo)
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from exchange.market import Feature, TradeError
from exchange.service import ExchangeService

app = FastAPI(title="Backlog Exchange",
              description="Prediction markets for product backlogs: the team bets on which "
                          "features will hit their success metric, and the backlog reprioritizes.")
service = ExchangeService()


class TraderIn(BaseModel):
    name: str
    role: str = "member"
    balance: float = Field(1000.0, gt=0)


class FeatureIn(BaseModel):
    id: str
    title: str
    success_metric: str
    value: float = Field(gt=0, description="business value (EUR) if the feature succeeds")
    effort: int = Field(gt=0, description="story points")
    owner: str = ""
    tags: list[str] = []
    prior: float = Field(0.5, gt=0, lt=1)


class BuyIn(BaseModel):
    trader: str
    side: str = Field(pattern="^(YES|NO|yes|no)$")
    budget: float = Field(gt=0)


class ForecastIn(BaseModel):
    trader: str
    probability: float = Field(gt=0, lt=1)
    max_spend: float | None = Field(None, gt=0)


class ResolveIn(BaseModel):
    success: bool


def _guard(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except KeyError as e:
        raise HTTPException(404, f"not found: {e}")
    except TradeError as e:
        raise HTTPException(400, str(e))


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(Path(__file__).parent / "web" / "index.html")


@app.get("/markets")
def list_markets(status: str | None = None):
    return service.list_markets(status)


@app.get("/markets/{market_id}")
def get_market(market_id: str):
    return _guard(service.market_view, market_id)


@app.post("/markets", status_code=201)
def create_market(body: FeatureIn):
    data = body.model_dump()
    prior = data.pop("prior")
    return _guard(service.list_feature, Feature(**data), prior)


@app.post("/markets/{market_id}/buy")
def buy(market_id: str, body: BuyIn):
    return _guard(service.buy, body.trader, market_id, body.side, body.budget)


@app.post("/markets/{market_id}/forecast")
def forecast(market_id: str, body: ForecastIn):
    return _guard(service.forecast, body.trader, market_id, body.probability, body.max_spend)


@app.post("/markets/{market_id}/resolve")
def resolve(market_id: str, body: ResolveIn):
    return _guard(service.resolve, market_id, body.success)


@app.post("/traders", status_code=201)
def add_trader(body: TraderIn):
    return _guard(service.add_trader, body.name, body.role, body.balance)


@app.get("/leaderboard")
def leaderboard():
    return service.leaderboard()


@app.get("/backlog")
def backlog(capacity: int = 20):
    return service.backlog(capacity)


@app.get("/report")
def report():
    return service.report()
