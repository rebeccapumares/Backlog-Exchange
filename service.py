"""Service layer shared by the REST API and the copilot (MCP) server.

Keeps one Exchange in memory and persists it to a JSON file after every change.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from .backlog import greedy_plan, plan_sprint, rank_backlog
from .demo import build_demo
from .market import NO, OUTCOME_NAMES, YES, Exchange, Feature, TradeError
from .scoring import market_report, trader_accuracy

DEFAULT_STATE = Path(os.environ.get("BACKLOG_EXCHANGE_STATE", "data/exchange_state.json"))


class ExchangeService:
    def __init__(self, state_path: Path | str = DEFAULT_STATE, demo_if_empty: bool = True):
        self.path = Path(state_path)
        self.lock = threading.Lock()
        if self.path.exists():
            self.ex = Exchange.from_dict(json.loads(self.path.read_text()))
        else:
            self.ex = build_demo() if demo_if_empty else Exchange()
            self._save()

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.ex.to_dict(), indent=1))
        tmp.replace(self.path)

    # ----------------------------------------------------------- reads
    def market_view(self, market_id: str) -> dict:
        m = self.ex.markets[market_id]
        f = m.feature
        return {
            "id": f.id, "title": f.title, "success_metric": f.success_metric,
            "value": f.value, "effort": f.effort, "owner": f.owner, "tags": f.tags,
            "status": m.status,
            "resolution": None if m.resolution is None else OUTCOME_NAMES[m.resolution],
            "yes_price": round(m.yes_price, 4),
            "consensus": round(self.ex.consensus(market_id), 4),
            "n_trades": len(m.history),
            "price_history": [round(p, 4) for p in self.ex.price_history(market_id)],
        }

    def list_markets(self, status: str | None = None) -> list[dict]:
        return [self.market_view(mid) for mid, m in self.ex.markets.items()
                if status is None or m.status == status]

    def backlog(self, capacity: int = 20) -> dict:
        ranked = rank_backlog(self.ex)
        optimal = plan_sprint(ranked, capacity)
        by_value = sorted(ranked, key=lambda r: -r.value)       # the "gut feel" order
        return {
            "ranking": [r.__dict__ for r in ranked],
            "sprint_plan": optimal,
            "naive_plan_by_raw_value": greedy_plan(by_value, capacity),
        }

    def leaderboard(self) -> list[dict]:
        return self.ex.leaderboard()

    def report(self) -> dict:
        return {"market": market_report(self.ex), "traders": trader_accuracy(self.ex)}

    # ---------------------------------------------------------- writes
    def add_trader(self, name: str, role: str = "member", balance: float = 1000.0) -> dict:
        with self.lock:
            t = self.ex.add_trader(name, balance, role)
            self._save()
            return t.__dict__

    def list_feature(self, feature: Feature, prior: float = 0.5) -> dict:
        with self.lock:
            self.ex.list_feature(feature, prior)
            self._save()
            return self.market_view(feature.id)

    def buy(self, trader: str, market_id: str, side: str, budget: float) -> dict:
        outcome = {"YES": YES, "NO": NO}[side.upper()]
        with self.lock:
            tr = self.ex.buy(trader, market_id, outcome, budget)
            self._save()
            return {"trade": tr.__dict__, "market": self.market_view(market_id)}

    def forecast(self, trader: str, market_id: str, probability: float,
                 max_spend: float | None = None) -> dict:
        """State a probability and trade the market toward it."""
        with self.lock:
            tr = self.ex.trade_to_belief(trader, market_id, probability, max_spend)
            self._save()
            return {"trade": None if tr is None else tr.__dict__,
                    "market": self.market_view(market_id)}

    def resolve(self, market_id: str, success: bool) -> dict:
        with self.lock:
            payouts = self.ex.resolve(market_id, success)
            self._save()
            return {"market": self.market_view(market_id),
                    "payouts": {k: round(v, 2) for k, v in payouts.items()}}

    def reset_demo(self) -> None:
        with self.lock:
            self.ex = build_demo()
            self._save()


__all__ = ["ExchangeService", "TradeError"]
