"""Tests for Backlog Exchange. Run with: python -m pytest tests -q"""
import itertools
import math
import os

import pytest

from exchange.backlog import RankedFeature, plan_sprint, rank_features
from exchange.lmsr import LMSR
from exchange.market import NO, YES, Exchange, Feature, TradeError
from exchange.scoring import brier, market_report, trader_accuracy
from exchange.simulate import SimConfig, run, summarize


def feat(fid="F1", value=10000, effort=3):
    return Feature(fid, f"Feature {fid}", "metric", value, effort)


# ------------------------------------------------------------------ LMSR
def test_prices_sum_to_one_and_start_even():
    m = LMSR(3, 50)
    assert all(math.isclose(p, 1 / 3) for p in m.prices())
    m.execute(1, 40)
    assert math.isclose(sum(m.prices()), 1.0)
    assert m.price(1) > 1 / 3


def test_budget_buys_exactly_that_much():
    m = LMSR(2, 100)
    shares = m.shares_for_budget(YES, 37.5)
    assert math.isclose(m.quote_shares(YES, shares), 37.5, rel_tol=1e-9)


def test_shares_to_reach_target_price():
    m = LMSR(2, 100)
    m.execute(YES, m.shares_to_reach(YES, 0.83))
    assert math.isclose(m.price(YES), 0.83, rel_tol=1e-9)
    m.execute(YES, m.shares_to_reach(YES, 0.2))     # selling back down
    assert math.isclose(m.price(YES), 0.2, rel_tol=1e-9)


def test_market_maker_loss_is_bounded_by_b_ln_n():
    m = LMSR(2, 100)
    collected = m.execute(YES, 5000)                 # someone buys a huge YES position
    payout = 5000                                    # ...and YES wins
    assert payout - collected <= m.max_loss() + 1e-9


# -------------------------------------------------------------- exchange
def make_exchange():
    ex = Exchange(liquidity=100)
    ex.add_trader("ana")
    ex.add_trader("ben")
    ex.list_feature(feat())
    return ex


def test_insufficient_funds_and_unknown_trader_rejected():
    ex = make_exchange()
    with pytest.raises(TradeError):
        ex.buy("ana", "F1", YES, 5000)
    with pytest.raises(TradeError):
        ex.buy("zoe", "F1", YES, 10)


def test_buy_then_sell_round_trip_costs_nothing():
    ex = make_exchange()
    t = ex.buy("ana", "F1", YES, 50)
    ex.sell("ana", "F1", YES, t.shares)
    assert math.isclose(ex.traders["ana"].balance, 1000.0, abs_tol=1e-9)
    assert math.isclose(ex.markets["F1"].yes_price, 0.5, abs_tol=1e-9)


def test_resolution_pays_winners_and_conserves_value():
    ex = make_exchange()
    ex.trade_to_belief("ana", "F1", 0.8)
    ex.trade_to_belief("ben", "F1", 0.3)
    m = ex.markets["F1"]
    spent = sum(tr.cost for tr in m.history)
    yes_out = sum(p[YES] for p in m.positions.values())
    ex.resolve("F1", success=True)
    total = sum(t.balance for t in ex.traders.values())
    assert math.isclose(total, 2000 - spent + yes_out)
    assert m.status == "resolved"
    with pytest.raises(TradeError):
        ex.buy("ana", "F1", YES, 1)                  # closed market


def test_trade_to_belief_respects_spend_cap():
    ex = make_exchange()
    t = ex.trade_to_belief("ana", "F1", 0.95, max_spend=5)
    assert math.isclose(t.cost, 5.0)
    assert ex.markets["F1"].yes_price < 0.95


def test_consensus_weights_by_wealth():
    ex = make_exchange()
    ex.traders["ana"].balance = 3000
    ex.submit_forecast("ana", "F1", 0.8)
    ex.submit_forecast("ben", "F1", 0.2)
    assert math.isclose(ex.consensus("F1"), (3000 * 0.8 + 1000 * 0.2) / 4000)


def test_persistence_round_trip():
    ex = make_exchange()
    ex.trade_to_belief("ana", "F1", 0.7)
    ex2 = Exchange.from_dict(ex.to_dict())
    assert math.isclose(ex2.markets["F1"].yes_price, ex.markets["F1"].yes_price)
    assert ex2.markets["F1"].beliefs == ex.markets["F1"].beliefs
    assert ex2.traders["ana"].balance == ex.traders["ana"].balance


# ----------------------------------------------------------------- agile
def test_knapsack_matches_brute_force():
    feats = [feat(f"F{i}", value=v, effort=e) for i, (v, e) in
             enumerate([(50000, 8), (30000, 3), (42000, 5), (12000, 2), (60000, 13), (25000, 5)])]
    probs = {f.id: p for f, p in zip(feats, [0.6, 0.7, 0.4, 0.9, 0.5, 0.65])}
    ranked = rank_features(feats, probs)
    plan = plan_sprint(ranked, 15)
    best = max((sum(r.expected_value for r in combo)
                for k in range(len(ranked) + 1) for combo in itertools.combinations(ranked, k)
                if sum(r.effort for r in combo) <= 15))
    assert math.isclose(plan["expected_value"], round(best, 2))
    assert plan["points_used"] <= 15


def test_ranking_uses_expected_value_per_point():
    feats = [feat("A", 100000, 10), feat("B", 20000, 1)]
    ranked = rank_features(feats, {"A": 0.5, "B": 0.5})
    assert [r.id for r in ranked] == ["B", "A"]      # 10k/pt beats 5k/pt


# ---------------------------------------------------------------- scoring
def test_brier_and_reports():
    assert brier([1.0, 0.0], [1, 0]) == 0.0
    assert brier([0.5, 0.5], [1, 0]) == 0.25
    ex = make_exchange()
    ex.trade_to_belief("ana", "F1", 0.8)
    ex.trade_to_belief("ben", "F1", 0.35, max_spend=10)
    ex.resolve("F1", True)
    rep = market_report(ex)
    assert rep["resolved"] == 1 and rep["brier_final"] < rep["brier_opening"]
    acc = {r["trader"]: r for r in trader_accuracy(ex)}
    assert acc["ana"]["profit"] > 0 and acc["ana"]["hit_rate"] == 1.0
    assert acc["ben"]["hit_rate"] == 0.0


# ------------------------------------------------------------- simulation
def test_simulation_orders_methods_sensibly():
    runs = [run(seed, SimConfig(sprints=12, features_per_sprint=8)) for seed in range(4)]
    s = summarize(runs, warmup=2)
    assert s["HiPPO"]["error_vs_truth"] > s["Vote"]["error_vs_truth"]
    assert s["Ideal"]["error_vs_truth"] <= s["Vote"]["error_vs_truth"]
    assert s["Meeting"]["error_vs_truth"] > s["Vote"]["error_vs_truth"]


# -------------------------------------------------------------------- API
def test_api_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKLOG_EXCHANGE_STATE", str(tmp_path / "state.json"))
    import importlib

    import api
    import exchange.service as service_mod
    importlib.reload(service_mod)
    importlib.reload(api)
    from fastapi.testclient import TestClient
    c = TestClient(api.app)
    assert len(c.get("/markets").json()) == 8
    r = c.post("/markets", json={"id": "NEW-1", "title": "Widgets", "success_metric": "x",
                                 "value": 1000, "effort": 2})
    assert r.status_code == 201
    r = c.post("/markets/NEW-1/forecast", json={"trader": "Sam", "probability": 0.8})
    assert r.json()["market"]["yes_price"] > 0.5
    assert c.post("/markets/NEW-1/buy", json={"trader": "Nobody", "side": "YES",
                                              "budget": 5}).status_code == 400
    assert c.get("/markets/NOPE").status_code == 404
    assert c.post("/markets/NEW-1/resolve", json={"success": True}).status_code == 200
    plan = c.get("/backlog?capacity=20").json()["sprint_plan"]
    assert plan["points_used"] <= 20
    assert os.path.exists(tmp_path / "state.json")
