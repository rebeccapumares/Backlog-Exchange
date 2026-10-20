"""Agile layer: turn market probabilities into backlog priorities and sprint plans.

Classic WSJF (Weighted Shortest Job First) ranks work by cost of delay divided
by job size, but the "value" in it is usually a guess from one person. Here
value is *risk-adjusted* by the market:

    expected value   = P(success) * value_if_success
    priority score   = expected value / effort          (value per story point)

The sprint planner then picks the set of features that maximizes total
expected value within the team's capacity (a 0/1 knapsack, solved exactly
with dynamic programming).
"""
from __future__ import annotations

from dataclasses import dataclass

from .market import Exchange, Feature


@dataclass
class RankedFeature:
    id: str
    title: str
    probability: float
    value: float
    effort: int
    expected_value: float
    score: float          # expected value per story point


def rank_features(features: list[Feature], probabilities: dict[str, float]) -> list[RankedFeature]:
    rows = []
    for f in features:
        p = probabilities[f.id]
        ev = p * f.value
        rows.append(RankedFeature(f.id, f.title, round(p, 4), f.value, f.effort,
                                  round(ev, 2), round(ev / max(1, f.effort), 2)))
    return sorted(rows, key=lambda r: -r.score)


def rank_backlog(exchange: Exchange, include_resolved: bool = False) -> list[RankedFeature]:
    markets = [m for m in exchange.markets.values()
               if include_resolved or m.status == "open"]
    return rank_features([m.feature for m in markets], {m.id: m.yes_price for m in markets})


def plan_sprint(ranked: list[RankedFeature], capacity: int) -> dict:
    """Choose features maximizing total expected value with total effort <= capacity."""
    n = len(ranked)
    # best[i][c] = max expected value using the first i features within capacity c
    best = [[0.0] * (capacity + 1) for _ in range(n + 1)]
    for i, f in enumerate(ranked, start=1):
        for c in range(capacity + 1):
            best[i][c] = best[i - 1][c]
            if f.effort <= c:
                best[i][c] = max(best[i][c], best[i - 1][c - f.effort] + f.expected_value)
    chosen, c = [], capacity
    for i in range(n, 0, -1):                     # walk back to recover the choice
        if best[i][c] != best[i - 1][c]:
            chosen.append(ranked[i - 1])
            c -= ranked[i - 1].effort
    chosen.reverse()
    return {
        "capacity": capacity,
        "points_used": sum(f.effort for f in chosen),
        "expected_value": round(sum(f.expected_value for f in chosen), 2),
        "features": [f.id for f in chosen],
    }


def greedy_plan(order: list[RankedFeature], capacity: int) -> dict:
    """What a team does without optimization: take items in order until full."""
    chosen, used = [], 0
    for f in order:
        if used + f.effort <= capacity:
            chosen.append(f)
            used += f.effort
    return {"capacity": capacity, "points_used": used,
            "expected_value": round(sum(f.expected_value for f in chosen), 2),
            "features": [f.id for f in chosen]}
