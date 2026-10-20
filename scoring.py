"""Forecast scoring: how good were the market and each trader?"""
from __future__ import annotations

import math

from .market import YES, Exchange


def brier(probs: list[float], outcomes: list[int]) -> float:
    """Mean squared error of probabilities vs 0/1 outcomes (0 = perfect, 0.25 = coin flip)."""
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / len(probs)


def log_loss(probs: list[float], outcomes: list[int], eps: float = 1e-6) -> float:
    total = 0.0
    for p, o in zip(probs, outcomes):
        p = min(max(p, eps), 1 - eps)
        total -= o * math.log(p) + (1 - o) * math.log(1 - p)
    return total / len(probs)


def calibration(probs: list[float], outcomes: list[int], bins: int = 5) -> list[dict]:
    """Group forecasts into probability bins and compare with observed success rates."""
    rows = []
    for k in range(bins):
        lo, hi = k / bins, (k + 1) / bins
        idx = [i for i, p in enumerate(probs) if lo <= p < hi or (k == bins - 1 and p == 1.0)]
        if idx:
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(idx),
                         "mean_forecast": round(sum(probs[i] for i in idx) / len(idx), 3),
                         "observed_rate": round(sum(outcomes[i] for i in idx) / len(idx), 3)})
    return rows


def market_report(exchange: Exchange) -> dict:
    """Accuracy of final market prices on resolved features."""
    resolved = [m for m in exchange.markets.values() if m.status == "resolved"]
    if not resolved:
        return {"resolved": 0}
    probs = [m.yes_price for m in resolved]
    outcomes = [1 if m.resolution == YES else 0 for m in resolved]
    opening = [m.opening_price for m in resolved]
    return {
        "resolved": len(resolved),
        "brier_final": round(brier(probs, outcomes), 4),
        "brier_opening": round(brier(opening, outcomes), 4),
        "log_loss_final": round(log_loss(probs, outcomes), 4),
        "calibration": calibration(probs, outcomes),
    }


def trader_accuracy(exchange: Exchange) -> list[dict]:
    """Per trader: realized profit on resolved markets and how often their net
    position pointed the right way."""
    stats = {name: {"trader": name, "markets": 0, "correct": 0, "profit": 0.0}
             for name in exchange.traders}
    for m in exchange.markets.values():
        if m.status != "resolved":
            continue
        spent: dict[str, float] = {}
        for tr in m.history:
            spent[tr.trader] = spent.get(tr.trader, 0.0) + tr.cost
        for name, (yes, no) in m.positions.items():
            if abs(yes - no) < 1e-9 and name not in spent:
                continue
            s = stats[name]
            s["markets"] += 1
            payout = yes if m.resolution == YES else no
            s["profit"] += payout - spent.get(name, 0.0)
            leaned_yes = yes > no
            s["correct"] += int(leaned_yes == (m.resolution == YES))
    rows = []
    for s in stats.values():
        if s["markets"]:
            s["hit_rate"] = round(s["correct"] / s["markets"], 3)
            s["profit"] = round(s["profit"], 2)
            rows.append(s)
    return sorted(rows, key=lambda r: -r["profit"])
