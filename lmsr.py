"""Logarithmic Market Scoring Rule (LMSR) automated market maker.

LMSR (Hanson, 2003) lets a market run with very few traders: an automated
market maker always quotes a price and takes the other side of every trade.

For outcomes i = 1..n with outstanding shares q_i and liquidity parameter b:

    cost function   C(q)   = b * ln( sum_i exp(q_i / b) )
    price           p_i(q) = exp(q_i / b) / sum_j exp(q_j / b)     (a softmax)
    trade cost      C(q + delta) - C(q)

Prices always sum to 1, so they read directly as probabilities. Each share
of the winning outcome pays 1 token. The market maker's worst-case loss is
bounded by b * ln(n), which is the subsidy that pays for the information.
"""
from __future__ import annotations

import math


def _logsumexp(xs: list[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


class LMSR:
    def __init__(self, n_outcomes: int = 2, liquidity: float = 100.0,
                 shares: list[float] | None = None):
        if n_outcomes < 2:
            raise ValueError("a market needs at least 2 outcomes")
        if liquidity <= 0:
            raise ValueError("liquidity b must be positive")
        self.n = n_outcomes
        self.b = float(liquidity)
        self.q = list(shares) if shares is not None else [0.0] * n_outcomes

    # ------------------------------------------------------------- core
    def cost(self, q: list[float] | None = None) -> float:
        q = self.q if q is None else q
        return self.b * _logsumexp([x / self.b for x in q])

    def prices(self) -> list[float]:
        z = [x / self.b for x in self.q]
        m = max(z)
        e = [math.exp(x - m) for x in z]
        s = sum(e)
        return [x / s for x in e]

    def price(self, outcome: int) -> float:
        return self.prices()[outcome]

    def max_loss(self) -> float:
        """Worst-case subsidy the market maker can lose."""
        return self.b * math.log(self.n)

    # ----------------------------------------------------------- quotes
    def quote_shares(self, outcome: int, shares: float) -> float:
        """Tokens it costs to buy `shares` of `outcome` (negative = sell, returns proceeds as negative)."""
        q2 = list(self.q)
        q2[outcome] += shares
        return self.cost(q2) - self.cost()

    def shares_for_budget(self, outcome: int, budget: float) -> float:
        """How many shares of `outcome` exactly `budget` tokens buys (closed form)."""
        if budget <= 0:
            return 0.0
        z = [x / self.b for x in self.q]
        m = max(z)
        total = sum(math.exp(x - m) for x in z)
        others = total - math.exp(z[outcome] - m)
        # solve b*ln(others + e^{(q_i+d)/b}) = b*ln(total) + budget  (in shifted space)
        target = total * math.exp(budget / self.b) - others
        return self.b * (math.log(target) + m) - self.q[outcome]

    def shares_to_reach(self, outcome: int, prob: float) -> float:
        """Shares of `outcome` to buy (or sell, if negative) to move its price to `prob`."""
        if not 0.0 < prob < 1.0:
            raise ValueError("target probability must be strictly between 0 and 1")
        z = [x / self.b for x in self.q]
        m = max(z)
        others = sum(math.exp(x - m) for i, x in enumerate(z) if i != outcome)
        x = math.log(prob * others / (1.0 - prob)) + m
        return self.b * x - self.q[outcome]

    # ------------------------------------------------------------ trade
    def execute(self, outcome: int, shares: float) -> float:
        """Apply a trade and return its cost in tokens (negative for a sale)."""
        if not 0 <= outcome < self.n:
            raise IndexError(f"outcome {outcome} out of range")
        c = self.quote_shares(outcome, shares)
        self.q[outcome] += shares
        return c
