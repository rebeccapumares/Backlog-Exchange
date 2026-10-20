"""The exchange: traders, feature markets, trading, resolution and payouts.

Each backlog feature gets a binary market: "Will this feature hit its success
metric?" YES shares pay 1 token if it does, NO shares pay 1 token if it
doesn't. The YES price is the team's collective probability of success.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import asdict, dataclass, field

from .lmsr import LMSR

YES, NO = 0, 1
OUTCOME_NAMES = ["YES", "NO"]


class TradeError(ValueError):
    """Raised for invalid trades (unknown trader, insufficient funds, closed market...)."""


@dataclass
class Trader:
    name: str
    balance: float = 1000.0
    role: str = "member"           # e.g. engineer, designer, PM, AI
    starting_balance: float = 1000.0


@dataclass
class Feature:
    id: str
    title: str
    success_metric: str            # e.g. "+2% 7-day retention within 4 weeks of launch"
    value: float                   # business value (EUR) if the feature succeeds
    effort: int                    # story points
    owner: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class Trade:
    trader: str
    market_id: str
    outcome: int
    shares: float
    cost: float
    price_after: float             # YES price after the trade
    timestamp: float


class FeatureMarket:
    def __init__(self, feature: Feature, liquidity: float = 100.0, prior: float = 0.5):
        self.feature = feature
        self.amm = LMSR(2, liquidity)
        if prior != 0.5:                                   # seed the opening price
            self.amm.q[YES] = self.amm.shares_to_reach(YES, prior)
        self.positions: dict[str, list[float]] = {}
        self.beliefs: dict[str, float] = {}         # each trader's latest stated probability
        self.history: list[Trade] = []
        self.status = "open"
        self.resolution: int | None = None
        self.opening_price = self.yes_price

    @property
    def id(self) -> str:
        return self.feature.id

    @property
    def yes_price(self) -> float:
        return self.amm.price(YES)

    def position(self, trader: str) -> list[float]:
        return self.positions.setdefault(trader, [0.0, 0.0])


class Exchange:
    def __init__(self, liquidity: float = 100.0):
        self.liquidity = liquidity
        self.traders: dict[str, Trader] = {}
        self.markets: dict[str, FeatureMarket] = {}
        self._clock = itertools.count()

    # -------------------------------------------------------------- setup
    def add_trader(self, name: str, balance: float = 1000.0, role: str = "member") -> Trader:
        if name in self.traders:
            raise TradeError(f"trader '{name}' already exists")
        t = Trader(name, balance, role, balance)
        self.traders[name] = t
        return t

    def list_feature(self, feature: Feature, prior: float = 0.5,
                     liquidity: float | None = None) -> FeatureMarket:
        if feature.id in self.markets:
            raise TradeError(f"market '{feature.id}' already exists")
        m = FeatureMarket(feature, liquidity or self.liquidity, prior)
        self.markets[feature.id] = m
        return m

    # ------------------------------------------------------------ helpers
    def _get(self, trader: str, market_id: str) -> tuple[Trader, FeatureMarket]:
        if trader not in self.traders:
            raise TradeError(f"unknown trader '{trader}'")
        if market_id not in self.markets:
            raise TradeError(f"unknown market '{market_id}'")
        m = self.markets[market_id]
        if m.status != "open":
            raise TradeError(f"market '{market_id}' is closed")
        return self.traders[trader], m

    def _record(self, t: Trader, m: FeatureMarket, outcome: int, shares: float,
                cost: float) -> Trade:
        t.balance -= cost
        m.position(t.name)[outcome] += shares
        trade = Trade(t.name, m.id, outcome, shares, cost, m.yes_price,
                      time.time() + next(self._clock) * 1e-6)
        m.history.append(trade)
        return trade

    # ------------------------------------------------------------- trading
    def buy(self, trader: str, market_id: str, outcome: int, budget: float) -> Trade:
        """Spend `budget` tokens on YES or NO shares."""
        t, m = self._get(trader, market_id)
        if budget <= 0:
            raise TradeError("budget must be positive")
        if budget > t.balance + 1e-9:
            raise TradeError(f"{trader} has only {t.balance:.2f} tokens")
        shares = m.amm.shares_for_budget(outcome, budget)
        cost = m.amm.execute(outcome, shares)
        return self._record(t, m, outcome, shares, cost)

    def sell(self, trader: str, market_id: str, outcome: int, shares: float) -> Trade:
        """Sell shares you hold back to the market maker."""
        t, m = self._get(trader, market_id)
        held = m.position(trader)[outcome]
        if shares <= 0 or shares > held + 1e-9:
            raise TradeError(f"{trader} holds {held:.2f} {OUTCOME_NAMES[outcome]} shares")
        cost = m.amm.execute(outcome, -shares)
        return self._record(t, m, outcome, -shares, cost)

    def trade_to_belief(self, trader: str, market_id: str, belief: float,
                        max_spend: float | None = None) -> Trade | None:
        """Move the YES price toward the trader's belief, spending at most `max_spend`.

        If the market is below their belief they buy YES, if above they buy NO.
        With a spending cap they move the price only part of the way, so a
        trader's influence is proportional to their wealth.
        """
        t, m = self._get(trader, market_id)
        belief = min(max(belief, 0.01), 0.99)
        m.beliefs[trader] = belief
        p = m.yes_price
        if abs(belief - p) < 1e-4:
            return None
        outcome = YES if belief > p else NO
        target = belief if outcome == YES else 1 - belief
        needed = m.amm.quote_shares(outcome, m.amm.shares_to_reach(outcome, target))
        cap = t.balance if max_spend is None else min(max_spend, t.balance)
        spend = min(needed, cap)
        if spend <= 1e-6:
            return None
        return self.buy(trader, market_id, outcome, spend)

    def submit_forecast(self, trader: str, market_id: str, probability: float) -> None:
        """Record a stated probability without trading (it still counts in the consensus)."""
        _, m = self._get(trader, market_id)
        m.beliefs[trader] = min(max(probability, 0.01), 0.99)

    def consensus(self, market_id: str) -> float:
        """Track-record consensus: stated probabilities weighted by each trader's
        current wealth. Wealth is earned by past forecasting accuracy, so people
        with good track records count for more. Falls back to the market price."""
        m = self.markets[market_id]
        weights = {n: max(self.portfolio_value(n), 0.0) for n in m.beliefs}
        total = sum(weights.values())
        if total <= 0:
            return m.yes_price
        return sum(weights[n] * p for n, p in m.beliefs.items()) / total

    # ---------------------------------------------------------- resolution
    def resolve(self, market_id: str, success: bool) -> dict[str, float]:
        """Settle a market: each winning share pays 1 token. Returns payouts."""
        if market_id not in self.markets:
            raise TradeError(f"unknown market '{market_id}'")
        m = self.markets[market_id]
        if m.status != "open":
            raise TradeError(f"market '{market_id}' is already resolved")
        winner = YES if success else NO
        payouts = {}
        for name, pos in m.positions.items():
            pay = pos[winner]
            self.traders[name].balance += pay
            payouts[name] = pay
        m.status, m.resolution = "resolved", winner
        return payouts

    # --------------------------------------------------------------- views
    def portfolio_value(self, trader: str) -> float:
        """Cash plus open positions marked at current prices."""
        t = self.traders[trader]
        value = t.balance
        for m in self.markets.values():
            if m.status == "open" and trader in m.positions:
                yes, no = m.positions[trader]
                value += yes * m.yes_price + no * (1 - m.yes_price)
        return value

    def leaderboard(self) -> list[dict]:
        rows = []
        for name, t in self.traders.items():
            v = self.portfolio_value(name)
            rows.append({"trader": name, "role": t.role, "value": round(v, 2),
                         "return_pct": round(100 * (v / t.starting_balance - 1), 2)})
        return sorted(rows, key=lambda r: -r["value"])

    def price_history(self, market_id: str) -> list[float]:
        m = self.markets[market_id]
        return [m.opening_price] + [tr.price_after for tr in m.history]

    # --------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {
            "liquidity": self.liquidity,
            "traders": [asdict(t) for t in self.traders.values()],
            "markets": [{
                "feature": asdict(m.feature), "liquidity": m.amm.b, "q": m.amm.q,
                "positions": m.positions, "beliefs": m.beliefs,
                "history": [asdict(h) for h in m.history],
                "status": m.status, "resolution": m.resolution,
                "opening_price": m.opening_price,
            } for m in self.markets.values()],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Exchange":
        ex = cls(d["liquidity"])
        for t in d["traders"]:
            ex.traders[t["name"]] = Trader(**t)
        for md in d["markets"]:
            m = FeatureMarket(Feature(**md["feature"]), md["liquidity"])
            m.amm.q = list(md["q"])
            m.positions = {k: list(v) for k, v in md["positions"].items()}
            m.beliefs = dict(md.get("beliefs", {}))
            m.history = [Trade(**h) for h in md["history"]]
            m.status, m.resolution = md["status"], md["resolution"]
            m.opening_price = md["opening_price"]
            ex.markets[m.id] = m
        return ex
