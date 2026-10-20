"""Agent-based study: does a feature prediction market beat how teams usually decide?

Each sprint a team weighs a batch of candidate features. Every feature has a
hidden true chance of success. Team members receive noisy private signals of
it, and the noise and biases are what real teams show:

  - skill varies: some people read users well, others are mostly guessing
  - owner optimism: whoever proposed a feature overrates it
  - HiPPO: the "highest paid person's opinion" carries the meeting
  - anchoring: in a meeting, people drift toward the first number said

Several ways of turning those opinions into a forecast are compared:

  HiPPO      the manager decides
  Meeting    everyone states a number after hearing the manager (anchored), averaged
  Vote       independent, equal-weight average of everyone's estimate
  Market     a persistent LMSR market; tokens carry over between sprints, so
             people who forecast well gain influence and biased owners lose it
  Consensus  the exchange's track-record consensus: each member's stated
             probability weighted by their market wealth (their track record)
  Ideal      an unreachable upper bound that weights members by their true
             skill, which no real team can observe

Each forecast feeds the same sprint planner, and features then succeed or fail
at their true rates. We measure forecast accuracy (Brier score) and how much
of the best achievable value each method delivers.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .backlog import plan_sprint, rank_features
from .market import Exchange, Feature
from .scoring import brier

METHODS = ["HiPPO", "Meeting", "Vote", "Market", "Consensus", "Ideal"]


def logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


@dataclass
class Member:
    name: str
    noise: float            # std dev of signal noise in log-odds
    owner_bias: float       # log-odds optimism about features they own
    is_manager: bool = False


@dataclass
class SimConfig:
    n_members: int = 8
    features_per_sprint: int = 12
    capacity: int = 26                 # story points per sprint
    sprints: int = 40
    liquidity: float = 60.0
    start_tokens: float = 1000.0
    step_fraction: float = 0.01        # share of wealth per trade, scaled by disagreement
    trading_rounds: int = 12           # many small trades -> wealth-weighted consensus
    anchor_weight: float = 0.45        # how far meeting estimates drift toward the manager


def make_team(rng: random.Random, cfg: SimConfig) -> list[Member]:
    team = [Member("manager", noise=rng.uniform(0.9, 1.3), owner_bias=0.6, is_manager=True)]
    for i in range(1, cfg.n_members):
        team.append(Member(f"member_{i}", noise=rng.choice([0.4, 0.7, 1.0, 1.4, 1.8]),
                           owner_bias=rng.uniform(0.4, 1.2)))
    return team


def make_features(rng: random.Random, cfg: SimConfig, sprint: int, team: list[Member]):
    feats, truth = [], {}
    for k in range(cfg.features_per_sprint):
        fid = f"S{sprint:02d}-F{k:02d}"
        p = rng.betavariate(2.0, 2.6)
        value = round(rng.lognormvariate(math.log(30000), 0.6), -2)
        effort = rng.choice([1, 2, 3, 5, 5, 8, 8, 13])
        owner = rng.choice(team).name
        feats.append(Feature(fid, f"Feature {fid}", "hits its success metric", value, effort, owner))
        truth[fid] = p
    return feats, truth


def signals(rng: random.Random, team: list[Member], feats: list[Feature], truth: dict):
    """Each member's private estimate of each feature's success probability."""
    est = {}
    for m in team:
        for f in feats:
            x = logit(truth[f.id]) + rng.gauss(0, m.noise)
            if f.owner == m.name:
                x += m.owner_bias
            est[(m.name, f.id)] = sigmoid(x)
    return est


def realized_value(plan: dict, feats: list[Feature], outcomes: dict) -> float:
    by_id = {f.id: f for f in feats}
    return sum(by_id[i].value for i in plan["features"] if outcomes[i])


def expected_true_value(plan: dict, feats: list[Feature], truth: dict) -> float:
    by_id = {f.id: f for f in feats}
    return sum(truth[i] * by_id[i].value for i in plan["features"])


def run(seed: int = 0, cfg: SimConfig | None = None) -> dict:
    cfg = cfg or SimConfig()
    rng = random.Random(seed)
    team = make_team(rng, cfg)
    manager = team[0].name
    ex = Exchange(cfg.liquidity)
    for m in team:
        ex.add_trader(m.name, cfg.start_tokens, "manager" if m.is_manager else "member")

    per_sprint = []
    for s in range(cfg.sprints):
        feats, truth = make_features(rng, cfg, s, team)
        est = signals(rng, team, feats, truth)
        outcomes = {f.id: rng.random() < truth[f.id] for f in feats}

        forecasts = {meth: {} for meth in METHODS}
        wealth = {m.name: ex.traders[m.name].balance for m in team}   # track record so far
        total_wealth = sum(wealth.values())
        for f in feats:
            hippo = est[(manager, f.id)]
            forecasts["HiPPO"][f.id] = hippo
            forecasts["Vote"][f.id] = sum(est[(m.name, f.id)] for m in team) / len(team)
            # track-record consensus: stated probabilities weighted by market wealth
            forecasts["Consensus"][f.id] = sum(wealth[m.name] * est[(m.name, f.id)]
                                               for m in team) / total_wealth
            # upper bound: weight each member by true skill (1 / noise^2), unknowable in practice
            w = {m.name: 1 / m.noise ** 2 for m in team}
            forecasts["Ideal"][f.id] = sigmoid(sum(w[m.name] * logit(est[(m.name, f.id)])
                                                   for m in team) / sum(w.values()))
            forecasts["Meeting"][f.id] = sum(
                hippo if m.name == manager else
                (1 - cfg.anchor_weight) * est[(m.name, f.id)] + cfg.anchor_weight * hippo
                for m in team) / len(team)

        # market: list every feature at 50%, then members trade in random order
        for f in feats:
            ex.list_feature(f, prior=0.5)
        for rnd in range(cfg.trading_rounds):
            # trades shrink as the session goes on, so the price settles at the
            # wealth-weighted consensus instead of bouncing with trade order
            decay = 1.0 / (1 + rnd)
            order = [(m.name, f.id) for m in team for f in feats]
            rng.shuffle(order)
            for name, fid in order:
                gap = abs(est[(name, fid)] - ex.markets[fid].yes_price)
                budget = ex.traders[name].balance * cfg.step_fraction * decay * gap / 0.1
                if budget > 0.01:
                    ex.trade_to_belief(name, fid, est[(name, fid)], max_spend=budget)
        for f in feats:
            forecasts["Market"][f.id] = ex.markets[f.id].yes_price
            ex.resolve(f.id, outcomes[f.id])

        row = {"sprint": s}
        oracle = plan_sprint(rank_features(feats, truth), cfg.capacity)
        row["oracle_ev"] = expected_true_value(oracle, feats, truth)
        for meth in METHODS:
            probs = [forecasts[meth][f.id] for f in feats]
            outs = [int(outcomes[f.id]) for f in feats]
            plan = plan_sprint(rank_features(feats, forecasts[meth]), cfg.capacity)
            row[meth] = {
                "brier": brier(probs, outs),
                "mse_true": sum((forecasts[meth][f.id] - truth[f.id]) ** 2 for f in feats) / len(feats),
                "ev": expected_true_value(plan, feats, truth),
                "realized": realized_value(plan, feats, outcomes),
            }
        per_sprint.append(row)

    wealth = {name: round(t.balance, 1) for name, t in ex.traders.items()}
    noise = {m.name: m.noise for m in team}
    return {"seed": seed, "per_sprint": per_sprint, "final_wealth": wealth, "noise": noise}


def summarize(runs: list[dict], warmup: int = 0) -> dict:
    """Average results across runs (optionally skipping early sprints)."""
    out = {}
    for meth in METHODS:
        briers, mses, ev_share, realized = [], [], [], []
        for r in runs:
            for row in r["per_sprint"][warmup:]:
                briers.append(row[meth]["brier"])
                mses.append(row[meth]["mse_true"])
                ev_share.append(row[meth]["ev"] / row["oracle_ev"])
                realized.append(row[meth]["realized"])
        out[meth] = {"brier": sum(briers) / len(briers),
                     "error_vs_truth": sum(mses) / len(mses),
                     "value_captured_pct": 100 * sum(ev_share) / len(ev_share),
                     "avg_realized_value": sum(realized) / len(realized)}
    return out


def learning_curve(runs: list[dict], method: str, metric: str = "mse_true") -> list[float]:
    n = len(runs[0]["per_sprint"])
    return [sum(r["per_sprint"][s][method][metric] for r in runs) / len(runs) for s in range(n)]


def wealth_vs_skill(runs: list[dict]) -> float:
    """Correlation between a member's signal noise and final wealth (expect negative)."""
    xs, ys = [], []
    for r in runs:
        for name, w in r["final_wealth"].items():
            xs.append(r["noise"][name])
            ys.append(w)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / math.sqrt(vx * vy)
