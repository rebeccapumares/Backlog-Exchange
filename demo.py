"""A realistic demo workspace: a note-taking app's product team and its backlog."""
from __future__ import annotations

import random

from .market import Exchange, Feature

TEAM = [
    ("Marta", "product manager"), ("Jordi", "backend engineer"), ("Aiko", "designer"),
    ("Sam", "data analyst"), ("Lucía", "frontend engineer"), ("Omar", "growth lead"),
]

BACKLOG = [
    Feature("NOTE-101", "Offline sync rewrite", "Sync-error support tickets down 40% in 4 weeks",
            120000, 13, "Jordi", ["reliability"]),
    Feature("NOTE-102", "AI meeting summaries", "15% of weekly active users try it in 30 days",
            90000, 8, "Marta", ["ai", "new"]),
    Feature("NOTE-103", "Annual plan price test (€59 to €69)", "Revenue per visitor +5% with churn flat",
            150000, 3, "Omar", ["pricing"]),
    Feature("NOTE-104", "Onboarding checklist", "Day-7 retention +2 points",
            70000, 5, "Aiko", ["activation"]),
    Feature("NOTE-105", "Dark mode", "App rating +0.1 within 6 weeks",
            20000, 3, "Lucía", ["ux"]),
    Feature("NOTE-106", "Shared workspaces for teams", "200 paying teams within 60 days",
            110000, 13, "Marta", ["b2b", "new"]),
    Feature("NOTE-107", "Faster search index", "Search latency p95 under 300 ms and +3% search usage",
            40000, 5, "Jordi", ["performance"]),
    Feature("NOTE-108", "Referral rewards", "8% of new signups from referrals",
            60000, 5, "Omar", ["growth"]),
]

# rough stated beliefs per person, used to seed demo trading activity
_BELIEFS = {
    "NOTE-101": [0.70, 0.85, 0.60, 0.65, 0.70, 0.55],
    "NOTE-102": [0.80, 0.45, 0.60, 0.40, 0.50, 0.55],
    "NOTE-103": [0.50, 0.45, 0.40, 0.60, 0.45, 0.75],
    "NOTE-104": [0.55, 0.50, 0.75, 0.50, 0.55, 0.60],
    "NOTE-105": [0.30, 0.35, 0.50, 0.25, 0.65, 0.30],
    "NOTE-106": [0.65, 0.30, 0.40, 0.30, 0.35, 0.45],
    "NOTE-107": [0.60, 0.80, 0.55, 0.70, 0.65, 0.50],
    "NOTE-108": [0.45, 0.40, 0.45, 0.35, 0.40, 0.70],
}


def build_demo(seed: int = 3) -> Exchange:
    rng = random.Random(seed)
    ex = Exchange(liquidity=80)
    for name, role in TEAM:
        ex.add_trader(name, 1000, role)
    for f in BACKLOG:
        ex.list_feature(f, prior=0.5)
    order = [(i, f.id) for i in range(len(TEAM)) for f in BACKLOG]
    for rnd in range(10):                      # small, shrinking trades let prices settle
        rng.shuffle(order)
        for i, fid in order:
            name, belief = TEAM[i][0], _BELIEFS[fid][i]
            gap = abs(belief - ex.markets[fid].yes_price)
            ex.trade_to_belief(name, fid, belief, max_spend=60 * gap / (1 + rnd))
    return ex
