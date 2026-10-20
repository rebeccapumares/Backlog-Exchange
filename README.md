# Backlog Exchange

**Prediction markets for agile product backlogs.** Team members bet virtual tokens on
whether each feature will hit its success metric. Market prices become probabilities, a
track-record consensus learns whose judgment to trust, and the sprint plan reorders itself
around risk-adjusted value. It also plugs into AI copilots through MCP.

> *"Will the onboarding checklist lift day-7 retention by 2 points?"*
> The market says 57%. Your PM says 90%. Who should you believe? Backlog Exchange keeps score.


## Why

Most teams prioritize with the HiPPO (the highest paid person's opinion) or in meetings
where everyone anchors on the first number said. Both throw away what the rest of the
team knows. A prediction market pools it:

- **Skin in the game:** you put tokens behind your estimate, so cheap talk gets filtered out.
- **Anonymous and continuous:** no anchoring, no seniority effects, and anyone can update
  when they learn something.
- **It keeps score:** people who read users well earn tokens, so their view counts for more
  over time.

## What's inside

| Component | What it does |
|---|---|
| `exchange/lmsr.py` | **LMSR automated market maker** (Hanson's logarithmic market scoring rule): softmax prices, closed-form budget-to-shares and price-targeting, bounded subsidy `b·ln(n)` |
| `exchange/market.py` | The exchange: traders, binary feature markets, buy/sell, trade-to-belief, resolution and payouts, a **track-record consensus**, JSON persistence |
| `exchange/backlog.py` | Agile layer: **risk-adjusted WSJF** (`P(success) × value ÷ story points`) and an exact **0/1-knapsack sprint planner** |
| `exchange/scoring.py` | Brier score, log loss, calibration bins, per-trader profit and hit rate |
| `exchange/simulate.py` | **Agent-based study** of biased, noisy teams comparing decision methods |
| `api.py` | FastAPI REST API with interactive docs at `/docs` |
| `mcp_server.py` | **Copilot integration**: seven MCP tools for Microsoft Copilot Studio, Claude, VS Code and other MCP clients |
| `web/index.html` | Interactive single-page demo: trade, resolve, re-plan the sprint, watch the leaderboard |

## How it works

**1. Every backlog item becomes a market.** A YES share pays 1 token if the feature hits
its success metric (for example *"Revenue per visitor +5% with churn flat"*), and 0 if it
doesn't. The LMSR market maker always quotes a price:

```
C(q) = b · ln Σ exp(q_i / b)        price_i = exp(q_i / b) / Σ_j exp(q_j / b)
```

So the YES price is the team's live probability of success.

**2. The exchange learns who to trust.** Each trade records the trader's stated
probability. The *track-record consensus* weights those probabilities by each person's
token wealth, which is earned by past accuracy.

**3. The backlog re-ranks itself.** Expected value = probability × business value. The
planner picks the set of features that maximizes total expected value within the sprint's
story-point capacity. In the demo workspace at 20 points, the market-informed plan
carries **€174.5k** of expected value against **€164.0k** for picking the biggest-value
items first, a 6.4% gain from the same capacity.

## The study: does a market beat how teams usually decide?

`run_simulation.py` simulates **30 teams over 40 sprints** with 12 candidate features each.
Every member receives a noisy signal of each feature's hidden true success probability,
with realistic flaws:

- skill varies widely, from sharp to mostly guessing
- feature owners are optimistic about their own ideas
- meetings anchor on the manager's opening estimate

Results, scored from sprint 11 onward once the market has had time to learn:

| Method | Error vs. truth (MSE) | Brier | Expected value captured |
|---|---|---|---|
| Manager decides (HiPPO) | 0.0422 | 0.2433 | 94.5% |
| Anchored meeting | 0.0136 | 0.2138 | 98.3% |
| Independent anonymous vote | 0.0064 | 0.2059 | 99.2% |
| Market price | 0.0064 | 0.2058 | 99.2% |
| **Track-record consensus** | **0.0050** | **0.2043** | **99.4%** |
| *Ideal (weights by true skill, unknowable)* | *0.0040* | *0.2030* | *99.6%* |

**Findings**

- The track-record consensus cuts forecast error by **22%** vs. an independent vote,
  **63%** vs. an anchored meeting, and **88%** vs. the manager deciding alone. It closes
  most of the gap to the unreachable ideal.
- The raw market price **ties** the independent vote. Its value is less in the price itself
  than in the track record it builds.
- Final token balances correlate **−0.71** with forecasting noise, so the market reliably
  moves influence toward better forecasters (see `results/wealth_vs_skill.png`).
- Brier scores look close because real outcomes are noisy coin flips. Error against the true
  probability, which only a simulation can see, shows the real differences.

**Honest notes from building it.** The first version of the simulated market trailed the
simple vote. Diagnosing why showed that wealth weighting was working, but traders were
moving the price in large, order-dependent jumps. That made the price settle at a weighted
*median* instead of a weighted mean. Smaller, decaying trades, the way real markets settle,
fixed price formation. I stopped tuning there rather than adjust the simulation until the
market "won." The data is simulated. A real deployment would need a few dozen resolved
features before the track record becomes reliable.

## Copilot integration (MCP)

`mcp_server.py` exposes the exchange to any MCP-compatible assistant:

| Tool | Example prompt |
|---|---|
| `list_features` | "What does the team think our features' odds are?" |
| `get_feature` | "Show me the price history for the price test." |
| `plan_sprint` | "Which features should we build with 20 points?" |
| `place_forecast` | "Put Sam at 70% on the onboarding checklist." |
| `resolve_feature` | "The referral program missed its target, so resolve it." |
| `leaderboard` | "Who on the team forecasts best?" |
| `forecasting_report` | "How calibrated have our markets been?" |

Example Claude Desktop config:

```json
{ "mcpServers": { "backlog-exchange": {
    "command": "python", "args": ["mcp_server.py"], "cwd": "/path/to/backlog-exchange" } } }
```

## Quick start

```bash
pip install -r requirements.txt

uvicorn api:app --reload        # API + web demo at http://127.0.0.1:8000  (docs at /docs)
python mcp_server.py            # copilot tools over stdio
python run_simulation.py        # reproduce the study and charts (about 20 s)
python -m pytest tests -q       # 15 tests
```

The web demo (`web/index.html`) also runs on its own in a browser, with the same LMSR math
ported to JavaScript.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/markets` | all feature markets with price, consensus and history |
| POST | `/markets` | list a new feature (`id, title, success_metric, value, effort, prior`) |
| POST | `/markets/{id}/forecast` | state a probability and trade toward it |
| POST | `/markets/{id}/buy` | spend tokens on YES or NO |
| POST | `/markets/{id}/resolve` | settle once the metric is measured |
| GET | `/backlog?capacity=20` | risk-adjusted ranking and optimal sprint plan |
| GET | `/leaderboard`, `/report` | track records, Brier score and calibration |

## Tests

15 tests cover:
- the LMSR invariants: prices sum to 1, exact budget-to-shares, price targeting, bounded loss
- trade validation, buy/sell round-trips, and value conservation on resolution
- wealth-weighted consensus and persistence
- the knapsack planner against brute force
- scoring and the simulation's method ordering
- the full REST API flow

## Project structure

```
backlog-exchange/
├── exchange/
│   ├── lmsr.py        market maker
│   ├── market.py      exchange, traders, resolution, consensus
│   ├── backlog.py     risk-adjusted WSJF + sprint knapsack
│   ├── scoring.py     Brier, calibration, trader accuracy
│   ├── simulate.py    agent-based team study
│   ├── service.py     shared service layer with JSON persistence
│   └── demo.py        example workspace (note-taking app team)
├── api.py             FastAPI app
├── mcp_server.py      copilot tools
├── run_simulation.py  study + charts
├── web/index.html     interactive demo
├── results/           simulation.json, charts, log
└── tests/
```
