"""Copilot integration: exposes Backlog Exchange as MCP tools.

MCP (Model Context Protocol) is an open standard for connecting AI assistants
to tools. Point Microsoft Copilot Studio, Claude Desktop, VS Code or any MCP
client at this server and you can ask things like:

    "Which features should we build this sprint with 20 points?"
    "What does the team think about the price test?"
    "Put 70% on NOTE-104 for Sam."

Run:  python mcp_server.py        (stdio transport)
"""
from __future__ import annotations

try:                                    # MCP Python SDK v2
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:                     # MCP Python SDK v1
    from mcp.server.fastmcp import FastMCP as _Server

from exchange.market import TradeError
from exchange.service import ExchangeService

mcp = _Server("backlog-exchange")
service = ExchangeService()


def _safe(fn, *args):
    try:
        return fn(*args)
    except (KeyError, TradeError) as e:
        return {"error": str(e)}


@mcp.tool()
def list_features(status: str = "open") -> list[dict]:
    """List backlog features with their market probability of success, the
    track-record consensus, value (EUR) and effort (story points)."""
    rows = service.list_markets(status or None)
    return [{k: r[k] for k in ("id", "title", "success_metric", "yes_price", "consensus",
                               "value", "effort", "owner")} for r in rows]


@mcp.tool()
def get_feature(feature_id: str) -> dict:
    """Full detail for one feature market, including its price history."""
    return _safe(service.market_view, feature_id)


@mcp.tool()
def plan_sprint(capacity: int = 20) -> dict:
    """Rank the backlog by risk-adjusted value per story point and return the
    sprint plan that maximizes expected value within `capacity` points."""
    return service.backlog(capacity)


@mcp.tool()
def place_forecast(trader: str, feature_id: str, probability: float,
                   max_spend: float = 50.0) -> dict:
    """Record a team member's probability (0-1) that a feature hits its metric
    and trade the market toward it, spending at most `max_spend` tokens."""
    return _safe(service.forecast, trader, feature_id, probability, max_spend)


@mcp.tool()
def resolve_feature(feature_id: str, success: bool) -> dict:
    """Settle a feature's market once its success metric is measured."""
    return _safe(service.resolve, feature_id, success)


@mcp.tool()
def leaderboard() -> list[dict]:
    """Team members ranked by portfolio value, i.e. their forecasting track record."""
    return service.leaderboard()


@mcp.tool()
def forecasting_report() -> dict:
    """Accuracy of resolved markets (Brier score, calibration) and per-person hit rates."""
    return service.report()


if __name__ == "__main__":
    mcp.run()
