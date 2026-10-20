"""Run the simulation study and write results/simulation.json plus two charts.

    python run_simulation.py            # 30 simulated teams x 40 sprints
    python run_simulation.py --teams 10 # faster
"""
import argparse
import json
from pathlib import Path

from exchange.simulate import METHODS, SimConfig, learning_curve, run, summarize, wealth_vs_skill

LABELS = {"HiPPO": "Manager decides (HiPPO)", "Meeting": "Anchored meeting",
          "Vote": "Independent vote", "Market": "Market price",
          "Consensus": "Track-record consensus", "Ideal": "Ideal (true skill, unknowable)"}


def charts(summary: dict, runs: list[dict], out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    accent, neutral = "#2a78d6", "#b9b8b2"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": ink,
                         "axes.labelcolor": muted, "xtick.color": muted, "ytick.color": muted})

    # 1) forecast error by method, sorted, exchange outputs highlighted
    order = sorted(METHODS, key=lambda m: summary[m]["error_vs_truth"], reverse=True)
    vals = [summary[m]["error_vs_truth"] for m in order]
    colors = [accent if m in ("Market", "Consensus") else neutral for m in order]
    fig, ax = plt.subplots(figsize=(7.5, 3.6), facecolor=surface)
    ax.set_facecolor(surface)
    bars = ax.barh([LABELS[m] for m in order], vals, color=colors, height=0.6)
    for b, v in zip(bars, vals):
        ax.text(v + max(vals) * 0.01, b.get_y() + b.get_height() / 2, f"{v:.4f}",
                va="center", fontsize=9, color=muted)
    ax.set_xlabel("Mean squared error vs. true success probability (lower is better)")
    ax.set_title("Forecast error by decision method", loc="left", fontsize=12, color=ink)
    ax.grid(axis="x", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(out / "forecast_error.png", dpi=160)
    plt.close(fig)

    # 2) the market learns who to trust: final wealth vs forecasting noise
    xs, ys = [], []
    for r in runs:
        for name, w in r["final_wealth"].items():
            xs.append(r["noise"][name])
            ys.append(w)
    fig, ax = plt.subplots(figsize=(7.5, 3.6), facecolor=surface)
    ax.set_facecolor(surface)
    ax.scatter(xs, ys, s=28, color=accent, alpha=0.55, edgecolors=surface, linewidths=1)
    ax.set_yscale("log")
    ax.set_xlabel("Member's signal noise (lower = better forecaster)")
    ax.set_ylabel("Tokens after 40 sprints (log)")
    ax.set_title("The market shifts influence to better forecasters", loc="left",
                 fontsize=12, color=ink)
    ax.axhline(1000, color=muted, linewidth=1, linestyle="--")
    ax.text(max(xs), 1000 * 1.15, "starting balance", ha="right", fontsize=9, color=muted)
    ax.grid(color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(grid)
    fig.tight_layout()
    fig.savefig(out / "wealth_vs_skill.png", dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teams", type=int, default=30)
    p.add_argument("--sprints", type=int, default=40)
    p.add_argument("--warmup", type=int, default=10, help="sprints skipped while the market learns")
    a = p.parse_args()

    cfg = SimConfig(sprints=a.sprints)
    runs = [run(seed, cfg) for seed in range(a.teams)]
    summary = summarize(runs, warmup=a.warmup)
    corr = wealth_vs_skill(runs)

    print(f"{a.teams} teams x {a.sprints} sprints ({cfg.features_per_sprint} features each), "
          f"scored after sprint {a.warmup}\n")
    print(f"{'method':32s} {'error vs truth':>15s} {'Brier':>8s} {'value captured':>15s}")
    for m in METHODS:
        s = summary[m]
        print(f"{LABELS[m]:32s} {s['error_vs_truth']:15.4f} {s['brier']:8.4f} "
              f"{s['value_captured_pct']:14.1f}%")
    print(f"\ncorrelation between forecasting noise and final tokens: {corr:+.2f}")

    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "simulation.json").write_text(json.dumps({
        "config": vars(cfg), "teams": a.teams, "warmup": a.warmup, "summary": summary,
        "noise_wealth_correlation": corr,
        "learning_curve": {m: learning_curve(runs, m) for m in METHODS},
    }, indent=2))
    charts(summary, runs, out)
    print("wrote results/simulation.json, results/forecast_error.png, results/wealth_vs_skill.png")


if __name__ == "__main__":
    main()
