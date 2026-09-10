"""Run the registered analysis (docs/ANALYSIS.md + Amendment 10) and print the verdict.

    uv run python scripts/analyze.py --runs runs/exp-v1

Writes runs/analysis/<timestamp>/{report.json,summary.txt} and, when matplotlib is
installed (`uv sync --group analysis`), the signature and Pareto plots as PNGs.
The registered constants (win margin, EWMA alpha, recovery fraction, window) are
NOT flags — see venjix/analysis.py.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from venjix.analysis import MODES, analyze, format_summary


def plot(report: dict, out_dir: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ["matplotlib not installed — plots skipped (uv sync --group analysis)"]
    written = []

    # Signature plot: mode share vs step offset from a shift, one panel per agent.
    sig = report.get("signature", {})
    if sig:
        fig, axes = plt.subplots(1, len(sig), figsize=(5 * len(sig), 3.6), sharey=True)
        axes = [axes] if len(sig) == 1 else list(axes)
        for ax, (agent, rows) in zip(axes, sig.items()):
            xs = [r["offset_bin"] for r in rows]
            bottom = [0.0] * len(rows)
            for mode in MODES:
                ys = [r[mode] for r in rows]
                ax.bar(xs, ys, width=9, bottom=bottom, label=mode, align="edge")
                bottom = [b + y for b, y in zip(bottom, ys)]
            ax.axvline(0, color="black", linewidth=1)
            ax.set_title(agent)
            ax.set_xlabel("global steps from shift")
        axes[0].set_ylabel("mode share")
        axes[-1].legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        p = out_dir / "signature.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(str(p))

    # Pareto: success rate vs cost per episode, full roster, CIs as error bars.
    par = report.get("pareto", {})
    if par:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for agent, p in par.items():
            x, y = p["cost_per_episode_usd"], p["success_rate"]
            ax.errorbar(x, y,
                        xerr=[[x - p["cost_per_episode_ci"][0]], [p["cost_per_episode_ci"][1] - x]],
                        yerr=[[y - p["success_rate_ci"][0]], [p["success_rate_ci"][1] - y]],
                        fmt="o", capsize=3)
            ax.annotate(agent, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.set_xlabel("cost per episode (USD, registered table)")
        ax.set_ylabel("success rate")
        ax.set_title("cost vs success, full roster")
        fig.tight_layout()
        p = out_dir / "pareto.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(str(p))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs/exp-v1")
    parser.add_argument("--spec", default="experiments/exp-v1.json")
    parser.add_argument("--out", default="runs/analysis")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    report = analyze(args.runs, args.spec, resamples=args.resamples, seed=args.seed)
    summary = format_summary(report)

    out_dir = Path(args.out) / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    (out_dir / "summary.txt").write_text(summary + "\n")
    notes = plot(report, out_dir)

    print(summary)
    print(f"\nreport: {out_dir}/report.json")
    for n in notes:
        print(n)


if __name__ == "__main__":
    main()
