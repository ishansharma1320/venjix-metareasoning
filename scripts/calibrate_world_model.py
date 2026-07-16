"""World-model calibration probe CLI — see src/venjix/calibration.py.

Usage (live, against the vast endpoint):
    set -a; source .env; set +a
    uv run python scripts/calibrate_world_model.py \
        --backend vast --model Qwen/Qwen3-4B --workers 8

Mock dress rehearsal (perfect world model, expects rate 0.00 GREEN):
    uv run python scripts/calibrate_world_model.py --backend mock --model mock
"""

import argparse

from venjix.calibration import (
    build_report,
    generate_cases,
    run_probe,
    write_outputs,
)
from venjix.experiments import load_spec
from venjix.llm import make_client


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="exact served model id")
    parser.add_argument("--backend", choices=("vast", "vllm", "openrouter", "mock"), default="vast")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--cases", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--out", default="runs/calibration")
    args = parser.parse_args()

    sizes = sorted({r["env"]["size"] for r in load_spec()["regimes"]})
    print(f"generating {args.cases} cases over grid sizes {sizes} (seed {args.seed})")
    cases = generate_cases(sizes, args.cases, args.seed)

    def client_factory():
        return make_client(args.backend, args.model, seed=args.seed, base_url=args.base_url)

    print(f"probing {args.model} via {args.backend} with {args.workers} workers...")
    results, usage = run_probe(cases, client_factory, workers=args.workers)

    report = build_report(
        results, usage,
        model=args.model, backend=args.backend,
        seed=args.seed, resamples=args.resamples,
    )
    out_dir = write_outputs(args.out, report, results)

    overall = report["overall"]
    print(f"\nreport: {out_dir}/report.json")
    print(f"cases:  {report['n_cases']}   parse errors: {report['parse_errors']['count']}")
    print("by action:  ", {k: v["misprediction_rate"] for k, v in report["by_action"].items()})
    print("by position:", {k: v["misprediction_rate"] for k, v in report["by_position"].items()})
    print("by belief:  ", {k: v["misprediction_rate"] for k, v in report["by_belief"].items()})
    print(
        f"\ncalm-state misprediction rate: {overall['misprediction_rate']:.4f} "
        f"(95% CI [{overall['ci95'][0]:.4f}, {overall['ci95'][1]:.4f}])"
    )
    print(
        f"VERDICT: {report['verdict']}  "
        f"(bands: <{report['bands']['green_below']} GREEN, "
        f"<{report['bands']['yellow_below']} YELLOW, else RED)"
    )
    if report["ci_straddles_band"]:
        print("WARNING: the 95% CI straddles a band edge — verdict is not CI-stable.")
    print("no action taken on this verdict (probe changes nothing by design).")


if __name__ == "__main__":
    main()
