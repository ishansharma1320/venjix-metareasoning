"""Concurrent executor over the frozen experiment set + pairing verification.

Conditions are independent, so they run in a thread pool (each condition gets
its OWN client — usage counters are per-run state). Blocking urllib calls in
threads are exactly what a vLLM server wants for throughput; the package stays
dependency-free.

Resume: a condition is complete when a run directory holds its config_hash and
a full episode count; completed conditions are skipped, so a killed batch can
be re-launched with the same command.

Pairing verification (Amendment 6c): for every (regime schedule, seed) group,
agents must agree on shift counts, and their (at_step, old_goal, new_goal)
sequences must be prefix-consistent. Prefix divergence means the paired-RNG
guarantee broke (a bug). Count inequality means some agent finished its 40
episodes before late shifts fired — those groups are flagged loudly and must
be handled explicitly by the analysis, never silently compared.
"""

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from venjix.experiments import DEFAULT_SET, Condition, conditions, load_spec
from venjix.llm import make_client
from venjix.runner import run


def _run_is_complete(manifest: dict, run_dir: Path) -> bool:
    """One completeness rule shared by resume and pairing verification: the
    manifest exists AND episodes.jsonl holds the full episode count. Partial
    dirs (instance died mid-run) fail this and are re-run / ignored."""
    jsonl = run_dir / "episodes.jsonl"
    if not jsonl.exists():
        return False
    episodes = sum(1 for line in jsonl.open() if '"type": "episode"' in line)
    return episodes >= manifest["config"]["n_episodes"]


def completed_hashes(out_root: str | Path) -> dict[str, Path]:
    """config_hash -> run_dir for every COMPLETE run under out_root."""
    done: dict[str, Path] = {}
    for manifest_path in Path(out_root).glob("*/manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        if _run_is_complete(manifest, manifest_path.parent):
            done[manifest["config_hash"]] = manifest_path.parent
    return done


def run_batch(
    todo: list[Condition],
    out_root: str | Path,
    backend: str,
    base_url: str | None = None,
    workers: int = 8,
) -> tuple[int, list[str]]:
    """Run conditions concurrently; returns (completed_count, failures)."""
    Path(out_root).mkdir(parents=True, exist_ok=True)
    failures = []
    completed = 0

    def one(condition: Condition):
        client = make_client(
            backend, condition.config.model, seed=condition.seed, base_url=base_url
        )
        return condition, run(condition.config, client, out_root)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, c): c for c in todo}
        for future in as_completed(futures):
            condition = futures[future]
            tag = f"{condition.regime}/{condition.agent}/seed{condition.seed}"
            try:
                _, summary = future.result()
                completed += 1
                print(
                    f"[{completed}/{len(todo)}] {tag}: "
                    f"{summary.successes}/{summary.episodes} successes, "
                    f"{summary.shifts} shifts, ${summary.cost_usd:.4f}"
                )
            except Exception as exc:  # keep the batch alive; report at the end
                failures.append(f"{tag}: {exc!r}")
                print(f"FAILED {tag}: {exc!r}")
    return completed, failures


def verify_pairing(out_root: str | Path) -> tuple[list[str], list[str], int]:
    """Returns (count_mismatches, prefix_divergences, ignored_incomplete).

    Partial run dirs (mid-run death, since re-run) are ignored — otherwise a
    partial and its complete re-run collide in the same (regime, seed, agent)
    slot and glob order decides which one gets compared."""
    ignored_incomplete = 0
    groups: dict[tuple, dict[str, list]] = defaultdict(dict)
    for manifest_path in Path(out_root).glob("*/manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        if not _run_is_complete(manifest, manifest_path.parent):
            ignored_incomplete += 1
            continue
        config = manifest["config"]
        key = (config["schedule"]["version"], config["seed"])
        shifts = []
        jsonl = manifest_path.parent / "episodes.jsonl"
        if jsonl.exists():
            for line in jsonl.open():
                record = json.loads(line)
                if record.get("type") == "shift":
                    shifts.append(
                        (
                            record["at_step"],
                            tuple(record["old_goal"]),
                            tuple(record["new_goal"]),
                        )
                    )
        groups[key][config["agent"]] = shifts

    count_mismatches, prefix_divergences = [], []
    for key, by_agent in sorted(groups.items()):
        counts = {agent: len(s) for agent, s in by_agent.items()}
        if len(set(counts.values())) > 1:
            count_mismatches.append(f"{key}: shift counts differ: {counts}")
        longest = max(by_agent.values(), key=len)
        for agent, shifts in by_agent.items():
            if shifts != longest[: len(shifts)]:
                prefix_divergences.append(
                    f"{key}: {agent} goal sequence diverges from the paired prefix"
                )
    return count_mismatches, prefix_divergences, ignored_incomplete


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen experiment set.")
    parser.add_argument("--set", default=str(DEFAULT_SET))
    parser.add_argument(
        "--backend", choices=("mock", "vllm", "vast", "anthropic"), default="vllm"
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="runs/exp-v1")
    parser.add_argument("--regime", action="append", help="repeatable filter")
    parser.add_argument("--agent", action="append", help="repeatable filter")
    parser.add_argument("--seeds", type=int, default=None, help="only seeds < N")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if not args.verify_only:
        spec = load_spec(args.set)
        selected = [
            c
            for c in conditions(spec)
            if (not args.regime or c.regime in args.regime)
            and (not args.agent or c.agent in args.agent)
            and (args.seeds is None or c.seed < args.seeds)
        ]
        done = completed_hashes(args.out)
        todo, skipped = [], []
        for condition in selected:
            run_dir = done.get(condition.config.config_hash())
            if run_dir is None:
                todo.append(condition)
            else:
                skipped.append((condition, run_dir))
        for condition, run_dir in skipped:
            print(
                f"skip (complete) {condition.regime}/{condition.agent}/"
                f"seed{condition.seed} -> {run_dir}"
            )
        print(
            f"{len(selected)} conditions selected, {len(skipped)} already "
            f"complete (skipped above), {len(todo)} to run "
            f"({args.backend}, {args.workers} workers)"
        )
        _, failures = run_batch(
            todo, args.out, args.backend, args.base_url, args.workers
        )
        if failures:
            print(f"\n{len(failures)} FAILED conditions:")
            for failure in failures:
                print(" ", failure)

    count_mismatches, prefix_divergences, ignored_incomplete = verify_pairing(args.out)
    if ignored_incomplete:
        print(
            f"\n{ignored_incomplete} incomplete run dir(s) ignored by pairing "
            f"verification (mid-run deaths; their conditions were re-run)"
        )
    if prefix_divergences:
        print("\nPAIRING BROKEN (bug — paired-RNG guarantee violated):")
        for violation in prefix_divergences:
            print(" ", violation)
    if count_mismatches:
        print(
            f"\n{len(count_mismatches)} (regime, seed) groups with unequal "
            f"shift counts (agent finished before late shifts; analysis must "
            f"handle these explicitly):"
        )
        for mismatch in count_mismatches:
            print(" ", mismatch)
    if not count_mismatches and not prefix_divergences:
        print("\npairing verified: shift counts equal and goal sequences "
              "prefix-consistent across agents for every (regime, seed)")
    if prefix_divergences or count_mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
