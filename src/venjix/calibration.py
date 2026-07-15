"""World-model calibration probe: the substrate's calm-state noise floor.

Measures the real model's misprediction rate in calm states (belief is the
true goal or unknown — never stale) so that pe_threshold = 0.25 can be
validated against pre-declared bands BEFORE the frozen exp-v1 run:

    rate < 0.15          GREEN   proceed
    0.15 <= rate < 0.25  YELLOW  proceed, disclose the noise floor
    rate >= 0.25         RED     substrate fails; escalate model (deliberate
                                 amendment — nothing here changes any config)

Reuses the exact experiment machinery: build_predict_prompt (no new prompt
format), WorldModel.predict (same parser and stay-in-place fallback the agents
experience), the env's own clipped-dynamics rules, and the agents' binary
scoring (next_pos AND reward must both match).
"""

import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from venjix.config import GridworldConfig, PriceTable
from venjix.gridworld import _MOVES
from venjix.llm import LLMClient
from venjix.world import Prediction, WorldModel

MOVE_ACTIONS = ("up", "down", "left", "right")

# Pre-declared verdict bands (upper-exclusive edges).
GREEN_BELOW = 0.15
YELLOW_BELOW = 0.25


@dataclass(frozen=True)
class Case:
    idx: int
    size: int
    pos: tuple[int, int]
    action: str
    believed_goal: tuple[int, int] | None
    true_goal: tuple[int, int]
    truth_next: tuple[int, int]
    truth_reward: int
    stratum: str  # "interior_move" | "edge_move" | "probe"
    edge: bool  # position on the boundary
    belief: str  # "known" | "unknown"


@dataclass(frozen=True)
class CaseResult:
    case: Case
    pred_next: tuple[int, int]
    pred_reward: int
    parse_error: bool
    mispredicted: bool


def clipped_next(size: int, pos: tuple[int, int], action: str) -> tuple[int, int]:
    """Ground truth: identical rules to Gridworld.step (same _MOVES, same clip)."""
    dr, dc = _MOVES.get(action, (0, 0))  # probe: stay in place
    last = size - 1
    return (min(max(pos[0] + dr, 0), last), min(max(pos[1] + dc, 0), last))


def is_edge(size: int, pos: tuple[int, int]) -> bool:
    return pos[0] in (0, size - 1) or pos[1] in (0, size - 1)


def _uniform_cell(rng: random.Random, size: int, exclude: set) -> tuple[int, int]:
    while True:
        cell = (rng.randrange(size), rng.randrange(size))
        if cell not in exclude:
            return cell


def generate_cases(sizes: list[int], n_cases: int, seed: int) -> list[Case]:
    """Deterministic stratified generation. Quotas: 40% interior moves,
    30% guaranteed edge-clipping moves, 30% probes. Belief 50/50 known
    (== true goal, never stale) / unknown; 25% of known cases place the goal
    at the ground-truth next cell so reward=1 predictions are exercised.
    Unknown-belief cases place the true goal away from the next cell so
    actual reward is 0 — unknowable reward surprise is shift-type, not calm
    noise, and must not contaminate the floor."""
    rng = random.Random(seed)
    n_interior = int(n_cases * 0.4)
    n_edge = int(n_cases * 0.3)
    n_probe = n_cases - n_interior - n_edge

    cases = []

    def build(stratum: str, size: int, pos: tuple[int, int], action: str) -> None:
        truth_next = clipped_next(size, pos, action)
        known = rng.random() < 0.5
        if known:
            if rng.random() < 0.25 and truth_next != pos:
                goal = truth_next  # reward-boundary enrichment
            else:
                goal = _uniform_cell(rng, size, exclude={pos})
            believed, belief = goal, "known"
        else:
            goal = _uniform_cell(rng, size, exclude={pos, truth_next})
            believed, belief = None, "unknown"
        cases.append(
            Case(
                idx=len(cases),
                size=size,
                pos=pos,
                action=action,
                believed_goal=believed,
                true_goal=goal,
                truth_next=truth_next,
                truth_reward=1 if truth_next == goal else 0,
                stratum=stratum,
                edge=is_edge(size, pos),
                belief=belief,
            )
        )

    for _ in range(n_interior):
        size = rng.choice(sizes)
        pos = (rng.randint(1, size - 2), rng.randint(1, size - 2))
        build("interior_move", size, pos, rng.choice(MOVE_ACTIONS))
    for _ in range(n_edge):
        size = rng.choice(sizes)
        last = size - 1
        # boundary position, then an action guaranteed to clip there
        side = rng.choice(("top", "bottom", "left", "right"))
        if side in ("top", "bottom"):
            pos = (0 if side == "top" else last, rng.randrange(size))
        else:
            pos = (rng.randrange(size), 0 if side == "left" else last)
        clipping = [a for a in MOVE_ACTIONS if clipped_next(size, pos, a) == pos]
        build("edge_move", size, pos, rng.choice(clipping))
    for _ in range(n_probe):
        size = rng.choice(sizes)
        pos = (rng.randrange(size), rng.randrange(size))
        build("probe", size, pos, "probe")
    return cases


def run_probe(cases, client_factory, workers: int = 8):
    """Score every case; one client per shard (LLMClient counters are
    per-instance state — never share a client across threads). Returns
    (results ordered by case idx, usage dict)."""
    shards = [cases[i::workers] for i in range(workers) if cases[i::workers]]
    results: dict[int, CaseResult] = {}
    usage = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0}

    def work(shard) -> tuple[list[CaseResult], LLMClient]:
        client = client_factory()
        worlds: dict[int, WorldModel] = {}
        out = []
        for case in shard:
            world = worlds.setdefault(
                case.size, WorldModel(client, GridworldConfig(size=case.size))
            )
            pred: Prediction = world.predict(case.pos, case.action, case.believed_goal)
            mispredicted = (
                pred.next_pos != case.truth_next or pred.reward != case.truth_reward
            )
            out.append(
                CaseResult(
                    case=case,
                    pred_next=pred.next_pos,
                    pred_reward=pred.reward,
                    parse_error=pred.parse_error,
                    mispredicted=mispredicted,
                )
            )
        return out, client

    with ThreadPoolExecutor(max_workers=len(shards)) as pool:
        for shard_results, client in pool.map(work, shards):
            for result in shard_results:
                results[result.case.idx] = result
            usage["llm_calls"] += client.total_calls
            usage["input_tokens"] += client.total_input_tokens
            usage["output_tokens"] += client.total_output_tokens

    return [results[i] for i in sorted(results)], usage


def bootstrap_ci(values: list[int], seed: int, resamples: int = 10_000):
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(rng.choices(values, k=n)) / n for _ in range(resamples)
    )
    return means[int(0.025 * resamples)], means[int(0.975 * resamples)]


def verdict(rate: float) -> str:
    if rate < GREEN_BELOW:
        return "GREEN"
    if rate < YELLOW_BELOW:
        return "YELLOW"
    return "RED"


def _rate(results) -> float:
    return sum(r.mispredicted for r in results) / len(results) if results else 0.0


def build_report(results, usage, *, model, backend, seed, resamples) -> dict:
    flags = [int(r.mispredicted) for r in results]
    ci_low, ci_high = bootstrap_ci(flags, seed=seed, resamples=resamples)
    rate = _rate(results)

    def group_rates(key):
        groups = defaultdict(list)
        for r in results:
            groups[key(r)].append(r)
        return {
            str(name): {"n": len(rs), "misprediction_rate": round(_rate(rs), 4)}
            for name, rs in sorted(groups.items())
        }

    parse_errors = sum(r.parse_error for r in results)
    prices = PriceTable()
    return {
        "probe": "world-model calm-state calibration",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "backend": backend,
        "seed": seed,
        "n_cases": len(results),
        "overall": {
            "misprediction_rate": round(rate, 4),
            "ci95": [round(ci_low, 4), round(ci_high, 4)],
            "bootstrap_resamples": resamples,
        },
        "by_action": group_rates(lambda r: r.case.action),
        "by_position": group_rates(lambda r: "edge" if r.case.edge else "interior"),
        "by_belief": group_rates(lambda r: r.case.belief),
        "by_stratum": group_rates(lambda r: r.case.stratum),
        "parse_errors": {
            "count": parse_errors,
            "rate": round(parse_errors / len(results), 4) if results else 0.0,
        },
        "usage": {**usage, "cost_usd": prices.cost_usd(
            usage["input_tokens"], usage["output_tokens"]
        )},
        "bands": {"green_below": GREEN_BELOW, "yellow_below": YELLOW_BELOW},
        "verdict": verdict(rate),
        "ci_straddles_band": verdict(ci_low) != verdict(ci_high),
        "note": "pre-registered probe; takes no action on any verdict "
        "(pe_threshold and all configs unchanged by design)",
    }


def write_outputs(out_root, report: dict, results) -> Path:
    slug = report["model"].replace("/", "-")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_dir = Path(out_root) / f"{stamp}-{slug}"
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    with (out_dir / "cases.jsonl").open("w") as f:
        for r in results:
            record = {**asdict(r.case), **{
                "pred_next": list(r.pred_next),
                "pred_reward": r.pred_reward,
                "parse_error": r.parse_error,
                "mispredicted": r.mispredicted,
            }}
            f.write(json.dumps(record, sort_keys=True) + "\n")
    return out_dir
