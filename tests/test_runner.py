import json
import math

from venjix.config import (
    GridworldConfig,
    PriceTable,
    RunConfig,
    ShiftEntry,
    ShiftSchedule,
)
from venjix.llm import MockModel
from venjix.runner import run

STEP_FIELDS = {
    "type", "episode", "step_in_episode", "global_step", "mode", "action",
    "parse_error", "pos", "reward", "done", "success", "probe_result",
    "llm_calls", "input_tokens", "output_tokens", "cost_usd", "wall_time_ms",
    "prediction_error",
}

VOLATILE = ("wall_time_ms",)


def make_config(seed=11, n_episodes=3):
    return RunConfig(
        env=GridworldConfig(size=7),
        schedule=ShiftSchedule(
            version="test-v1", entries=(ShiftEntry(10, 3), ShiftEntry(40, 5))
        ),
        seed=seed,
        n_episodes=n_episodes,
        model="mock",
        prices=PriceTable(),
    )


def do_run(tmp_path, name, seed=11):
    config = make_config(seed=seed)
    summary = run(config, MockModel(seed=seed), tmp_path / name)
    lines = [
        json.loads(line)
        for line in (tmp_path / name / summary.run_dir.split("/")[-1] / "episodes.jsonl")
        .read_text()
        .splitlines()
    ]
    manifest = json.loads(
        (tmp_path / name / summary.run_dir.split("/")[-1] / "manifest.json").read_text()
    )
    return summary, lines, manifest


def test_run_writes_manifest_and_complete_step_schema(tmp_path):
    config = make_config()
    summary, lines, manifest = do_run(tmp_path, "a")

    assert manifest["config_hash"] == config.config_hash()
    assert manifest["config"]["seed"] == 11
    assert manifest["config"]["schedule"]["version"] == "test-v1"

    steps = [r for r in lines if r["type"] == "step"]
    episodes = [r for r in lines if r["type"] == "episode"]
    assert steps and len(episodes) == 3
    for record in steps:
        assert set(record) == STEP_FIELDS
        assert record["mode"] == "act"
        assert record["prediction_error"] is None
    assert summary.episodes == 3


def test_step_sums_equal_episode_totals(tmp_path):
    _, lines, _ = do_run(tmp_path, "a")
    steps = [r for r in lines if r["type"] == "step"]
    for ep_record in (r for r in lines if r["type"] == "episode"):
        ep_steps = [s for s in steps if s["episode"] == ep_record["episode"]]
        assert ep_record["steps_used"] == len(ep_steps)
        for field in ("llm_calls", "input_tokens", "output_tokens"):
            assert ep_record[field] == sum(s[field] for s in ep_steps)
        assert math.isclose(
            ep_record["cost_usd"], sum(s["cost_usd"] for s in ep_steps)
        )
        assert ep_record["success"] == ep_steps[-1]["success"]


def test_shift_records_land_at_scheduled_global_steps(tmp_path):
    summary, lines, _ = do_run(tmp_path, "a")
    shift_steps = [r["global_step"] for r in lines if r["type"] == "shift"]
    total_env_steps = sum(
        r["steps_used"] for r in lines if r["type"] == "episode"
    )
    expected = [s for s in (10, 40) if s <= total_env_steps]
    assert shift_steps == expected
    assert summary.shifts == len(expected)
    for record in (r for r in lines if r["type"] == "shift"):
        assert record["requested_distance"] in (3, 5)
        assert record["actual_distance"] >= 1


def test_costs_match_price_table_exactly(tmp_path):
    prices = PriceTable()
    _, lines, _ = do_run(tmp_path, "a")
    for record in (r for r in lines if r["type"] == "step"):
        assert record["cost_usd"] == prices.cost_usd(
            record["input_tokens"], record["output_tokens"]
        )
        assert record["llm_calls"] == 1  # reactive: exactly one call per step


def test_reproducibility_same_config_and_seed(tmp_path):
    def strip(lines, manifest):
        cleaned = []
        for record in lines:
            cleaned.append({k: v for k, v in record.items() if k not in VOLATILE})
        stable_manifest = {
            k: v for k, v in manifest.items() if k not in ("run_id", "created_at")
        }
        return cleaned, stable_manifest

    _, lines_a, manifest_a = do_run(tmp_path, "a")
    _, lines_b, manifest_b = do_run(tmp_path, "b")
    assert strip(lines_a, manifest_a) == strip(lines_b, manifest_b)


def test_config_hash_stable_and_sensitive(tmp_path):
    assert make_config(seed=1).config_hash() == make_config(seed=1).config_hash()
    assert make_config(seed=1).config_hash() != make_config(seed=2).config_hash()
