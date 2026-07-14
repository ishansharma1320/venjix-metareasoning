import json

from venjix.experiments import DEFAULT_SET, Condition, conditions, load_spec
from venjix.llm import MockModel
from venjix.runner import run


def test_frozen_set_loads_and_registers_the_bandit():
    spec = load_spec()
    assert spec["version"] == "exp-v1"
    assert "bandit" in spec["agents"]  # registered before it exists
    assert spec["n_seeds"] == 20 and spec["n_episodes"] == 40
    assert len(spec["regimes"]) == 5


def test_all_conditions_materialize_and_are_unique():
    spec = load_spec()
    all_conditions = conditions(spec)
    # 5 regimes x 6 agents x 20 seeds — registered == implemented since rung 4
    assert len(all_conditions) == 5 * 6 * 20
    assert sorted({c.agent for c in all_conditions}) == sorted(spec["agents"])
    assert all(isinstance(c, Condition) for c in all_conditions)
    hashes = {c.config.config_hash() for c in all_conditions}
    assert len(hashes) == len(all_conditions)  # every condition distinct


def test_shifts_land_within_the_plausible_horizon():
    spec = load_spec()
    for regime in spec["regimes"]:
        max_steps = spec["n_episodes"] * 4 * regime["env"]["size"]  # budget ceiling
        for step, distance in regime["shifts"]:
            assert 0 < step < max_steps
            assert 1 <= distance <= 2 * (regime["env"]["size"] - 1)


def test_signal_params_match_the_recorded_amendment():
    params = load_spec()["agent_params"]
    assert params["ewma_alpha"] == 0.3
    assert params["pe_threshold"] == 0.25
    assert params["pe_threshold"] < params["ewma_alpha"]  # single hit must fire
    # Amendment 5: bandit hyperparams registered pre-first-run.
    assert params["ucb_alpha"] == 1.0
    assert params["cost_weight"] == 100.0


def test_one_condition_runs_end_to_end(tmp_path):
    spec = load_spec()
    condition = next(
        c
        for c in conditions(spec, agents=["heuristic"])
        if c.regime == "easy-7" and c.seed == 0
    )
    summary = run(condition.config, MockModel(seed=0), tmp_path)
    assert summary.episodes == 40
    assert summary.shifts >= 1  # the schedule actually fired
    manifest = json.loads((tmp_path / summary.run_dir.split("/")[-1] / "manifest.json").read_text())
    assert manifest["config"]["schedule"]["version"] == "exp-v1/easy-7"


def test_default_set_path_points_at_committed_file():
    assert DEFAULT_SET.exists()
    assert DEFAULT_SET.name == "exp-v1.json"
