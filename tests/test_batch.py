import json

from venjix.batch import completed_hashes, run_batch, verify_pairing
from venjix.experiments import conditions, load_spec


def small_todo():
    spec = load_spec()
    return [
        c
        for c in conditions(spec, agents=["retrieve", "heuristic"])
        if c.regime == "easy-7" and c.seed < 2
    ]


def test_batch_runs_concurrently_resumes_and_pairing_holds(tmp_path):
    todo = small_todo()
    assert len(todo) == 4
    completed, failures = run_batch(todo, tmp_path, backend="mock", workers=4)
    assert completed == 4 and failures == []

    done = completed_hashes(tmp_path)
    assert {c.config.config_hash() for c in todo} == done  # resume skips all

    count_mismatches, prefix_divergences = verify_pairing(tmp_path)
    assert prefix_divergences == []  # Amendment 6c guarantee holds in practice
    # count mismatches are allowed (fast agents can finish before late shifts);
    # if any occur they must reference this regime's groups only
    assert all("easy-7" in m for m in count_mismatches)


def write_fake_run(root, name, agent, seed, shifts):
    run_dir = root / name
    run_dir.mkdir()
    manifest = {
        "config": {
            "schedule": {"version": "exp-v1/fake"},
            "seed": seed,
            "agent": agent,
            "n_episodes": 1,
        },
        "config_hash": name,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    lines = [
        json.dumps(
            {"type": "shift", "at_step": s, "old_goal": og, "new_goal": ng}
        )
        for s, og, ng in shifts
    ]
    lines.append(json.dumps({"type": "episode"}))
    (run_dir / "episodes.jsonl").write_text("\n".join(lines) + "\n")


def test_verify_pairing_detects_divergence_and_count_mismatch(tmp_path):
    write_fake_run(
        tmp_path, "a", "agent1", 0,
        [(10, [0, 0], [1, 1]), (20, [1, 1], [2, 2])],
    )
    write_fake_run(tmp_path, "b", "agent2", 0, [(10, [0, 0], [3, 3])])  # diverges
    count_mismatches, prefix_divergences = verify_pairing(tmp_path)
    assert len(count_mismatches) == 1 and "shift counts differ" in count_mismatches[0]
    assert len(prefix_divergences) == 1 and "agent2" in prefix_divergences[0]


def test_verify_pairing_accepts_consistent_prefix(tmp_path):
    write_fake_run(
        tmp_path, "a", "agent1", 3,
        [(10, [0, 0], [1, 1]), (20, [1, 1], [2, 2])],
    )
    write_fake_run(tmp_path, "b", "agent2", 3, [(10, [0, 0], [1, 1])])  # prefix
    count_mismatches, prefix_divergences = verify_pairing(tmp_path)
    assert prefix_divergences == []
    assert len(count_mismatches) == 1  # flagged, but not a pairing bug
