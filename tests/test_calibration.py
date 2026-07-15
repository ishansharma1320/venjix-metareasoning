from venjix.calibration import (
    Case,
    bootstrap_ci,
    build_report,
    clipped_next,
    generate_cases,
    run_probe,
    verdict,
)
from venjix.llm import MockModel

SIZES = [7, 11, 15]


def test_generation_is_deterministic_and_quotas_hold():
    a = generate_cases(SIZES, 500, seed=0)
    b = generate_cases(SIZES, 500, seed=0)
    assert a == b
    assert a != generate_cases(SIZES, 500, seed=1)
    strata = [c.stratum for c in a]
    assert strata.count("interior_move") == 200
    assert strata.count("edge_move") == 150
    assert strata.count("probe") == 150


def test_case_invariants():
    for case in generate_cases(SIZES, 400, seed=3):
        assert case.truth_next == clipped_next(case.size, case.pos, case.action)
        assert case.true_goal != case.pos  # standing on the goal is unreachable
        if case.stratum == "edge_move":
            assert case.truth_next == case.pos  # genuinely clips
            assert case.edge
        if case.stratum == "interior_move":
            assert case.truth_next != case.pos  # never clips
        if case.belief == "known":
            assert case.believed_goal == case.true_goal  # calm: never stale
        else:
            assert case.believed_goal is None
            assert case.truth_reward == 0  # unknowable reward never scored
        assert case.truth_reward == int(case.truth_next == case.true_goal)


def test_reward_positive_cases_are_exercised():
    cases = generate_cases(SIZES, 500, seed=0)
    assert sum(c.truth_reward for c in cases) > 20  # enrichment worked


def test_end_to_end_mock_is_a_perfect_world_model():
    cases = generate_cases(SIZES, 200, seed=0)
    results, usage = run_probe(cases, lambda: MockModel(seed=0), workers=4)
    assert len(results) == 200
    assert [r.case.idx for r in results] == list(range(200))  # reassembled in order
    assert all(not r.mispredicted for r in results)
    assert all(not r.parse_error for r in results)
    assert usage["llm_calls"] == 200

    report = build_report(results, usage, model="mock", backend="mock", seed=0, resamples=200)
    assert report["overall"]["misprediction_rate"] == 0.0
    assert report["verdict"] == "GREEN"
    assert report["parse_errors"]["count"] == 0


def test_garbage_replies_score_exactly_like_agents_experience_them():
    edge = Case(
        idx=0, size=5, pos=(0, 2), action="up", believed_goal=None,
        true_goal=(4, 4), truth_next=(0, 2), truth_reward=0,
        stratum="edge_move", edge=True, belief="unknown",
    )
    interior = Case(
        idx=1, size=5, pos=(2, 2), action="down", believed_goal=None,
        true_goal=(4, 4), truth_next=(3, 2), truth_reward=0,
        stratum="interior_move", edge=False, belief="unknown",
    )
    results, usage = run_probe(
        [edge, interior],
        lambda: MockModel(scripted=["garbage", "garbage"]),
        workers=1,
    )
    assert all(r.parse_error for r in results)
    # fallback = stay-in-place: correct for the clip case, wrong for the move
    assert results[0].mispredicted is False
    assert results[1].mispredicted is True
    # classification: short garbage = format drift (extraction), not truncation
    assert all(r.error_kind == "extraction" for r in results)
    assert all(r.raw_text == "garbage" for r in results)
    from venjix.calibration import classify_parse_error

    assert classify_parse_error(results[0], usage.get("max_tokens_per_call")) == (
        "format_drift"
    )


def test_classification_separates_out_of_range_and_truncation():
    from venjix.calibration import classify_parse_error

    base = Case(
        idx=0, size=5, pos=(2, 2), action="down", believed_goal=None,
        true_goal=(4, 4), truth_next=(3, 2), truth_reward=0,
        stratum="interior_move", edge=False, belief="unknown",
    )

    def result(error_kind, output_tokens):
        from venjix.calibration import CaseResult

        return CaseResult(
            case=base, pred_next=(2, 2), pred_reward=0, parse_error=True,
            mispredicted=True, error_kind=error_kind, raw_text="x",
            output_tokens=output_tokens,
        )

    assert classify_parse_error(result("out_of_range", 10), 256) == "out_of_range"
    assert classify_parse_error(result("extraction", 256), 256) == "truncation"
    assert classify_parse_error(result("extraction", 12), 256) == "format_drift"


def test_bootstrap_ci_brackets_rate_and_is_seeded():
    values = [1] * 20 + [0] * 80
    low_a, high_a = bootstrap_ci(values, seed=7, resamples=2000)
    low_b, high_b = bootstrap_ci(values, seed=7, resamples=2000)
    assert (low_a, high_a) == (low_b, high_b)
    assert low_a <= 0.2 <= high_a
    assert 0 < low_a < high_a < 1


def test_verdict_bands():
    assert verdict(0.0) == "GREEN"
    assert verdict(0.149) == "GREEN"
    assert verdict(0.15) == "YELLOW"
    assert verdict(0.249) == "YELLOW"
    assert verdict(0.25) == "RED"
    assert verdict(0.9) == "RED"
