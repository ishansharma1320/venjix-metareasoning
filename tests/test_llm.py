import pytest

from venjix.config import PriceTable
from venjix.llm import MockModel


def test_mock_determinism_across_instances():
    a, b = MockModel(seed=7), MockModel(seed=7)
    for prompt in ("where to?", "current position: (1, 2)", "x"):
        assert a.complete(prompt) == b.complete(prompt)


def test_mock_varies_with_seed_or_prompt():
    responses = {
        MockModel(seed=s).complete(f"prompt {p}").text
        for s in range(5)
        for p in range(5)
    }
    assert len(responses) > 1


def test_mock_answers_contain_a_legal_action():
    model = MockModel(seed=3)
    for i in range(20):
        text = model.complete(f"prompt {i}").text
        assert any(action in text for action in MockModel.ACTIONS)


def test_cumulative_usage_counters():
    model = MockModel(seed=0)
    totals_in = totals_out = 0
    for i in range(4):
        response = model.complete("p" * (10 * (i + 1)))
        totals_in += response.input_tokens
        totals_out += response.output_tokens
    assert model.total_calls == 4
    assert model.total_input_tokens == totals_in
    assert model.total_output_tokens == totals_out


def test_scripted_queue():
    model = MockModel(scripted=["go up", "then down"])
    assert model.complete("a").text == "go up"
    assert model.complete("b").text == "then down"
    with pytest.raises(RuntimeError):
        model.complete("c")


def test_price_table_math():
    prices = PriceTable(input_per_mtok_usd=1.0, output_per_mtok_usd=5.0)
    assert prices.cost_usd(1_000_000, 0) == 1.0
    assert prices.cost_usd(0, 1_000_000) == 5.0
    assert prices.cost_usd(500, 100) == (500 * 1.0 + 100 * 5.0) / 1_000_000
    assert prices.cost_usd(0, 0) == 0.0
