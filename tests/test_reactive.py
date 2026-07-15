import pytest

from venjix.agents import FALLBACK_ACTION, ReactiveAgent, parse_action
from venjix.config import GridworldConfig
from venjix.gridworld import Gridworld
from venjix.llm import MockModel


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("up", "up"),
        ("I will move DOWN now.", "down"),
        ("action: left", "left"),
        ("Go right, then reassess.", "right"),
        ("Best to probe here.", "probe"),
        ("upward is not a word match, so probe", "probe"),  # \b guards 'upward'
        ("right or left? I choose right first... no, left", "right"),  # first wins
    ],
)
def test_parse_action_variants(text, expected):
    action, parse_error = parse_action(text)
    assert (action, parse_error) == (expected, False)


def test_parse_action_garbage_falls_back_to_probe():
    action, parse_error = parse_action("I cannot decide on anything.")
    assert (action, parse_error) == (FALLBACK_ACTION, True)


def test_prompt_contains_current_observation():
    captured = []

    class Spy(MockModel):
        def _complete(self, prompt, response_regex=None):
            captured.append(prompt)
            return super()._complete(prompt, response_regex)

    config = GridworldConfig(size=5, goal=(4, 4), probe_radius=2)
    env = Gridworld(config, seed=0)
    agent = ReactiveAgent(Spy(seed=0), config)
    obs = env.reset()
    obs = env.step("down")
    agent.choose(obs)
    prompt = captured[0]
    assert "(1, 0)" in prompt  # current position
    assert "Steps used: 1" in prompt
    assert "5x5" in prompt and "probe" in prompt


def test_one_llm_call_per_choose():
    config = GridworldConfig(size=5, goal=(4, 4))
    env = Gridworld(config, seed=0)
    client = MockModel(seed=0)
    agent = ReactiveAgent(client, config)
    obs = env.reset()
    for expected_calls in range(1, 6):
        decision = agent.choose(obs)
        assert client.total_calls == expected_calls
        assert decision.mode == "act"
        obs = env.step(decision.action)
        if obs.done:
            obs = env.reset()


def test_garbage_response_yields_probe_decision():
    config = GridworldConfig(size=5, goal=(4, 4))
    env = Gridworld(config, seed=0)
    agent = ReactiveAgent(MockModel(scripted=["blah blah"]), config)
    decision = agent.choose(env.reset())
    assert decision.action == FALLBACK_ACTION
    assert decision.parse_error is True
    assert decision.mode == "act"
