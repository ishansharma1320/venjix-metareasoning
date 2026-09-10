"""Amendment 11: thinking budget on policy calls (two-phase Qwen3 recipe) and
hash compatibility of the new RunConfig field."""

import io
import json
from unittest.mock import patch

import pytest

from venjix.config import GridworldConfig, RunConfig, ShiftEntry, ShiftSchedule
from venjix.experiments import conditions, load_spec
from venjix.llm import THINK_CLOSE, OpenAICompatibleClient, make_client


def _resp(content, prompt_tokens=10, completion_tokens=5):
    class Ctx:
        def __enter__(self):
            return io.BytesIO(json.dumps({
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
            }).encode())

        def __exit__(self, *exc):
            return False
    return Ctx()


def _client(budget=512):
    return OpenAICompatibleClient("m", base_url="http://x", api_key="k", thinking_budget=budget)


def test_finished_thinking_is_one_call():
    client = _client()
    bodies = []

    def fake(request, timeout=None):
        bodies.append(json.loads(request.data))
        return _resp("<think>\nshort reasoning\n</think>\n\nright", 10, 20)

    with patch("venjix.llm.urllib.request.urlopen", fake):
        r = client.complete("policy prompt")
    assert r.text == "right"
    assert len(bodies) == 1
    assert bodies[0]["chat_template_kwargs"] == {"enable_thinking": True}
    assert bodies[0]["max_tokens"] == 512 + 16
    assert (r.input_tokens, r.output_tokens) == (10, 20)
    assert client.total_forced_answers == 0


def test_budget_hit_forces_answer_in_phase_two_and_sums_usage():
    client = _client(budget=8)
    bodies = []

    def fake(request, timeout=None):
        body = json.loads(request.data)
        bodies.append(body)
        if len(bodies) == 1:
            return _resp("<think>\nstill thinking about whether to go", 10, 24)  # truncated
        return _resp("down", 40, 2)

    with patch("venjix.llm.urllib.request.urlopen", fake):
        r = client.complete("policy prompt")
    assert r.text == "down"
    assert len(bodies) == 2
    p2 = bodies[1]
    assert p2["continue_final_message"] is True and p2["add_generation_prompt"] is False
    assert p2["messages"][-1]["role"] == "assistant"
    assert p2["messages"][-1]["content"].endswith(THINK_CLOSE)
    assert p2["messages"][-1]["content"].startswith("<think>\nstill thinking")
    assert (r.input_tokens, r.output_tokens) == (50, 26)  # both phases counted
    assert client.total_forced_answers == 1
    assert client.total_output_tokens == 26


def test_regex_calls_never_think():
    client = _client()
    bodies = []

    def fake(request, timeout=None):
        bodies.append(json.loads(request.data))
        return _resp("NEXT: (1, 2) REWARD: 0")

    with patch("venjix.llm.urllib.request.urlopen", fake):
        client.complete("PREDICT ...", response_regex=r"NEXT: .*")
    assert len(bodies) == 1
    assert bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert bodies[0]["structured_outputs"] == {"regex": r"NEXT: .*"}


def test_no_think_block_is_passed_through():
    client = _client()
    with patch("venjix.llm.urllib.request.urlopen", lambda req, timeout=None: _resp("left")):
        assert client.complete("p").text == "left"
    assert client.total_forced_answers == 0


def test_thinking_budget_rejected_off_vllm():
    with pytest.raises(ValueError):
        OpenAICompatibleClient("m", base_url="http://x", api_key="k",
                               thinking_budget=512, vllm_dialect=False)
    with pytest.raises(ValueError):
        make_client("openrouter", "Qwen/Qwen3-8B", thinking_budget=512)


def test_config_hash_unchanged_when_budget_is_none():
    # A real exp-v1 hash recorded in a run manifest before the field existed
    # (runs/exp-v1/*/manifest.json, hard-15-far/simulate/seed9).
    known = "04ca382b31e44aa60f0127c09ab4bff3a6d9e94234d3471814a9967527a0a7a2"
    spec = load_spec("experiments/exp-v1.json")
    c = [c for c in conditions(spec) if c.regime == "hard-15-far" and c.agent == "simulate" and c.seed == 9][0]
    assert c.config.policy_thinking_budget is None
    assert "policy_thinking_budget" not in c.config.to_dict()
    assert c.config.config_hash() == known


def test_exp_v2_hashes_differ_and_carry_budget():
    v1 = {c.config.config_hash() for c in conditions(load_spec("experiments/exp-v1.json"))}
    v2 = list(conditions(load_spec("experiments/exp-v2.json")))
    assert len(v2) == 600 and all(c.config.policy_thinking_budget == 512 for c in v2)
    assert not ({c.config.config_hash() for c in v2} & v1)


def test_budget_validation():
    env = GridworldConfig(size=7, probe_radius=3)
    sched = ShiftSchedule(version="t/r", entries=(ShiftEntry(4, 3),))
    with pytest.raises(ValueError):
        RunConfig(env=env, schedule=sched, seed=0, n_episodes=1, policy_thinking_budget=0)
