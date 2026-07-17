import io
import json
import urllib.error
from unittest.mock import patch

import pytest

from venjix.agents import parse_action
from venjix.llm import OpenAICompatibleClient


def fake_response(payload):
    class Ctx:
        def __enter__(self):
            return io.BytesIO(json.dumps(payload).encode())

        def __exit__(self, *exc):
            return False

    return Ctx()


def payload_with(content, reasoning_content=None):
    message = {"content": content}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    return {
        "choices": [{"message": message}],
        "usage": {"prompt_tokens": 42, "completion_tokens": 7},
    }


def call(client, prompt, content, reasoning_content=None):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        captured["auth"] = request.get_header("Authorization")
        return fake_response(payload_with(content, reasoning_content))

    with patch("venjix.llm.urllib.request.urlopen", fake_urlopen):
        response = client.complete(prompt)
    return response, captured


def test_request_shape_and_usage_mapping():
    client = OpenAICompatibleClient(
        "Qwen/Qwen3-4B", base_url="http://gpu-box:8000", api_key="k"
    )
    response, captured = call(client, "PROMPT", "I choose down.")
    assert captured["url"] == "http://gpu-box:8000/v1/chat/completions"
    assert captured["auth"] == "Bearer k"
    assert captured["body"]["model"] == "Qwen/Qwen3-4B"
    assert captured["body"]["temperature"] == 0
    assert captured["body"]["messages"] == [{"role": "user", "content": "PROMPT"}]
    assert response.text == "I choose down."
    assert (response.input_tokens, response.output_tokens) == (42, 7)
    assert client.total_calls == 1 and client.total_input_tokens == 42


def test_think_trace_is_stripped_before_parsing():
    client = OpenAICompatibleClient("m", base_url="http://x", api_key="k")
    response, _ = call(
        client,
        "p",
        "<think>maybe up? or right? going with left in the end</think>action: left",
    )
    assert response.text == "action: left"
    assert parse_action(response.text) == ("left", False)


def test_unclosed_think_trace_from_truncation_is_dropped():
    client = OpenAICompatibleClient("m", base_url="http://x", api_key="k")
    response, _ = call(
        client, "p", "<think>maybe up, or down, or maybe I should go right and"
    )
    assert response.text == ""  # truncated reasoning can't reach the parser
    assert parse_action(response.text) == ("probe", True)


def test_thinking_disabled_in_request():
    client = OpenAICompatibleClient("m", base_url="http://x", api_key="k")
    _, captured = call(client, "p", "up")
    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_response_regex_becomes_guided_decoding():
    client = OpenAICompatibleClient("m", base_url="http://x", api_key="k")
    _, captured = call(client, "p", "up")
    assert "guided_regex" not in captured["body"]  # plain calls unconstrained

    captured2 = {}

    def fake_urlopen(request, timeout=None):
        captured2["body"] = json.loads(request.data)
        return fake_response(payload_with("NEXT: (1, 2) REWARD: 0"))

    with patch("venjix.llm.urllib.request.urlopen", fake_urlopen):
        client.complete("p", response_regex=r"NEXT: \(\d+, \d+\) REWARD: [01]")
    assert captured2["body"]["guided_regex"] == r"NEXT: \(\d+, \d+\) REWARD: [01]"


def make_429(retry_after=None):
    import email.message

    headers = email.message.Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError(
        "http://x", 429, "Too Many Requests", headers, io.BytesIO(b'{"error":"rate"}')
    )


def run_with_429s(retry_after, monkeypatch):
    """First call 429s with the given Retry-After; second succeeds.
    Returns the delays slept."""
    slept = []
    monkeypatch.setattr("venjix.llm.time.sleep", lambda s: slept.append(s))
    client = OpenAICompatibleClient("m", base_url="http://x", api_key="k")
    state = {"calls": 0}

    def fake_urlopen(request, timeout=None):
        state["calls"] += 1
        if state["calls"] == 1:
            raise make_429(retry_after)
        return fake_response(payload_with("up"))

    with patch("venjix.llm.urllib.request.urlopen", fake_urlopen):
        response = client.complete("p")
    assert response.text == "up"
    return slept


def test_retry_after_zero_never_rehammers_instantly(monkeypatch):
    slept = run_with_429s("0", monkeypatch)
    assert len(slept) == 1 and slept[0] >= 1.0  # floored, no tight loop


def test_retry_after_date_string_falls_back_to_backoff(monkeypatch):
    slept = run_with_429s("Wed, 15 Jul 2026 21:00:00 GMT", monkeypatch)
    assert len(slept) == 1 and 1.0 <= slept[0] <= 60.0  # no ValueError crash


def test_retry_after_numeric_is_honored_and_capped(monkeypatch):
    assert run_with_429s("7", monkeypatch) == [7.0]
    assert run_with_429s("500", monkeypatch) == [60.0]  # capped


def test_non_retryable_http_error_raises(monkeypatch):
    client = OpenAICompatibleClient("m", base_url="http://x", api_key="k")

    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError("http://x", 401, "Unauthorized", None, io.BytesIO(b""))

    with patch("venjix.llm.urllib.request.urlopen", fake_urlopen):
        with pytest.raises(urllib.error.HTTPError):
            client.complete("p")


def test_null_content_yields_empty_text():
    client = OpenAICompatibleClient("m", base_url="http://x", api_key="k")
    response, _ = call(client, "p", None)
    assert response.text == ""
    assert parse_action(response.text) == ("probe", True)  # safe fallback path


def test_reasoning_parser_misattribution_falls_back_to_reasoning_content():
    # Observed live: vLLM reasoning parsers return content=null with the whole
    # (non-thinking) answer in reasoning_content.
    client = OpenAICompatibleClient("m", base_url="http://x", api_key="k")
    response, _ = call(client, "p", None, reasoning_content="probe")
    assert response.text == "probe"


def test_content_wins_over_reasoning_content():
    client = OpenAICompatibleClient("m", base_url="http://x", api_key="k")
    response, _ = call(client, "p", "down", reasoning_content="ignore this up")
    assert response.text == "down"


def test_openrouter_serving_registry(monkeypatch):
    from venjix.llm import make_client

    monkeypatch.setenv("OPENROUTER_API_KEY", "k")

    gemma = make_client("openrouter", "google/gemma-3-4b-it")
    assert gemma.extra_body["provider"] == {"only": ["DeepInfra"]}
    assert "reasoning" not in gemma.extra_body  # no reasoning control -> omit
    assert gemma.vllm_dialect is False

    qwen = make_client("openrouter", "Qwen/Qwen3-8B")  # HF id -> slug
    assert qwen.model == "qwen/qwen3-8b"
    assert qwen.extra_body["provider"] == {"only": ["Alibaba"]}
    assert qwen.extra_body["reasoning"] == {"enabled": False}

    import pytest as _pytest

    with _pytest.raises(ValueError, match="serving registry"):
        make_client("openrouter", "some/unpinned-model")
