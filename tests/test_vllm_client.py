import io
import json
from unittest.mock import patch

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
        "Qwen/Qwen3-8B", base_url="http://gpu-box:8000", api_key="k"
    )
    response, captured = call(client, "PROMPT", "I choose down.")
    assert captured["url"] == "http://gpu-box:8000/v1/chat/completions"
    assert captured["auth"] == "Bearer k"
    assert captured["body"]["model"] == "Qwen/Qwen3-8B"
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
