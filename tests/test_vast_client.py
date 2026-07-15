import io
import json
import ssl
import urllib.error
from unittest.mock import patch

import pytest

from venjix.llm import VastServerlessClient

ROUTE = {"url": "https://worker-7.vast.example:8443/", "signature": "sig", "cost": 256}
CHAT = {
    "choices": [{"message": {"content": "<think>hmm</think>action: up"}}],
    "usage": {"prompt_tokens": 33, "completion_tokens": 9},
}


def fake_response(payload):
    class Ctx:
        def __enter__(self):
            return io.BytesIO(json.dumps(payload).encode())

        def __exit__(self, *exc):
            return False

    return Ctx()


def make_client(**kwargs):
    return VastServerlessClient(
        "Qwen/Qwen3-4B", endpoint="qwen-llm", api_key="vk", **kwargs
    )


def dispatching_urlopen(calls, fail_first_route=False):
    def fake_urlopen(request, timeout=None, context=None):
        body = json.loads(request.data)
        calls.append({"url": request.full_url, "body": body, "context": context})
        if request.full_url == VastServerlessClient.ROUTE_URL:
            if fail_first_route and len([c for c in calls if c["url"] == request.full_url]) == 1:
                raise urllib.error.URLError("transient route failure")
            return fake_response(ROUTE)
        return fake_response(CHAT)

    return fake_urlopen


def test_two_hop_protocol_and_envelope():
    calls = []
    client = make_client(max_tokens=256)
    with patch("venjix.llm.urllib.request.urlopen", dispatching_urlopen(calls)):
        response = client.complete("PROMPT")

    route_call, worker_call = calls
    assert route_call["url"] == "https://run.vast.ai/route/"
    assert route_call["body"] == {"endpoint": "qwen-llm", "api_key": "vk", "cost": 256}
    assert route_call["context"] is None  # route hop verifies TLS normally

    assert worker_call["url"] == "https://worker-7.vast.example:8443/v1/chat/completions"
    assert worker_call["body"]["auth_data"] == ROUTE  # signed route passed as-is
    payload = worker_call["body"]["payload"]
    assert payload["model"] == "Qwen/Qwen3-4B"
    assert payload["temperature"] == 0
    assert payload["messages"] == [{"role": "user", "content": "PROMPT"}]
    # worker hop: self-signed cert -> verification disabled
    assert worker_call["context"].verify_mode == ssl.CERT_NONE

    assert response.text == "action: up"  # think trace stripped
    assert (response.input_tokens, response.output_tokens) == (33, 9)
    assert client.total_calls == 1


def test_fresh_route_per_call():
    calls = []
    client = make_client()
    with patch("venjix.llm.urllib.request.urlopen", dispatching_urlopen(calls)):
        client.complete("a")
        client.complete("b")
    route_calls = [c for c in calls if c["url"] == VastServerlessClient.ROUTE_URL]
    assert len(route_calls) == 2  # one-time signatures: never reused


def test_transient_failure_retries(monkeypatch):
    monkeypatch.setattr("venjix.llm.time.sleep", lambda s: None)
    calls = []
    client = make_client()
    with patch(
        "venjix.llm.urllib.request.urlopen",
        dispatching_urlopen(calls, fail_first_route=True),
    ):
        response = client.complete("PROMPT")
    assert response.text == "action: up"  # second attempt succeeded


def test_exhausted_retries_raise(monkeypatch):
    monkeypatch.setattr("venjix.llm.time.sleep", lambda s: None)
    client = make_client(retries=2)

    def always_fail(request, timeout=None, context=None):
        raise urllib.error.URLError("down")

    with patch("venjix.llm.urllib.request.urlopen", always_fail):
        with pytest.raises(RuntimeError, match="after 2 attempts"):
            client.complete("PROMPT")


def test_missing_api_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    with pytest.raises(ValueError, match="VAST_API_KEY"):
        VastServerlessClient("m", endpoint="e")
