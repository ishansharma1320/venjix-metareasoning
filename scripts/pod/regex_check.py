"""Constrained-decoding field check against a local vLLM (see Amendment 9).

vLLM 0.29 accepts the legacy `guided_regex` field but silently ignores it; only
`structured_outputs` masks logits. Prints one line per field; the pod
preparation script requires the structured_outputs line to be HTTP 200.
"""

import json
import urllib.error
import urllib.request

PROMPT = ("PREDICT\nGRID: 7\nPOSITION: (3, 3)\nACTION: right\nBELIEVED_GOAL: (3, 4)\n"
          "Reply exactly as: NEXT: (r, c) REWARD: 0|1")
RX = r"NEXT: \([0-9]{1,2}, [0-9]{1,2}\) REWARD: [01]"


def call(extra):
    body = {"model": "Qwen/Qwen3-8B", "max_tokens": 256, "temperature": 0,
            "messages": [{"role": "user", "content": PROMPT}],
            "chat_template_kwargs": {"enable_thinking": False}, **extra}
    r = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
                               data=json.dumps(body).encode(),
                               headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=60) as raw:
            p = json.load(raw)
        m = p["choices"][0]["message"]
        return "200", repr(m.get("content")), p["usage"]["completion_tokens"]
    except urllib.error.HTTPError as e:
        return str(e.code), e.read()[:200].decode(errors="replace"), None


for name, extra in [("none", {}), ("guided_regex", {"guided_regex": RX}),
                    ("structured_outputs", {"structured_outputs": {"regex": RX}}),
                    ("structured_outputs_enforced", {"structured_outputs": {"regex": r"ZEBRA-[0-9]{3}"}})]:
    print(name, "->", call(extra))
