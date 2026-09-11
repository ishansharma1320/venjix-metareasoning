"""exp-v3 M2 diagnostic (pre-registration): what does the served model make of the
MiniGrid policy prompt? Run on the pod against the live vLLM server.

Not an experiment. Output is evidence for the prompt decision at M3 and is
copied into results/exp-v3-gate/. Every candidate is re-gated with whatever
prompt is final.
"""

import json
import sys
import urllib.request

from venjix.config import GridworldConfig
from venjix.envspec import DIR_DELTA, DIRS, MINIGRID
from venjix.minigrid_env import MiniGridEmpty

cfg = GridworldConfig(size=8, step_budget=32)
model = json.load(urllib.request.urlopen("http://127.0.0.1:8000/v1/models"))["data"][0]["id"]
print("model:", model)


def ask(prompt, think=False, max_tokens=64):
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": think}}
    r = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
                               data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    return (json.load(urllib.request.urlopen(r, timeout=180))["choices"][0]["message"]["content"] or "").strip()


def explicit_prompt(o, env):
    """Alternative encoding: the three neighbouring cells in words, no ASCII view."""
    x, y = o.pos
    d = o.dir

    def cell(dx, dy):
        cx, cy = x + dx, y + dy
        if (cx, cy) == env.goal:
            return "the goal"
        return "floor" if 1 <= cx <= cfg.size - 2 and 1 <= cy <= cfg.size - 2 else "wall"

    fwd = DIR_DELTA[d]
    left = DIR_DELTA[DIRS[(DIRS.index(d) - 1) % 4]]
    right = DIR_DELTA[DIRS[(DIRS.index(d) + 1) % 4]]
    seen = "" if o.probe_result is None else f" The goal is in view at offset (dx, dy) = {o.probe_result} from you."
    return (f"You are in a {cfg.size}x{cfg.size} MiniGrid room at (x, y) = {o.pos}, facing {d} "
            f"(x grows to the right, y grows downward). The cell directly ahead of you is {cell(*fwd)}; "
            f"on your left-hand side is {cell(*left)}; on your right-hand side is {cell(*right)}.{seen} "
            f"Reaching the goal gives reward 1. Actions: left (turn 90 degrees left, no movement), "
            f"right (turn 90 degrees right, no movement), forward (move one cell ahead; walls block). "
            f"Reply with exactly one action word: left, right, or forward.")


states = []
env = MiniGridEmpty(cfg, seed=3)
o = env.reset()
states.append(("start (1,1) facing right, goal not in view", o, env))
for _ in range(5):
    o = env.step("forward")
env._goal = (6, 4)
env._place_goal()
o = env._observation(env._env.gen_obs()["image"], 0, False, False)
states.append(("(6,1) facing right: WALL ahead, goal 3 cells to the right-hand side", o, env))
env2 = MiniGridEmpty(cfg, seed=3)
o2 = env2.reset()
env2._goal = (4, 1)
env2._place_goal()
o2 = env2._observation(env2._env.gen_obs()["image"], 0, False, False)
states.append(("(1,1) facing right: goal straight ahead 3 cells", o2, env2))

for label, o, e in states:
    base = MINIGRID.policy_prompt(cfg, o)
    print(f"\n=== {label} | probe_result={o.probe_result}")
    print("  [registered-style prompt]     ", repr(ask(base)))
    comp = base.rsplit("Reply with exactly one action word", 1)[0] + (
        "Answer briefly: (1) Is the cell directly in front of you floor or wall? "
        "(2) Where is the goal relative to you, if visible?")
    print("  [comprehension of the view]   ", repr(ask(comp, max_tokens=120)))
    print("  [explicit-neighbour encoding] ", repr(ask(explicit_prompt(o, e))))
    if "--think" in sys.argv:
        t = ask(base, think=True, max_tokens=2048)
        print("  [registered prompt, thinking] ", repr(t[-80:]))
        t = ask(explicit_prompt(o, e), think=True, max_tokens=2048)
        print("  [explicit, thinking]          ", repr(t[-80:]))
