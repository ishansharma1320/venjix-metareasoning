from venjix.agents import ThresholdHeuristicAgent
from venjix.config import GridworldConfig
from venjix.gridworld import Gridworld
from venjix.llm import MockModel
from venjix.memory import Experience

CONFIG = GridworldConfig(size=5, start=(0, 0), goal=(3, 2))


def make_agent(seed=1, client=None, **kwargs):
    return ThresholdHeuristicAgent(client or MockModel(seed=0), CONFIG, seed, **kwargs)


def seed_belief(agent, goal=(3, 2)):
    agent.memory.append(Experience((goal[0], goal[1] - 1), "right", goal, 1, None))


def run_episode(env, agent):
    obs = env.reset()
    trace = []
    while not obs.done:
        decision = agent.choose(obs)
        prev_obs = obs
        obs = env.step(decision.action)
        agent.observe(prev_obs, decision.action, obs)
        trace.append(
            (decision.mode, agent.last_prediction_error, obs.pos, agent.signal.value)
        )
    return obs, trace


# --- the two-line mode rule --------------------------------------------------


def test_mode_rule():
    env = Gridworld(CONFIG, seed=0)
    obs = env.reset()

    agent = make_agent()  # ewma 0, no belief -> act
    assert agent.choose(obs).mode == "act"

    agent = make_agent()  # ewma 0, belief -> retrieve
    seed_belief(agent)
    assert agent.choose(obs).mode == "retrieve"

    agent = make_agent()  # ewma above threshold -> gather, belief or not
    seed_belief(agent)
    agent.signal.update(True)
    agent.signal.update(True)
    assert agent.signal.value > agent.pe_threshold
    assert agent.choose(obs).mode == "gather_evidence"


def test_signal_costs_one_extra_world_model_call():
    client = MockModel(seed=0)
    agent = make_agent(client=client)
    seed_belief(agent)
    env = Gridworld(CONFIG, seed=0)
    obs = env.reset()
    agent.choose(obs)  # retrieve is free; the predict call is the signal cost
    assert client.total_calls == 1
    agent2_client = MockModel(seed=0)
    agent2 = make_agent(client=agent2_client)  # no belief -> act (1 policy call)
    agent2.choose(obs)
    assert agent2_client.total_calls == 2  # policy + predict


def test_correct_steps_score_zero_error():
    env = Gridworld(CONFIG, seed=0)
    agent = make_agent()
    seed_belief(agent)
    obs, trace = run_episode(env, agent)
    assert obs.success and obs.steps_used == 5  # retrieve navigates optimally
    assert all(err == 0 for _, err, _, _ in trace)
    assert agent.signal.value == 0.0


def test_arrival_at_stale_goal_scores_misprediction():
    env = Gridworld(CONFIG, seed=0)
    agent = make_agent()
    seed_belief(agent)
    run_episode(env, agent)  # clean success, ewma stays 0
    env.relocate_goal(4)
    _, trace = run_episode(env, agent)
    stale_arrivals = [err for _, err, pos, _ in trace if pos == (3, 2)]
    assert stale_arrivals and stale_arrivals[0] == 1  # predicted reward, got none


def test_gather_policy_alternates_probe_and_moves():
    agent = make_agent(pe_threshold=0.0)
    agent.signal.update(True)  # value stays > 0 forever -> always gather
    env = Gridworld(CONFIG, seed=0)
    obs = env.reset()
    actions = []
    for _ in range(12):
        decision = agent.choose(obs)
        assert decision.mode == "gather_evidence"
        prev_obs = obs
        obs = env.step(decision.action)
        agent.observe(prev_obs, decision.action, obs)
        actions.append(decision.action)
        if obs.done:
            obs = env.reset()
    assert all(
        not (a == "probe" and b == "probe") for a, b in zip(actions, actions[1:])
    )
    assert "probe" in actions and any(a != "probe" for a in actions)


# --- the load-bearing recovery test ------------------------------------------


def test_heuristic_recovers_after_silent_shift():
    """The full predicted mechanism: learn -> silent shift -> EWMA spike ->
    gather_evidence burst -> belief refresh -> successes resume. Retrieve-only
    provably cannot do this (its test asserts permanent failure)."""
    env = Gridworld(CONFIG, seed=0)
    agent = make_agent(seed=1)

    # Learn the goal (act-mode exploration until first success).
    for _ in range(50):
        obs, _ = run_episode(env, agent)
        if obs.success:
            break
    assert obs.success
    assert agent.memory.believed_goal() == (3, 2)

    # Steady state: pure retrieve, optimal, EWMA at 0.
    obs, trace = run_episode(env, agent)
    assert obs.success and obs.steps_used == 5
    assert {mode for mode, _, _, _ in trace} == {"retrieve"}
    assert agent.signal.value == 0.0

    # Silent shift.
    old_goal, new_goal, _ = env.relocate_goal(4)

    # Recovery: within a handful of episodes the agent must spike, gather,
    # re-find the goal, and succeed again.
    modes_seen = set()
    peak_ewma = 0.0
    recovered_episode = None
    for episode in range(10):
        obs, trace = run_episode(env, agent)
        modes_seen |= {mode for mode, _, _, _ in trace}
        peak_ewma = max(peak_ewma, max(ewma for _, _, _, ewma in trace))
        if obs.success:
            recovered_episode = episode
            break

    assert recovered_episode is not None, "never recovered after the shift"
    assert "gather_evidence" in modes_seen  # the error spike bought evidence
    assert peak_ewma > agent.pe_threshold  # the signal actually fired
    assert agent.memory.believed_goal() == new_goal  # belief refreshed
    assert agent.memory.believed_goal() != old_goal

    # And the steady state after recovery is cheap retrieval again, with the
    # signal spent back down below threshold.
    obs, trace = run_episode(env, agent)
    assert obs.success
    assert trace[-1][0] == "retrieve"
    assert agent.signal.value <= agent.pe_threshold
