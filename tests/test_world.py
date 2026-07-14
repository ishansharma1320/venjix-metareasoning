from venjix.config import GridworldConfig
from venjix.llm import MockModel
from venjix.world import WorldModel, build_predict_prompt


def make_world(scripted=None):
    client = MockModel(seed=0, scripted=scripted)
    return WorldModel(client, GridworldConfig(size=5)), client


def test_prompt_carries_labeled_fields():
    prompt = build_predict_prompt(5, (2, 3), "down", (4, 4))
    for fragment in ("GRID: 5", "POSITION: (2, 3)", "ACTION: down", "BELIEVED_GOAL: (4, 4)"):
        assert fragment in prompt
    assert "unknown" in build_predict_prompt(5, (0, 0), "up", None)


def test_mock_predicts_clipped_dynamics():
    world, _ = make_world()
    assert world.predict((0, 0), "up", None).next_pos == (0, 0)  # clipped
    assert world.predict((0, 0), "down", None).next_pos == (1, 0)
    assert world.predict((2, 4), "right", None).next_pos == (2, 4)  # clipped
    assert world.predict((2, 2), "probe", None).next_pos == (2, 2)  # no movement


def test_mock_predicts_reward_at_believed_goal_only():
    world, _ = make_world()
    hit = world.predict((3, 4), "down", (4, 4))
    assert (hit.next_pos, hit.reward, hit.parse_error) == ((4, 4), 1, False)
    miss = world.predict((3, 4), "up", (4, 4))
    assert (miss.next_pos, miss.reward) == ((2, 4), 0)
    unknown = world.predict((3, 4), "down", None)
    assert unknown.reward == 0


def test_garbage_reply_falls_back_to_stay_in_place():
    world, _ = make_world(scripted=["no idea, sorry"])
    prediction = world.predict((2, 2), "down", (4, 4))
    assert prediction == prediction.__class__((2, 2), 0, True)


def test_out_of_bounds_reply_is_a_parse_error():
    world, _ = make_world(scripted=["NEXT: (9, 9) REWARD: 0"])
    prediction = world.predict((2, 2), "down", None)
    assert prediction.next_pos == (2, 2) and prediction.parse_error


def test_one_llm_call_per_predict():
    world, client = make_world()
    for i in range(1, 4):
        world.predict((2, 2), "down", None)
        assert client.total_calls == i
