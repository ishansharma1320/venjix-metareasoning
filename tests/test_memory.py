from venjix.memory import EpisodicLog, Experience


def exp(pos, action="up", next_pos=None, reward=0, probe_result=None):
    return Experience(pos, action, next_pos or pos, reward, probe_result)


def test_append_only_ordering():
    log = EpisodicLog()
    entries = [exp((0, i)) for i in range(5)]
    for e in entries:
        log.append(e)
    assert list(log.entries) == entries
    assert len(log) == 5


def test_cold_start_has_no_belief():
    assert EpisodicLog().believed_goal() is None


def test_belief_from_reward_evidence():
    log = EpisodicLog()
    log.append(exp((1, 1)))
    log.append(exp((2, 3), action="down", next_pos=(3, 3), reward=1))
    assert log.believed_goal() == (3, 3)


def test_belief_from_probe_offset():
    log = EpisodicLog()
    log.append(exp((2, 2), action="probe", probe_result=(1, -2)))
    assert log.believed_goal() == (3, 0)


def test_out_of_range_probe_is_not_evidence():
    log = EpisodicLog()
    log.append(exp((2, 2), action="probe", probe_result="out_of_range"))
    assert log.believed_goal() is None


def test_most_recent_evidence_wins():
    log = EpisodicLog()
    log.append(exp((0, 0), next_pos=(0, 1), reward=1))
    log.append(exp((4, 4), action="probe", probe_result=(0, 1)))
    assert log.believed_goal() == (4, 5)
    log.append(exp((2, 2), next_pos=(2, 3), reward=1))
    assert log.believed_goal() == (2, 3)
