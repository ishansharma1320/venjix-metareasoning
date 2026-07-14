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


def test_later_rewardless_visit_disproves_evidence():
    log = EpisodicLog()
    log.append(exp((3, 1), next_pos=(3, 2), reward=1))  # goal seen at (3, 2)
    assert log.believed_goal() == (3, 2)
    log.append(exp((3, 1), action="right", next_pos=(3, 2), reward=0))  # revisit: empty
    assert log.believed_goal() is None


def test_earlier_visit_does_not_disprove_later_evidence():
    log = EpisodicLog()
    log.append(exp((3, 1), action="right", next_pos=(3, 2), reward=0))
    log.append(exp((3, 1), next_pos=(3, 2), reward=1))
    assert log.believed_goal() == (3, 2)


def test_disproof_falls_back_to_older_undisproven_evidence_and_new_evidence_restores():
    log = EpisodicLog()
    log.append(exp((0, 0), action="probe", probe_result=(4, 4)))  # evidence: (4, 4)
    log.append(exp((3, 1), next_pos=(3, 2), reward=1))  # newer evidence: (3, 2)
    log.append(exp((3, 1), action="right", next_pos=(3, 2), reward=0))  # disproves (3, 2)
    assert log.believed_goal() == (4, 4)  # older evidence still stands
    log.append(exp((2, 2), action="probe", probe_result=(3, 3)))  # fresh evidence
    assert log.believed_goal() == (5, 5)
