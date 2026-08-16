from types import SimpleNamespace

from room_presence import learner_is_in_room


def participant(identity):
    return SimpleNamespace(identity=identity)


def test_exact_learner_identity_is_present_in_livekit_roster():
    room = SimpleNamespace(remote_participants={
        "learner": participant("S-1"),
        "other": participant("S-2"),
    })

    assert learner_is_in_room(room, "S-1") is True
    assert learner_is_in_room(room, "S-3") is False


def test_missing_or_unusual_rosters_fail_closed_without_crashing():
    assert learner_is_in_room(SimpleNamespace(), "S-1") is False
    assert learner_is_in_room(
        SimpleNamespace(remote_participants=[participant("S-1")]),
        "S-1",
    ) is True
