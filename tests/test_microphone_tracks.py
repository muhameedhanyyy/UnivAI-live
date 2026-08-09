from types import SimpleNamespace

from microphone_tracks import (
    existing_learner_microphones,
    is_learner_microphone,
    track_key,
)


AUDIO = 1
MICROPHONE = 2
SCREEN_AUDIO = 4


def publication(track, source=MICROPHONE):
    return SimpleNamespace(track=track, source=source)


def participant(identity, *publications):
    return SimpleNamespace(
        identity=identity,
        track_publications={str(index): item for index, item in enumerate(publications)},
    )


def test_existing_track_is_found_when_subscription_event_was_missed():
    mic = SimpleNamespace(sid="TR_MIC", kind=AUDIO)
    screen = SimpleNamespace(sid="TR_SCREEN", kind=AUDIO)
    learner = participant(
        "S-1",
        publication(mic),
        publication(screen, SCREEN_AUDIO),
    )
    stranger = participant("S-2", publication(SimpleNamespace(sid="TR_OTHER", kind=AUDIO)))
    room = SimpleNamespace(remote_participants={"one": learner, "two": stranger})

    found = list(existing_learner_microphones(
        room,
        "S-1",
        audio_kind=AUDIO,
        microphone_source=MICROPHONE,
    ))

    assert found == [(mic, learner.track_publications["0"], learner)]
    assert track_key(found[0][0]) == "TR_MIC"


def test_only_exact_learner_microphone_is_accepted():
    mic = SimpleNamespace(sid="TR_MIC", kind=AUDIO)
    learner = participant("S-1", publication(mic))
    assert is_learner_microphone(
        mic, learner.track_publications["0"], learner, "S-1",
        audio_kind=AUDIO, microphone_source=MICROPHONE,
    )
    assert not is_learner_microphone(
        mic, learner.track_publications["0"], learner, "S-2",
        audio_kind=AUDIO, microphone_source=MICROPHONE,
    )
    assert not is_learner_microphone(
        mic, publication(mic, SCREEN_AUDIO), learner, "S-1",
        audio_kind=AUDIO, microphone_source=MICROPHONE,
    )
