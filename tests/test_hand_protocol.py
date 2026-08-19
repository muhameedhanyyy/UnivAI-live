from hand_protocol import hand_event, normalize_hand_request_id


REQUEST_ID = "22222222-2222-4222-8222-222222222222"


def test_valid_uuid4_is_preserved_for_browser_correlation():
    assert normalize_hand_request_id(REQUEST_ID) == REQUEST_ID


def test_invalid_or_non_v4_request_ids_are_not_reflected():
    assert normalize_hand_request_id("not-a-request-id") is None
    assert normalize_hand_request_id("11111111-1111-1111-8111-111111111111") is None
    assert normalize_hand_request_id(None) is None


def test_hand_lifecycle_events_keep_the_same_request_id():
    assert hand_event("raised", REQUEST_ID) == {
        "type": "hand",
        "state": "raised",
        "request_id": REQUEST_ID,
    }
    assert hand_event("acked", REQUEST_ID)["request_id"] == REQUEST_ID
    assert hand_event("lowered", REQUEST_ID)["request_id"] == REQUEST_ID


def test_busy_rejection_is_explicit_and_actionable():
    assert hand_event(
        "rejected",
        REQUEST_ID,
        detail="Finish the current question before raising your hand again.",
    ) == {
        "type": "hand",
        "state": "rejected",
        "request_id": REQUEST_ID,
        "detail": "Finish the current question before raising your hand again.",
    }
