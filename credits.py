"""Trusted Credit reservation transitions for the live Q&A worker."""

from __future__ import annotations

import re

from campus_imports import configure_campus_imports

configure_campus_imports()

from common.db import fetch_one  # noqa: E402

UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


class CreditReservationError(RuntimeError):
    pass


def validate_raise_hand_reservation(
    reservation_id: str,
    registration_number: str,
    lecture_public_id: str,
) -> dict:
    if not UUID.fullmatch(reservation_id):
        raise CreditReservationError("The Credit reservation id is invalid.")
    row = fetch_one(
        """SELECT reservation.id::text, reservation.user_id::text, reservation.status,
                  reservation.amount, reservation.purpose, reservation.expires_at
             FROM credit_reservations AS reservation
             JOIN \"user\" AS learner ON learner.\"id\" = reservation.user_id
            WHERE reservation.id = %s::uuid
              AND learner.\"registrationNumber\" = %s
              AND reservation.purpose = 'raise_hand'
              AND reservation.amount = 2
              AND reservation.reference_type = 'lecture'
              AND reservation.reference_id = %s
              AND reservation.status = 'reserved'
              AND reservation.expires_at > CURRENT_TIMESTAMP""",
        (reservation_id, registration_number, lecture_public_id),
    )
    if not row:
        raise CreditReservationError(
            "The 2-Credit reservation is missing, expired, or belongs to another lecture."
        )
    return row


def settle_reservation(user_id: str, reservation_id: str) -> None:
    row = fetch_one(
        "SELECT id::text FROM settle_credit_reservation(%s::uuid, %s::uuid)",
        (user_id, reservation_id),
    )
    if not row:
        raise CreditReservationError("The Credit reservation could not be settled.")


def release_reservation(user_id: str, reservation_id: str) -> None:
    row = fetch_one(
        "SELECT id::text FROM release_credit_reservation(%s::uuid, %s::uuid)",
        (user_id, reservation_id),
    )
    if not row:
        raise CreditReservationError("The Credit reservation could not be released.")


__all__ = [
    "CreditReservationError",
    "release_reservation",
    "settle_reservation",
    "validate_raise_hand_reservation",
]
