"""Durable lecture checkpoint and learner-presence persistence.

The LiveKit worker is intentionally disposable.  Everything needed to resume a
lecture after a page refresh, network loss, or worker restart therefore lives
in Postgres rather than in a Python process.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


REPLAY_SENTENCES = 3
GRACE_MINUTES = 5


class LectureAdmissionClosed(PermissionError):
    """The learner never joined and the first-admission cutoff has passed."""


@dataclass(frozen=True)
class LectureCheckpoint:
    """The next new sentence and the total number of sentences in the script."""

    next_sentence_index: int
    total_sentences: int

    @property
    def replay_from(self) -> int:
        return replay_start(self.next_sentence_index, self.total_sentences)

    @property
    def is_resume(self) -> bool:
        return self.next_sentence_index > 0


def replay_start(
    next_sentence_index: int,
    total_sentences: int,
    replay_sentences: int = REPLAY_SENTENCES,
) -> int:
    """Return a safe flat-script index, rewound by the requested sentence count."""

    total = max(0, int(total_sentences))
    checkpoint = min(max(0, int(next_sentence_index)), total)
    rewind = max(0, int(replay_sentences))
    return max(0, checkpoint - rewind)


class LectureProgressRepository:
    """Small synchronous repository; callers keep it off the asyncio loop."""

    def __init__(self, *, lecture_id: int, learner_id: str) -> None:
        if lecture_id < 1 or not learner_id:
            raise ValueError("a persisted lecture and learner are required")
        self.lecture_id = lecture_id
        self.learner_id = learner_id

    def initialise(self, total_sentences: int) -> LectureCheckpoint:
        from common.db import fetch_one

        total = max(0, int(total_sentences))
        row = fetch_one(
            """UPDATE attendance
                  SET total_sentences = %s,
                      last_sentence_index = LEAST(last_sentence_index, %s)
                WHERE lecture_id = %s AND student_id = %s
                  AND completed_at IS NULL
                RETURNING last_sentence_index, total_sentences""",
            (total, total, self.lecture_id, self.learner_id),
        )
        if not row:
            raise RuntimeError("attendance row is missing for the live lecture")
        return LectureCheckpoint(
            next_sentence_index=int(row["last_sentence_index"]),
            total_sentences=int(row["total_sentences"]),
        )

    def ensure_joined(self, joined_at: datetime) -> bool:
        """Create the first attendance row; return whether this was first admission."""

        from common.db import fetch_one

        admission = fetch_one(
            """WITH inserted AS (
                 INSERT INTO attendance
                     (student_id, lecture_id, joined_at, status, late_minutes)
                SELECT %s, l.id, %s,
                       CASE
                         WHEN %s > l.starts_at + (%s * INTERVAL '1 minute')
                           THEN 'late'
                         ELSE 'on_time'
                       END,
                       CASE
                         WHEN %s > l.starts_at + (%s * INTERVAL '1 minute')
                           THEN GREATEST(
                             0,
                             FLOOR(EXTRACT(EPOCH FROM (%s - l.starts_at)) / 60)::integer
                           )
                         ELSE 0
                       END
                  FROM lectures l
                  LEFT JOIN lecture_artifacts la
                    ON la.artifact_id = l.lecture_artifact_id
                 WHERE l.id = %s AND l.student_id = %s
                   AND %s <= l.starts_at
                     + ((CASE
                           WHEN (la.script_payload->>'durationMinutes') ~ '^[0-9]+$'
                            AND (la.script_payload->>'durationMinutes')::integer
                                BETWEEN 45 AND 120
                             THEN (la.script_payload->>'durationMinutes')::integer
                           ELSE 60
                         END)::double precision / 2 * INTERVAL '1 minute')
                ON CONFLICT (student_id, lecture_id) DO NOTHING
                RETURNING TRUE AS first_admission
              )
              SELECT first_admission FROM inserted
              UNION ALL
              SELECT FALSE AS first_admission
                FROM attendance
               WHERE lecture_id = %s AND student_id = %s
                 AND completed_at IS NULL
                 AND NOT EXISTS (SELECT 1 FROM inserted)
               LIMIT 1""",
            (
                self.learner_id,
                joined_at,
                joined_at,
                GRACE_MINUTES,
                joined_at,
                GRACE_MINUTES,
                joined_at,
                self.lecture_id,
                self.learner_id,
                joined_at,
                self.lecture_id,
                self.learner_id,
            ),
        )
        if admission is None:
            raise LectureAdmissionClosed("the first-admission cutoff has passed")
        return bool(admission["first_admission"])

    def mark_present(self, seen_at: datetime) -> None:
        from common.db import execute

        execute(
            """UPDATE attendance
                  SET is_connected = TRUE,
                      presence_last_seen_at = %s,
                      last_connected_at = %s
                WHERE lecture_id = %s AND student_id = %s
                  AND completed_at IS NULL""",
            (seen_at, seen_at, self.lecture_id, self.learner_id),
        )

    def touch_presence(self, attended_seconds: float, seen_at: datetime) -> None:
        from common.db import execute

        execute(
            """UPDATE attendance
                  SET attended_seconds = attended_seconds + %s,
                      is_connected = TRUE,
                      presence_last_seen_at = %s
                WHERE lecture_id = %s AND student_id = %s
                  AND completed_at IS NULL""",
            (
                max(0.0, float(attended_seconds)),
                seen_at,
                self.lecture_id,
                self.learner_id,
            ),
        )

    def mark_absent(self, attended_seconds: float, disconnected_at: datetime) -> None:
        from common.db import execute

        execute(
            """UPDATE attendance
                  SET attended_seconds = attended_seconds + %s,
                      is_connected = FALSE,
                      presence_last_seen_at = %s,
                      last_disconnected_at = %s,
                      disconnect_count = disconnect_count + 1
                WHERE lecture_id = %s AND student_id = %s
                  AND completed_at IS NULL""",
            (
                max(0.0, float(attended_seconds)),
                disconnected_at,
                disconnected_at,
                self.lecture_id,
                self.learner_id,
            ),
        )

    def record_sentence(self, next_sentence_index: int, total_sentences: int) -> None:
        from common.db import execute

        total = max(0, int(total_sentences))
        checkpoint = min(max(0, int(next_sentence_index)), total)
        execute(
            """UPDATE attendance
                  SET total_sentences = %s,
                      last_sentence_index = GREATEST(last_sentence_index, %s)
                WHERE lecture_id = %s AND student_id = %s
                  AND completed_at IS NULL""",
            (total, checkpoint, self.lecture_id, self.learner_id),
        )

    def complete(self, attended_seconds: float, completed_at: datetime) -> None:
        from common.db import execute

        execute(
            """UPDATE attendance
                  SET attended_seconds = attended_seconds + %s,
                      is_connected = FALSE,
                      presence_last_seen_at = %s,
                      last_sentence_index = total_sentences,
                      completed_at = COALESCE(completed_at, %s)
                WHERE lecture_id = %s AND student_id = %s
                  AND completed_at IS NULL""",
            (
                max(0.0, float(attended_seconds)),
                completed_at,
                completed_at,
                self.lecture_id,
                self.learner_id,
            ),
        )
