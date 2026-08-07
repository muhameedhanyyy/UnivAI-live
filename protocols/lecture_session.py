"""App → Live session contract: room-metadata schema and parser.

The LiveKit App mints a token for each lecture room and places a JSON blob in
``roomMetadata``.  This module owns the parsing and validation of that blob so
that *one* place in the codebase defines the contract between the App team and
the Live team.

Contract shape (version 2)
--------------------------
::

    {
        "schema_name":  "univai.live.lecture-session",
        "schema_version": "2",
        "artifact_id":  "database-generated-uuid",
        "learner_id":   "S-2026-000042",
        "programme_id": "7",
        "course_id":    "database-generated-uuid",
        "plan_version": 1,
        "week":         1,
        "lecture_id":   "database-generated-uuid",
        "nonce":        "request-uuid"
    }

Unknown extra keys are ignored (forward-compatible).

Integrated / production mode
-----------------------------
Only identity and lookup fields are accepted. The lecture segments are loaded
from PostgreSQL by ``artifact_id`` and are never copied through LiveKit room
metadata. A missing or empty field raises
``SessionMetadataError`` immediately — the caller is responsible for publishing
an actionable error message to the room and stopping the affected feature rather
than silently substituting defaults.

Standalone / test mode
-----------------------
``LectureSessionMeta.standalone_fixture()`` returns a deterministic fixture
value whose fields are clearly labelled as test data (``standalone-fixture-*``).
This constructor is only accessible when ``runtime_mode()`` returns
``RuntimeMode.STANDALONE``; calling it in any other mode raises
``RuntimeError``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

# runtime.py lives in the same directory — import without sys.path games.
from runtime import RuntimeMode, runtime_mode

# --------------------------------------------------------------------------- #
# Schema version recorded in the contract.  Bump this when field names change. #
# --------------------------------------------------------------------------- #
METADATA_SCHEMA = "univai.live.lecture-session"
METADATA_SCHEMA_VERSION = "2"

# Canonical field names — defined here so tests and validation share one source.
_REQUIRED_FIELDS = (
    "programme_id",
    "course_id",
    "plan_version",
    "week",
    "lecture_id",
    "artifact_id",
    "learner_id",
    "nonce",
)


class SessionMetadataError(ValueError):
    """Raised when roomMetadata is absent, malformed, or missing required fields.

    Carrying the field name separately lets the caller build a structured error
    message (e.g. a ``progress`` payload to the browser) without string-parsing.
    """

    def __init__(self, message: str, *, field: str = "") -> None:
        super().__init__(message)
        self.field = field


@dataclass(frozen=True)
class LectureSessionMeta:
    """Validated session identity for one lecture room.

    Attributes
    ----------
    programme_id:
        Identifier of the academic programme (e.g. ``"programme-cs-2026"``).
    course_id:
        Identifier of the course within the programme.
    lecture_id:
        Identifier of the specific lecture (e.g. ``"lecture-week-1"``).
    plan_version:
        Approved curriculum plan version (e.g. ``1``).
    week:
        1-indexed week number within the course.
    sid:
        LiveKit participant / student identity; populated separately from the
        room name (not from roomMetadata).
    """

    programme_id: str
    course_id: str
    lecture_id: str
    plan_version: int
    week: int
    artifact_id: str
    nonce: str
    display_name: str | None = None
    sid: str = ""

    # ---------------------------------------------------------------------- #
    # Constructors                                                              #
    # ---------------------------------------------------------------------- #

    @classmethod
    def from_room_metadata(
        cls,
        room_name: str,
        room_metadata_json: str | None,
        *,
        sid: str = "",
    ) -> "LectureSessionMeta":
        """Parse and validate session identity from LiveKit room metadata.

        Parameters
        ----------
        room_name:
            The LiveKit room name, e.g. ``"lecture-S-2026-000042-week-1"``.
            Used only to derive ``sid`` when it is not supplied separately.
        room_metadata_json:
            The raw ``roomMetadata`` string from the LiveKit room object.
            Must be a JSON object containing all required fields.
        sid:
            Student / participant identity.  When empty, derived from the room
            name using the ``lecture-<sid>-week-N`` convention.

        Raises
        ------
        SessionMetadataError
            In integrated/production mode when metadata is absent, not valid
            JSON, or missing any required field.  In standalone mode this
            constructor still validates — use ``standalone_fixture()`` instead
            if you want deterministic test data without supplying real metadata.
        """
        mode = runtime_mode()

        # Derive sid from room name when not explicitly supplied.
        effective_sid = sid or _sid_from_room_name(room_name)

        if not room_metadata_json or not room_metadata_json.strip():
            if mode is RuntimeMode.INTEGRATED:
                raise SessionMetadataError(
                    "roomMetadata is absent or empty. "
                    "The App must embed programme_id, course_id, plan_version, "
                    "week, and lecture_id in the LiveKit room's roomMetadata JSON.",
                    field="roomMetadata",
                )
            # Standalone: caller should use standalone_fixture() instead, but
            # tolerate a missing blob so simulate.py can call this path safely.
            return cls.standalone_fixture(sid=effective_sid)

        try:
            data = json.loads(room_metadata_json)
        except (ValueError, TypeError) as exc:
            raise SessionMetadataError(
                f"roomMetadata is not valid JSON: {exc}", field="roomMetadata"
            ) from exc

        if not isinstance(data, dict):
            raise SessionMetadataError(
                "roomMetadata must be a JSON object.", field="roomMetadata"
            )

        legacy = data.get("schema_version") in {None, "1"}
        if not legacy:
            if data.get("schema_name") != METADATA_SCHEMA:
                raise SessionMetadataError("unsupported lecture metadata schema", field="schema_name")
            if data.get("schema_version") != METADATA_SCHEMA_VERSION:
                raise SessionMetadataError("unsupported lecture metadata version", field="schema_version")
        _validate_fields(data, mode, legacy=legacy)

        week = data["week"]
        if not isinstance(week, int) or week < 1:
            raise SessionMetadataError(
                f"roomMetadata.week must be a positive integer, got {week!r}.",
                field="week",
            )

        plan_version = data["plan_version"]
        if (
            isinstance(plan_version, bool)
            or not isinstance(plan_version, int)
            or plan_version < 1
        ):
            raise SessionMetadataError(
                "roomMetadata.plan_version must be a positive integer.",
                field="plan_version",
            )

        learner_id = data.get("learner_id", effective_sid)
        if learner_id != effective_sid:
            raise SessionMetadataError(
                "roomMetadata learner does not own this lecture room.",
                field="learner_id",
            )
        artifact_id = data.get("artifact_id", data["lecture_id"])
        display_name = data.get("display_name")
        if display_name is not None and not isinstance(display_name, str):
            raise SessionMetadataError("display_name must be text or null", field="display_name")

        return cls(
            programme_id=data["programme_id"].strip(),
            course_id=data["course_id"].strip(),
            lecture_id=data["lecture_id"].strip(),
            plan_version=plan_version,
            week=week,
            artifact_id=artifact_id.strip(),
            nonce=str(data.get("nonce", "legacy-room-metadata")).strip(),
            display_name=display_name.strip() if isinstance(display_name, str) and display_name.strip() else None,
            sid=effective_sid,
        )

    @classmethod
    def standalone_fixture(cls, *, sid: str = "S-2026-000042") -> "LectureSessionMeta":
        """Return the canonical standalone fixture session.

        This constructor is only available in standalone runtime mode so that
        fixture values can never silently appear in a configured integration or
        production environment.

        Raises
        ------
        RuntimeError
            If called outside standalone mode.
        """
        if runtime_mode() is not RuntimeMode.STANDALONE:
            raise RuntimeError(
                "LectureSessionMeta.standalone_fixture() may only be called in "
                "UNIVAI_MODE=standalone. Use from_room_metadata() in integrated mode."
            )
        return cls(
            programme_id="programme-demo-001",
            course_id="course-demo-001",
            lecture_id="lecture-week-1",
            plan_version=1,
            week=1,
            artifact_id="standalone-fixture-artifact",
            nonce="standalone-fixture-nonce",
            sid=sid,
        )

    def as_citation_scope(self) -> dict:
        """Return a dict of the identity fields used for citation enrichment."""
        return {
            "programme_id": self.programme_id,
            "course_id": self.course_id,
            "lecture_id": self.lecture_id,
            "plan_version": self.plan_version,
        }


# --------------------------------------------------------------------------- #
# Internal helpers                                                              #
# --------------------------------------------------------------------------- #


def _sid_from_room_name(room_name: str) -> str:
    """Extract student ID from a ``lecture-<sid>-week-N`` room name.

    Returns an empty string when the room name does not match the convention.
    """
    prefix, separator, _ = room_name.rpartition("-week-")
    if not separator or not prefix.startswith("lecture-"):
        return ""
    return prefix[len("lecture-") :]


def _validate_fields(data: dict, mode: RuntimeMode, *, legacy: bool) -> None:
    """Validate that all required string fields are present and non-empty.

    In integrated mode every missing or empty field raises immediately.
    In standalone mode the same rules apply when a metadata blob *is* present
    (the caller should use ``standalone_fixture()`` to skip validation).
    """
    fields = tuple(field for field in _REQUIRED_FIELDS if not legacy or field not in {"artifact_id", "learner_id", "nonce"})
    for field in fields:
        if field in {"week", "plan_version"}:
            if field not in data:
                raise SessionMetadataError(
                    f"roomMetadata is missing required field '{field}'.",
                    field=field,
                )
            continue
        value = data.get(field)
        if value is None:
            raise SessionMetadataError(
                f"roomMetadata is missing required field '{field}'.", field=field
            )
        if not isinstance(value, str) or not value.strip():
            raise SessionMetadataError(
                f"roomMetadata field '{field}' must be a non-empty string, "
                f"got {value!r}.",
                field=field,
            )

# --------------------------------------------------------------------------- #
# Canonical fixture metadata string (used in tests and simulator)              #
# --------------------------------------------------------------------------- #

STANDALONE_ROOM_METADATA: str = json.dumps(
    {
        "schema_name": METADATA_SCHEMA,
        "schema_version": METADATA_SCHEMA_VERSION,
        "programme_id": "programme-demo-001",
        "course_id": "course-demo-001",
        "plan_version": 1,
        "week": 1,
        "lecture_id": "lecture-week-1",
        "artifact_id": "standalone-fixture-artifact",
        "learner_id": "S-2026-000042",
        "nonce": "standalone-fixture-nonce",
    },
    separators=(",", ":"),
)
