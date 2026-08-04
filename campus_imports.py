"""Make the shared campus services importable from Live entry points."""

from __future__ import annotations

import sys
from pathlib import Path


CAMPUS_ROOT = Path(__file__).resolve().parent.parent


def configure_campus_imports(campus_root: Path = CAMPUS_ROOT) -> tuple[Path, Path]:
    """Expose both supported shared-service package names.

    Live historically imports ``common.*`` from ``UnivAI/services`` while the
    shared modules import one another as ``services.*`` from ``UnivAI``. Both
    roots are therefore required when a Live script is launched directly.
    """

    campus_root = campus_root.resolve()
    services_root = campus_root / "services"
    for root in (campus_root, services_root):
        value = str(root)
        if value not in sys.path:
            # Live's own modules (for example health.py) must win over campus
            # modules with the same basename, so shared roots belong at the end.
            sys.path.append(value)
    return campus_root, services_root
