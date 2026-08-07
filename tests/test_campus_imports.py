from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from campus_imports import configure_campus_imports


def test_configures_common_and_services_package_roots(tmp_path: Path):
    campus = tmp_path / "campus"
    common = campus / "services" / "common"
    common.mkdir(parents=True)
    (common / "__init__.py").write_text("", encoding="utf-8")
    (common / "live_probe.py").write_text("VALUE = 'common'\n", encoding="utf-8")
    (campus / "services" / "campus_probe.py").write_text(
        "VALUE = 'services'\n",
        encoding="utf-8",
    )

    module_names = (
        "common",
        "common.live_probe",
        "services",
        "services.campus_probe",
    )
    original_path = sys.path.copy()
    original_modules = {name: sys.modules.get(name) for name in module_names}
    try:
        for name in module_names:
            sys.modules.pop(name, None)
        configured_campus, configured_services = configure_campus_imports(campus)
        assert configured_campus == campus.resolve()
        assert configured_services == (campus / "services").resolve()
        assert importlib.import_module("common.live_probe").VALUE == "common"
        assert importlib.import_module("services.campus_probe").VALUE == "services"
    finally:
        sys.path[:] = original_path
        for name in module_names:
            sys.modules.pop(name, None)
            if original_modules[name] is not None:
                sys.modules[name] = original_modules[name]


@pytest.mark.parametrize("entry_point", ["worker.py", "qa.py", "tts.py"])
def test_every_integrated_entry_point_configures_campus_imports(entry_point: str):
    source = (Path(__file__).resolve().parents[1] / entry_point).read_text("utf-8")
    assert "configure_campus_imports()" in source
