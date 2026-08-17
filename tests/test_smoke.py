"""Smoke tests: package and foundation modules import cleanly."""
import importlib

import pytest

MODULES = [
    "aquarov",
    "aquarov.core",
    "aquarov.core.dto",
    "aquarov.ui.video_view",
    "aquarov.core.camera_manager",
]

GUI_MODULES = [
    "aquarov.core.video_view",
    "aquarov.ui.main_window",
]


@pytest.mark.parametrize("name", MODULES)
def test_import_core(name: str) -> None:
    assert importlib.import_module(name) is not None


@pytest.mark.parametrize("name", GUI_MODULES)
def test_import_gui(name: str) -> None:
    pytest.importorskip("PySide6")
    assert importlib.import_module(name) is not None


def test_dto_contract_intact() -> None:
    from aquarov.core import dto

    for symbol in ("CameraChannel", "Detection", "DetectionSet", "AppConfig"):
        assert hasattr(dto, symbol)
