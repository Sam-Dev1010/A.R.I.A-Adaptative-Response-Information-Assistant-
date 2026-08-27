"""Smoke test de la app de escritorio (FASE 12)."""
import os


def test_desktop_window_constructs():
    import pytest

    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        pytest.skip("QtWebEngine necesita una pantalla real para renderizar")
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        pytest.skip("no se puede renderizar WebEngine en offscreen")

    import sys

    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication

    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    from app.desktop.window import SiaWindow

    app = QApplication.instance() or QApplication(sys.argv)
    window = SiaWindow()

    assert window.windowTitle() == "A.R.I.A"
    assert window.windowFlags() & Qt.WindowType.FramelessWindowHint
    assert window._bridge is not None
    app.quit()