"""Ventana de escritorio de A.R.I.A (FASE 12): interfaz futurista como app nativa.

Ventana sin marco con la interfaz HUD embebida (QtWebEngine). Expone un
puente JS ↔ Python para los controles de ventana (minimizar/cerrar).
"""
import logging

from PyQt6.QtCore import QObject, Qt, QUrl, pyqtSlot
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QMainWindow

logger = logging.getLogger("sia.desktop")

_SERVER_URL = "http://127.0.0.1:8000/"


class WindowBridge(QObject):
    """Puente JS → Python: controles de la ventana desde la interfaz HUD."""

    def __init__(self, window: QMainWindow) -> None:
        super().__init__()
        self._window = window

    @pyqtSlot()
    def closeWindow(self) -> None:
        self._window.close()

    @pyqtSlot()
    def minimize(self) -> None:
        self._window.showMinimized()


class SiaWindow(QMainWindow):
    """Ventana principal sin marco: la interfaz HUD ocupa toda la pantalla."""

    def __init__(self, url: str = _SERVER_URL) -> None:
        super().__init__()
        self.setWindowTitle("A.R.I.A")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint)
        self.resize(1280, 800)

        self._view = QWebEngineView(self)
        self.setCentralWidget(self._view)

        page = self._view.page()
        page.featurePermissionRequested.connect(self._grant_mic)
        page.settings().setAttribute(
            QWebEngineSettings.WebAttribute.PlaybackRequiresUserGesture, False
        )
        self._channel = QWebChannel(page)
        self._bridge = WindowBridge(self)
        self._channel.registerObject("sia", self._bridge)
        page.setWebChannel(self._channel)

        self._view.load(QUrl(url))

    def _grant_mic(self, url: QUrl, feature: QWebEnginePage.Feature) -> None:
        """Concede el micrófono sin preguntar (la app ES para hablar con A.R.I.A)."""
        if feature == QWebEnginePage.Feature.MediaAudioCapture:
            self._view.page().setFeaturePermission(
                url,
                feature,
                QWebEnginePage.PermissionPolicy.PermissionGrantedByUser,
            )

    def show_on_screen(self) -> None:
        self.show()