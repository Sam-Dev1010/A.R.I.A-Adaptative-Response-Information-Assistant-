#!/usr/bin/env python3
"""SIA como app de escritorio (FASE 12).

Ventana nativa sin marco con la interfaz futurista. Arranca el servidor web
local si no está corriendo, pausa el modo voz (para no escuchar dos veces)
y lo reanuda al cerrar la app.

Uso:
    source .venv/bin/activate
    pip install -r requirements-desktop.txt
    python scripts/sia_app.py [--no-server-check]
"""
import argparse
import logging
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication

# Obligatorio para QtWebEngine: debe declararse antes de crear QApplication.
QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)


def _install_sigterm_handler(app: QApplication) -> None:
    """SIGTERM/SIGINT → cierre elegante.

    QtWebEngine no despierta socketpairs; un timer de vigilancia garantiza
    que el handler pendiente de Python se ejecute y que app.quit() ocurra.
    """
    stop = threading.Event()

    def _forward(sig: int, frame) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, _forward)
    signal.signal(signal.SIGINT, _forward)

    def _watch() -> None:
        if stop.is_set():
            app.quit()
        else:
            QTimer.singleShot(200, _watch)

    QTimer.singleShot(200, _watch)

logger = logging.getLogger("sia.desktop")

_HEALTH_URL = "http://127.0.0.1:8000/health"
_DESKTOP = Path(__file__).resolve().parent.parent / "app" / "desktop"


def _server_running() -> bool:
    import httpx

    try:
        httpx.get(_HEALTH_URL, timeout=2)
        return True
    except httpx.HTTPError:
        return False


def ensure_server() -> subprocess.Popen | None:
    """Arranca uvicorn si hace falta; devuelve el proceso (o None si ya corría)."""
    if _server_running():
        return None
    venv_python = Path(sys.executable)
    process = subprocess.Popen(
        [str(venv_python), "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=venv_python.parent.parent,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(40):
        if _server_running():
            return process
        if process.poll() is not None:
            raise RuntimeError("El servidor de SIA no pudo arrancar")
        time.sleep(0.5)
    raise RuntimeError("El servidor de SIA tardó demasiado en arrancar")


def _toggle_voice_service(run: bool) -> None:
    """Pausa/reanuda el modo voz del sistema mientras la app está abierta."""
    action = "start" if run else "stop"
    try:
        subprocess.run(
            ["systemctl", "--user", action, "sia-voice.service"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="SIA: app de escritorio")
    parser.add_argument("--no-server-check", action="store_true")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("SIA")
    _install_sigterm_handler(app)

    from app.desktop.window import SiaWindow

    server_process = None
    if not args.no_server_check:
        try:
            server_process = ensure_server()
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    _toggle_voice_service(run=False)
    window = SiaWindow()
    window.show()

    closed = False

    def on_close() -> None:
        nonlocal closed
        if closed:
            return
        closed = True
        try:
            _toggle_voice_service(run=True)
        except Exception:
            logging.getLogger("sia.desktop").exception("fallo al restaurar el modo voz")
        if server_process is not None:
            server_process.terminate()
        app.quit()

    app.aboutToQuit.connect(on_close)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())