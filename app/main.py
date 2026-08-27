"""Punto de entrada de SIA.

FASE 1: base del proyecto (FastAPI + /health).
FASE 8: GUI web (chat por WebSocket en /ws/chat).
FASE 11: interfaz futurista por voz en / (ws/interface).
"""
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.ai.factory import build_orchestrator
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.web.device_ws import DeviceHubConnection, get_hub
from app.web.interface_ws import InterfaceConnection
from app.web.presencia_ws import PresenciaConnection
from app.web.satelite_ws import SateliteConnection
from app.web.ws import ChatConnection

_STARTED_AT = time.monotonic()
_STATIC_DIR = Path(__file__).parent / "web" / "static"

logger = get_logger("sia")


def _orchestrator_for(app: FastAPI) -> object:
    """Devuelve el orquestador compartido de la aplicación (lo crea si hace falta)."""
    if getattr(app.state, "orchestrator", None) is None:
        app.state.orchestrator = build_orchestrator()
    return app.state.orchestrator


async def _rechazar_acceso(ws: WebSocket, motivo: str) -> None:
    """Acepta solo para informar y cierra con código de política (1008)."""
    await ws.accept()
    await ws.send_json({"type": "error", "message": motivo})
    await ws.close(code=1008)


def _es_local(ws: WebSocket) -> bool:
    """True si la conexión nació en esta misma PC (loopback)."""
    host = ws.client.host if ws.client else ""
    return host in {"127.0.0.1", "::1", "localhost"}


def _acceso_permitido(ws: WebSocket) -> bool:
    """Si hay ACCESS_TOKEN definido, el WebSocket debe traerlo como ?token=…

    La propia PC (localhost) queda exenta: usar SIA desde el escritorio nunca
    pide token; el celular u otros equipos de la red sí.
    """
    esperado = (get_settings().access_token or "").strip()
    if not esperado or _es_local(ws):
        return True
    return ws.query_params.get("token", "") == esperado


async def chat_socket(ws: WebSocket) -> None:
    if not _acceso_permitido(ws):
        await _rechazar_acceso(ws, "Acceso denegado: token inválido.")
        return
    await ChatConnection(ws, _orchestrator_for(ws.app)).run()


async def interface_socket(ws: WebSocket) -> None:
    if not _acceso_permitido(ws):
        await _rechazar_acceso(ws, "Acceso denegado: token inválido.")
        return
    await InterfaceConnection(ws, _orchestrator_for(ws.app)).run()


async def satelite_socket(ws: WebSocket) -> None:
    if not _acceso_permitido(ws):
        await _rechazar_acceso(ws, "Acceso denegado: token inválido.")
        return
    await SateliteConnection(ws, _orchestrator_for(ws.app)).run()


async def device_socket(ws: WebSocket) -> None:
    """Canal de control del celular: la app S.I.A se registra como ejecutora."""
    if not _acceso_permitido(ws):
        await _rechazar_acceso(ws, "Acceso denegado: token inválido.")
        return
    await DeviceHubConnection(get_hub(), ws).run()


async def presencia_socket(ws: WebSocket) -> None:
    """Satélite ESP32 de presencia: reporta si el jefe está en casa."""
    if not _acceso_permitido(ws):
        await _rechazar_acceso(ws, "Acceso denegado: token inválido.")
        return
    await PresenciaConnection(ws).run()


def create_app(orchestrator=None) -> FastAPI:
    settings = get_settings()
    setup_logging(settings.log_level)

    logger.info(
        "Iniciando %s v%s (env=%s)",
        settings.app_name,
        settings.app_version,
        settings.environment,
        extra=settings.safe_dict(),
    )

    settings.data_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Asistente personal de IA inspirado en JARVIS.",
    )
    app.state.orchestrator = orchestrator

    @app.get("/health", tags=["system"])
    async def health():
        return {
            "status": "ok",
            "service": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
            "uptime_seconds": round(time.monotonic() - _STARTED_AT, 1),
            "server_time": datetime.now(UTC).astimezone().isoformat(),
            "data_dir": str(settings.data_dir),
        }

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(_STATIC_DIR / "interface.html")

    @app.get("/chat", include_in_schema=False)
    async def chat_page():
        return FileResponse(_STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    app.add_api_websocket_route("/ws/chat", chat_socket)
    app.add_api_websocket_route("/ws/interface", interface_socket)
    app.add_api_websocket_route("/ws/satelite", satelite_socket)
    app.add_api_websocket_route("/ws/device", device_socket)
    app.add_api_websocket_route("/ws/presencia", presencia_socket)

    @asynccontextmanager
    async def _vida(app: FastAPI):
        """Ciclo de vida: arranca y detiene la charla espontánea."""
        hablante = None
        if settings.proactive_enabled:
            from app.ai.proactive import ProactiveSpeaker
            from app.voice.tts import build_tts_provider

            orquestador = _orchestrator_for(app)
            hablante = ProactiveSpeaker(
                orquestador.provider,
                orquestador.memory,
                build_tts_provider(settings),
                settings,
            )
            hablante.start()
        yield
        if hablante is not None:
            await hablante.stop()

    app.router.lifespan_context = _vida

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)