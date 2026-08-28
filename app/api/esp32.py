"""Recepción de alertas del nodo de IA local (ESP32).

El ESP32 ejecuta la inferencia embebida con TensorFlow Lite Micro y, cuando la
probabilidad supera el umbral, notifica por HTTP:

    POST /api/esp32/event
    {"node_id": "esp32_ia_local", "prediction": 0.87}

Este módulo:
1. Define el esquema Pydantic ``ESP32Event`` (valida y documenta la carga útil).
2. Expone el router FastAPI con la ruta ``POST /api/esp32/event``.
3. Registra la alerta en el logger estructurado del sistema.
4. Mantiene un registro de *hooks* por aplicación para que el orquestador
   (``VoiceAssistant`` / ``MemoryManager``) pueda responder a la alerta o
   guardarla, sin acoplar el firmware a la lógica de voz/memoria.

Los hooks se asocian a cada ``app`` (no globales), de modo que varios
``create_app()`` en pruebas nunca comparten manejadores.
"""
from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import APIRouter, FastAPI, Request
from pydantic import BaseModel, Field

from app.core.logging import get_logger

logger = get_logger("sia.esp32")

# Un handler recibe (node_id, prediction) y decide qué hacer: hablar, guardar,
# encender un LED en pantalla, etc. Nunca debe lanzar excepciones.
Esp32Handler = Callable[[str, float], Awaitable[None]]
_HANDLERS_ATTR = "esp32_handlers"


class ESP32Event(BaseModel):
    """Evento enviado por el nodo de IA local (ESP32) tras una detección."""

    node_id: Annotated[
        str,
        Field(
            min_length=1,
            max_length=64,
            description="Identificador del nodo ESP32 (p. ej. esp32_ia_local).",
        ),
    ]
    prediction: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Probabilidad de detección normalizada entre 0 y 1.",
        ),
    ]


router = APIRouter(prefix="/api/esp32", tags=["esp32"])


# --- registro de hooks ------------------------------------------------------


def _handlers(app: FastAPI) -> set[Esp32Handler]:
    """Conjunto de hooks registrados en una aplicación (se crea perezosamente)."""
    registrados = getattr(app.state, _HANDLERS_ATTR, None)
    if registrados is None:
        registrados = set()
        setattr(app.state, _HANDLERS_ATTR, registrados)
    return registrados


def register_esp32_handler(app: FastAPI, handler: Esp32Handler) -> None:
    """Asocia un handler a la alerta. Idempotente: evita duplicados."""
    _handlers(app).add(handler)


def unregister_esp32_handler(app: FastAPI, handler: Esp32Handler) -> None:
    """Desasocia un handler (útil al detener la aplicación)."""
    _handlers(app).discard(handler)


def esp32_handlers(app: FastAPI) -> tuple[Esp32Handler, ...]:
    """Copia de los hooks actuales (para inspección o pruebas)."""
    return tuple(_handlers(app))


async def _disparar(app: FastAPI, event: ESP32Event) -> None:
    """Ejecuta todos los hooks sin que uno falle rompa el resto."""
    for handler in list(_handlers(app)):
        try:
            await handler(event.node_id, event.prediction)
        except Exception as exc:  # noqa: BLE001 — un hook nunca rompe el endpoint
            logger.warning(
                "Hook del nodo ESP32 falló",
                extra={"node_id": event.node_id, "error": str(exc)},
            )


# --- ruta ------------------------------------------------------------------


@router.post(
    "/event",
    summary="Alerta del nodo de IA local (ESP32)",
    response_model=dict,
)
async def esp32_event(event: ESP32Event, request: Request) -> dict:
    """Recibe una alerta del ESP32, la registra y la reparte a los hooks.

    El firmware considera éxito cualquier respuesta HTTP >= 200, así que se
    devuelve 200 con el evento tal y como se validó.
    """
    logger.info(
        "Alerta del nodo de IA local",
        extra={
            "node_id": event.node_id,
            "prediction": event.prediction,
            "source": "esp32",
        },
    )
    await _disparar(request.app, event)
    return {"status": "ok", "node_id": event.node_id, "prediction": event.prediction}