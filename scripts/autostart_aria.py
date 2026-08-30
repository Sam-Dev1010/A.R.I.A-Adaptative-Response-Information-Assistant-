#!/usr/bin/env python3
"""Arranque automático de A.R.I.A al encender el PC.

Diseñado para ejecutarse en segundo plano al iniciar sesión (systemd user o
desktop autostart). Al arrancar saluda al jefe por voz, verifica
actualizaciones pendientes si se pide, y luego queda escuchando por micrófono.

Uso:
    python scripts/autostart_aria.py [--boot-greeting] [--start-active]

Opciones:
    --boot-greeting  Habla un saludo al arrancar (para encendido del PC).
    --start-active   Arranca conversando (sin esperar la palabra de activación).
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.factory import build_memory, build_orchestrator, build_speaker_manager
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.voice.assistant import VoiceAssistant
from app.voice.stt import GoogleSTTProvider
from app.voice.tts import build_tts_provider

logger = logging.getLogger("sia.autostart")


def _saludo_arranque() -> str:
    """Saludo breve que A.R.I.A dice al terminar de encender el PC."""
    return (
        "Buenos días, jefe. Ya estoy despierta y con todos los sistemas "
        "en línea. Dime ARIA cuando me necesites."
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="A.R.I.A autostart")
    parser.add_argument("--boot-greeting", action="store_true")
    parser.add_argument("--start-active", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)

    memory = build_memory(settings) if settings.memory_enabled else None
    orquestador = build_orchestrator(settings, memory=memory, max_history_messages=8)
    stt = GoogleSTTProvider(language=settings.stt_language)
    tts = build_tts_provider(settings)
    speaker_manager = build_speaker_manager(settings)

    assistant = VoiceAssistant(
        orquestador,
        stt,
        tts,
        wake_word=None if args.start_active else settings.wake_word or "aria",
        language=settings.stt_language,
        active=True if args.start_active else None,
        speaker_manager=speaker_manager,
    )

    if args.boot_greeting or assistant.active:
        try:
            await tts.speak(_saludo_arranque())
        except Exception as exc:  # noqa: BLE001 — saludar nunca tumba el arranque
            logger.warning("No pude hablar el saludo de arranque: %s", exc)

    if assistant.active:
        print("A.R.I.A despierta y escuchando… Ctrl+C para salir.")
    else:
        print(
            "A.R.I.A en espera. Di 'ARIA buenos días' para activarla. Ctrl+C para salir."
        )
    await assistant.run_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n¡Hasta luego!")
