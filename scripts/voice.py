#!/usr/bin/env python3
"""SIA en modo voz (FASE 7): escucha, piensa y responde hablando.

SIA arranca **en espera** y solo conversa cuando la activas por voz:
    "SIA buenos días" / "SIA buenas tardes" / "SIA buenas noches" / "SIA hola"
Mientras está activa responde sin repetir la palabra de activación, y se
apaga con:
    "SIA ya acabamos gracias"  (o "SIA apágate", "adiós")

Uso:
    source .venv/bin/activate
    python scripts/voice.py [--no-wake] [--start-active] [--reset-memory]

Opciones:
    --no-wake       Responde a todo lo que escuche (sin activación por voz).
    --start-active  Arranca conversando (sin esperar la activación).
    --reset-memory  Borra la memoria persistente antes de empezar.
"""
import argparse
import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.factory import build_memory, build_orchestrator, build_speaker_manager
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.voice.assistant import VoiceAssistant
from app.voice.stt import GoogleSTTProvider
from app.voice.tts import build_tts_provider

logging.getLogger("sia.voice").setLevel(logging.INFO)

logger = logging.getLogger("sia.voice")

_SCRIPTS_DIR = Path(__file__).resolve().parent


def _launch_interface() -> None:
    """Abre la ventana de escritorio de SIA (una sola instancia)."""
    try:
        already = subprocess.run(
            ["pgrep", "-f", "scripts/sia_app.py"],
            capture_output=True,
            check=False,
        )
        if already.returncode == 0:
            logger.info("La interfaz ya está abierta")
            return
        env = dict(os.environ)
        env.setdefault("DISPLAY", ":0")
        subprocess.Popen(
            [sys.executable, str(_SCRIPTS_DIR / "sia_app.py")],
            cwd=str(_SCRIPTS_DIR.parent),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        logger.info("Interfaz de escritorio lanzada por voz")
    except Exception:
        logger.exception("No se pudo abrir la interfaz de escritorio")


async def main() -> None:
    parser = argparse.ArgumentParser(description="SIA en modo voz")
    parser.add_argument("--no-wake", action="store_true")
    parser.add_argument("--start-active", action="store_true")
    parser.add_argument("--reset-memory", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)

    memory = build_memory(settings) if settings.memory_enabled else None
    if memory is not None:
        if args.reset_memory:
            memory.clear()
        memory.start_session("voz")

    orchestrator = build_orchestrator(
        settings, memory=memory, max_history_messages=8
    )
    stt = GoogleSTTProvider(language=settings.stt_language)
    tts = build_tts_provider(settings)
    speaker_manager = build_speaker_manager(settings)

    if not settings.llm_api_key:
        print("ADVERTENCIA: LLM_API_KEY vacía (configura .env si el proveedor la exige).")

    assistant = VoiceAssistant(
        orchestrator,
        stt,
        tts,
        wake_word=None if args.no_wake else settings.wake_word or "sia",
        language=settings.stt_language,
        active=True if (args.no_wake or args.start_active) else None,
        on_wake=lambda: asyncio.to_thread(_launch_interface),
        speaker_manager=speaker_manager,
    )
    if assistant.active:
        print("SIA escuchando… Ctrl+C para salir.")
    else:
        print(
            "SIA en espera. Di 'SIA buenos días' (o tardes/noches) para activarla. "
            "Ctrl+C para salir."
        )
    await assistant.run_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n¡Hasta luego!")