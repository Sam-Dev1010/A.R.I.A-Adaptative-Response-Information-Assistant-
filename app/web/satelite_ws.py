"""Satélite de voz ESP32: un botón en cualquier cuarto habla con SIA.

Protocolo por WebSocket en /ws/satelite:
    esp32     → frames BINARIOS con PCM crudo s16le 16 kHz mono (micrófono
                INMP441) mientras se mantiene pulsado el botón
    esp32     → {"type": "fin"} al soltar el botón
    servidor  → {"type": "estado", "valor": "pensando"}
    servidor  → {"type": "estado", "valor": "hablando"}
    servidor  → frames BINARIOS con PCM crudo s16le 16 kHz mono (bocina
                MAX98357A), listos para escribir directo al I2S
    servidor  → {"type": "audio_end"}
    servidor  → {"type": "error", "message": "..."} si algo falla

El PC hace toda la inteligencia; el ESP32 solo graba y reproduce, así el
firmware queda mínimo y la voz/memoria/herramientas son las mismas que en
casa. Medio dúplex por diseño: nunca se escucha mientras habla.
"""
import asyncio
import json
import logging
import subprocess
import uuid
from pathlib import Path

from fastapi import WebSocket, WebSocketDisconnect

from app.ai.orchestrator import AssistantOrchestrator
from app.ai.providers.base import LLMError
from app.voice.base import STTProvider, TTSProvider, VoiceError
from app.voice.stt import build_stt_provider
from app.voice.tts import build_tts_provider

logger = logging.getLogger("sia.satelite")

_SAMPLE_RATE = 16000  # idéntica en ambos sentidos; la fija el firmware
_FRAME_BYTES = 4096  # trozos cómodos para la librería WebSockets del ESP32


def _a_pcm(ruta: Path) -> bytes:
    """Convierte cualquier audio (mp3 de Edge, wav de Piper…) a PCM 16 kHz."""
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(ruta),
            "-ar",
            str(_SAMPLE_RATE),
            "-ac",
            "1",
            "-f",
            "s16le",
            "pipe:1",
        ],
        capture_output=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise VoiceError(f"ffmpeg: {result.stderr.decode(errors='replace')[:200]}")
    return result.stdout


class SateliteConnection:
    """Turnos push-to-talk contra el orquestador (STT/TTS inyectables)."""

    def __init__(
        self,
        ws: WebSocket,
        orchestrator: AssistantOrchestrator,
        *,
        stt: STTProvider | None = None,
        tts: TTSProvider | None = None,
    ) -> None:
        self._ws = ws
        self._orchestrator = orchestrator
        self._stt = stt
        self._tts = tts

    async def run(self) -> None:
        await self._ws.accept()
        session = AssistantOrchestrator(
            self._orchestrator.provider,
            system_prompt=(
                f"{self._orchestrator.system_prompt}\n"
                "Estás en conversación hablada: responde en 1 o 2 frases "
                "cortas y naturales, sin listas ni markdown ni emojis."
            ),
            registry=self._orchestrator.registry,
            policy=self._orchestrator.policy,
            confirm=self._confirm,
            memory=self._orchestrator.memory,
            auto_learner=self._orchestrator.auto_learner,
            max_history_messages=8,
        )
        stt = self._stt or build_stt_provider()
        tts = self._tts or build_tts_provider()

        pcm = bytearray()
        try:
            while True:
                mensaje = await self._ws.receive()
                if mensaje["type"] == "websocket.disconnect":
                    break
                if mensaje.get("bytes"):
                    pcm += mensaje["bytes"]
                    continue
                try:
                    datos = json.loads(mensaje.get("text") or "{}")
                except ValueError:
                    continue
                if datos.get("type") == "fin":
                    await self._turno(session, stt, tts, bytes(pcm))
                    pcm.clear()  # mismo socket sirve para el siguiente turno
        except WebSocketDisconnect:
            pass

    async def _turno(self, session, stt, tts, pcm: bytes) -> None:
        try:
            await self._responder(session, stt, tts, pcm)
        except VoiceError as exc:
            logger.warning("Error de voz del satélite: %s", exc)
            await self._ws.send_json({"type": "error", "message": f"No te entendí: {exc}"})
        except LLMError as exc:
            logger.warning("Error del LLM: %s", exc)
            await self._ws.send_json({"type": "error", "message": f"El cerebro falló: {exc}"})
        except Exception as exc:  # avisar y seguir vivo
            logger.exception("Error inesperado en el satélite")
            await self._ws.send_json({"type": "error", "message": f"Error interno: {exc}"})

    async def _responder(self, session, stt, tts, pcm: bytes) -> None:
        if not pcm:
            raise VoiceError("silencio")
        await self._ws.send_json({"type": "estado", "valor": "pensando"})

        text = await asyncio.to_thread(stt.transcribe_bytes, pcm)
        if not text.strip():
            raise VoiceError("silencio")
        logger.info("Satélite transcribió: %r", text[:80])

        respuesta = await session.ask(text)
        ext = getattr(tts, "audio_ext", ".mp3")
        ruta = Path(f"/tmp/sia_sat_{uuid.uuid4().hex}{ext}")
        try:
            await asyncio.to_thread(tts.synthesize, respuesta.content, ruta)
            audio = await asyncio.to_thread(_a_pcm, ruta)
        finally:
            await asyncio.to_thread(ruta.unlink, True)

        await self._ws.send_json({"type": "estado", "valor": "hablando"})
        for i in range(0, len(audio), _FRAME_BYTES):
            await self._ws.send_bytes(audio[i : i + _FRAME_BYTES])
        await self._ws.send_json({"type": "audio_end"})
        logger.info("Respuesta al satélite: %d bytes de PCM", len(audio))

    async def _confirm(self, call) -> bool:
        # El satélite es de uso personal: las herramientas se auto-aprueban.
        return True
