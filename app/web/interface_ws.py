"""Interfaz de voz de SIA (FASE 11): el navegador graba, SIA responde hablando.

Flujo por WebSocket:
    navegador → {"type": "audio", "data": "<base64 webm>"}
    servidor  → {"type": "thinking"}
    servidor  → {"type": "speaking"}  (en cuanto llega el primer fragmento)
    servidor  → {"type": "audio_chunk", "audio": "<base64>", "mime": "...",
                 "last": false}  (uno por oración, EN CUANTO ESTÉ LISTO)
    servidor  → {"type": "response", "text": "..."}  (tras cerrar el stream)
    servidor  → {"type": "audio_end"}  (cierra el turno de habla)
    servidor  → {"type": "error", "message": "..."} si algo falla

La VOZ VA PRIMERO: la respuesta del LLM se consume en streaming y cada oración
se sintetiza y se envía apenas está lista — suena mientras el modelo sigue
escribiendo el resto. El texto completo llega después.
"""
import asyncio
import base64
import logging
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from fastapi import WebSocket, WebSocketDisconnect

from app.ai.orchestrator import AssistantOrchestrator
from app.ai.providers.base import LLMError
from app.voice.base import STTProvider, TTSProvider, VoiceError
from app.voice.stt import build_stt_provider
from app.voice.tts import build_tts_provider

logger = logging.getLogger("sia.web")

_SAMPLE_RATE = 16000
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")
_CLAUSE_SPLIT = re.compile(r"(?<=[,;:])\s+")
_MIME_BY_EXT = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg"}
_LONG_BUFFER_CHARS = 120  # corta buffers sin puntuación para no esperar de más
_CLAUSE_MIN_CHARS = 60  # mínimo antes de cortar por coma: suena natural, no entrecortado

# Símbolos que la voz lee mal o provoca pausas artificiales.
_RUIDO_VOZ = re.compile(
    r"[*_`#>|~\[\]{}<>]|https?://\S+|[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]"
)

# ---------------------------------------------------------------------------
# Registro global: permite a SIA hablar por su cuenta a quien esté escuchando.
# ---------------------------------------------------------------------------
_conexiones: set["InterfaceConnection"] = set()


def hay_alguien_escuchando() -> bool:
    """True si hay al menos una interfaz conectada y libre."""
    return any(not c.ocupada for c in _conexiones)


async def hablar_a_todas(texto: str, tts: TTSProvider) -> int:
    """Sintetiza ``texto`` y se lo manda a cada interfaz libre. Devuelve a
    cuántas les llegó (0 si nadie escuchaba o la voz falló)."""
    limpia = _limpiar_para_voz(texto)
    if not limpia:
        return 0
    ext = getattr(tts, "audio_ext", ".mp3")
    mime = _MIME_BY_EXT.get(ext, "audio/mpeg")
    ruta = Path(f"/tmp/sia_proactiva_{uuid.uuid4().hex}{ext}")
    try:
        await tts.synthesize(limpia, ruta)
        audio = base64.b64encode(await asyncio.to_thread(ruta.read_bytes)).decode(
            "ascii"
        )
    except Exception as exc:  # noqa: BLE001 — hablar nunca debe romper el server
        logger.warning("No pude sintetizar charla espontánea: %s", exc)
        return 0
    finally:
        await asyncio.to_thread(ruta.unlink, True)

    enviados = 0
    for conexion in list(_conexiones):
        if conexion.ocupada:
            continue  # no interrumpir: está procesando o hablando con el usuario
        try:
            await conexion.ws.send_json({"type": "speaking"})
            await conexion.ws.send_json(
                {"type": "audio_chunk", "audio": audio, "mime": mime, "last": True}
            )
            await conexion.ws.send_json({"type": "audio_end"})
            enviados += 1
        except Exception:  # noqa: BLE001 — una conexión muerta no corta a las demás
            logger.debug("No pude hablarle a una interfaz (desconectada)")
    if enviados:
        logger.info("Charla espontánea enviada a %d interfaz(es)", enviados)
    return enviados


def _limpiar_para_voz(texto: str) -> str:
    """Quita markdown, emojis y URLs: la voz suena natural, no deletreada."""
    limpio = _RUIDO_VOZ.sub(" ", texto)
    return re.sub(r"\s{2,}", " ", limpio).strip()


def _extract_sentence(buffer: str) -> tuple[str | None, str]:
    """Saca el primer trozo hablable del buffer; None si aún no hay.

    Corta en la oración completa si ya existe; si no, en la primera coma
    después de unos caracteres (la pausa de coma es natural al hablar) y
    así el primer audio sale mucho antes de que el LLM termine.
    """
    match = _SENTENCE_SPLIT.search(buffer)
    if match:
        return buffer[: match.end()].strip(), buffer[match.end() :]
    clausula = _CLAUSE_SPLIT.search(buffer, _CLAUSE_MIN_CHARS)
    if clausula:
        return buffer[: clausula.end()].strip(), buffer[clausula.end() :]
    if len(buffer) > _LONG_BUFFER_CHARS:
        corte = buffer.rfind(" ", 40)
        if corte > 0:
            return buffer[:corte].strip(), buffer[corte:]
    return None, buffer


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _guardar_audio_rechazado(webm: bytes) -> None:
    """Guarda el blob que ffmpeg rechazó para poder inspeccionarlo después."""
    try:
        from pathlib import Path

        destino = Path("/tmp/opencode") / f"audio_fallido_{int(time.time())}.bin"
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(webm)
    except OSError as err:
        logger.debug("No se pudo guardar el audio rechazado: %s", err)


class InterfaceConnection:
    """Conversación por voz contra un orquestador (con STT/TTS inyectables)."""

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
        self._queue: asyncio.Queue = asyncio.Queue()
        self.ocupada = False  # True mientras procesa/responde al usuario

    @property
    def ws(self) -> WebSocket:
        return self._ws

    async def run(self) -> None:
        await self._ws.accept()
        _conexiones.add(self)

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

        reader = asyncio.create_task(self._receive_loop())
        try:
            while True:
                item = await self._queue.get()
                if not item:
                    continue
                try:
                    if isinstance(item, tuple) and item[0] == "texto":
                        self.ocupada = True
                        try:
                            await self._responder(session, tts, item[1])
                        finally:
                            self.ocupada = False
                    else:
                        await self._handle_audio(session, stt, tts, item)
                except VoiceError as exc:
                    logger.warning("Error de voz: %s", exc)
                    await self._ws.send_json(
                        {"type": "error", "message": f"No te entendí: {exc}"}
                    )
                except LLMError as exc:
                    logger.warning("Error del LLM: %s", exc)
                    await self._ws.send_json(
                        {"type": "error", "message": f"El cerebro falló: {exc}"}
                    )
                except Exception as exc:
                    logger.exception("Error inesperado en la interfaz de voz")
                    await self._ws.send_json(
                        {"type": "error", "message": f"Error interno: {exc}"}
                    )
        except WebSocketDisconnect:
            pass
        finally:
            reader.cancel()
            _conexiones.discard(self)

    async def _handle_audio(self, session, stt, tts, raw: bytes) -> None:
        if not _ffmpeg_available():
            raise VoiceError("ffmpeg no está instalado")
        self.ocupada = True
        try:
            await self._procesar_audio(session, stt, tts, raw)
        finally:
            self.ocupada = False

    async def _procesar_audio(self, session, stt, tts, raw: bytes) -> None:
        wav = await asyncio.to_thread(self._decode_to_pcm, raw)
        text = await asyncio.to_thread(stt.transcribe_bytes, wav)
        if not text.strip():
            raise VoiceError("silencio")
        logger.info("Audio recibido y transcrito: %r", text[:80])
        await self._responder(session, tts, text.strip())

    async def _responder(self, session, tts, texto: str) -> None:
        """Genera la respuesta a un texto (voz o teclado) y la emite hablando."""
        await self._ws.send_json({"type": "thinking"})

        ext = getattr(tts, "audio_ext", ".mp3")
        self._mime_actual = _MIME_BY_EXT.get(ext, "audio/mpeg")
        sintetizadas: asyncio.Queue = asyncio.Queue()
        fragmentos: list[str] = []
        buffer = ""
        hablando = False

        def _synthesizar(pieza: str) -> None:
            limpia = _limpiar_para_voz(pieza)
            if not limpia:
                return
            ruta = f"/tmp/sia_{uuid.uuid4().hex}{ext}"
            sintetizadas.put_nowait(asyncio.create_task(tts.synthesize(limpia, ruta)))

        # La VOZ VA PRIMERO: un emisor en segundo plano envía cada oración al
        # navegador apenas queda lista, sin esperar ni al LLM ni al texto.
        emisor = asyncio.create_task(self._emitir_audio(sintetizadas))

        try:
            async for delta in session.ask_stream(texto):
                if not hablando:
                    hablando = True
                    await self._ws.send_json({"type": "speaking"})
                fragmentos.append(delta)
                buffer += delta
                while True:
                    oracion, buffer = _extract_sentence(buffer)
                    if oracion is None:
                        break
                    _synthesizar(oracion)
            if buffer.strip():
                _synthesizar(buffer.strip())
        finally:
            # El texto va al final de la misma cola: los fragmentos ya lanzados
            # suenan primero y el orden del WebSocket queda determinista.
            await sintetizadas.put(
                ("texto", {"type": "response", "text": "".join(fragmentos).strip()})
            )
            await sintetizadas.put(None)  # cierra el turno de habla
            await asyncio.gather(emisor, return_exceptions=True)

    async def _emitir_audio(self, sintetizadas: asyncio.Queue) -> None:
        """Envía audios y texto EN ORDEN según van quedando listos."""
        enviados = 0
        while True:
            item = await sintetizadas.get()
            if item is None:
                break
            if isinstance(item, tuple):
                _, mensaje = item
                await self._ws.send_json(mensaje)
                continue
            try:
                path = await item
                audio = await asyncio.to_thread(path.read_bytes)
                await asyncio.to_thread(path.unlink, True)
                await self._ws.send_json(
                    {
                        "type": "audio_chunk",
                        "audio": base64.b64encode(audio).decode("ascii"),
                        "mime": self._mime_actual,
                        "last": False,
                    }
                )
                enviados += 1
            except Exception as exc:  # noqa: BLE001 - un fragmento no corta la voz
                logger.warning("Fragmento de audio descartado: %s", exc)
        # Cierra el turno de habla en la GUI aunque no hubiera fragmentos.
        await self._ws.send_json({"type": "audio_end"})
        logger.info("Respuesta de voz enviada en %d fragmento(s)", enviados)

    @staticmethod
    def _decode_to_pcm(webm: bytes) -> bytes:
        """Convierte audio del navegador (webm/opus) a PCM crudo 16 kHz mono.

        Se usa ``s16le`` (PCM sin cabecera) porque ``transcribe_bytes`` envuelve
        los bytes directamente en ``sr.AudioData``, que no espera cabecera WAV.
        """
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-ar",
                str(_SAMPLE_RATE),
                "-ac",
                "1",
                "-f",
                "s16le",
                "pipe:1",
            ],
            input=webm,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            # Diagnóstico: registrar tamaño y "firma" del blob rechazado
            # (1A45DFA3=webm, 66747970=mp4, vacío=captura sin datos).
            firma = webm[:12].hex() if webm else "(vacío)"
            logger.warning(
                "Audio rechazado por ffmpeg: %d bytes, firma %s", len(webm), firma
            )
            _guardar_audio_rechazado(webm)
            raise VoiceError(f"ffmpeg: {result.stderr.decode(errors='replace')[:200]}")
        return result.stdout

    async def _confirm(self, call) -> bool:
        # En la interfaz de voz las herramientas se auto-aprueban (sin diálogo).
        return True

    async def _receive_loop(self) -> None:
        while True:
            raw = await self._ws.receive_json()
            if raw.get("type") == "audio":
                try:
                    self._queue.put_nowait(base64.b64decode(raw.get("data", "")))
                except ValueError:
                    logger.warning("Audio base64 inválido")
            elif raw.get("type") == "text":
                texto = str(raw.get("data", "")).strip()[:2000]
                if texto:
                    self._queue.put_nowait(("texto", texto))