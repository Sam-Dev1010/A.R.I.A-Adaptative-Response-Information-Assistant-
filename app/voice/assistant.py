"""Conversación por voz integrada (FASE 7).

Une STT + LLM + TTS: escucha al usuario, procesa con el orquestador y
responde hablando. A.R.I.A vive en **espera**: cualquier frase que empiece
con la palabra de activación la despierta ("ARIA", "ARIA buenos días",
"ARIA qué hora es"…). Mientras está activa conversa libremente, y se apaga
con "ARIA ya acabamos gracias". El gancho ``on_wake`` se ejecuta tras la
activación (p. ej., para abrir la interfaz gráfica).
"""
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime

from app.ai.orchestrator import AssistantOrchestrator
from app.voice.base import STTProvider, TTSProvider, VoiceError

logger = logging.getLogger("sia.voice")

_DEFAULT_WAKE_WORD = "aria"
_ACTIVATION_SALUTATIONS = ("hola", "buenos días", "buenos dias", "buenas tardes", "buenas noches")
_DEACTIVATION_PHRASES = ("ya acabamos", "apágate", "apagate", "adiós", "adios")
_EXIT_PHRASES = {"salir", "exit", "quit"}


def greeting_for(hour: int) -> str:
    """Saludo de A.R.I.A según la hora del día (5-11: días, 12-19: tardes, resto: noches)."""
    if 5 <= hour < 12:
        part = "¡Buenos días!"
    elif 12 <= hour < 20:
        part = "¡Buenas tardes!"
    else:
        part = "¡Buenas noches!"
    return f"{part} Soy A.R.I.A, ¿en qué puedo ayudarle, jefe?"


class ExitConversation(Exception):
    """El usuario pidió terminar la sesión de voz."""


class VoiceAssistant:
    """Ciclo completo voz → texto → LLM → texto → voz."""

    def __init__(
        self,
        orchestrator: AssistantOrchestrator,
        stt: STTProvider,
        tts: TTSProvider,
        *,
        wake_word: str | None = _DEFAULT_WAKE_WORD,
        language: str = "es-ES",
        active: bool | None = None,
        on_wake: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._stt = stt
        self._tts = tts
        self._wake_word = (wake_word or "").strip().lower() or None
        self._language = language
        if active is None:
            active = self._wake_word is None
        self._active = active
        self._on_wake = on_wake

    @property
    def wake_word(self) -> str | None:
        return self._wake_word

    @property
    def active(self) -> bool:
        """True = conversación activa; False = en espera (solo activación por voz)."""
        return self._active

    def _is_deactivation(self, command: str) -> bool:
        return any(phrase in command for phrase in _DEACTIVATION_PHRASES)

    def _detect_salutation(self, command: str) -> str | None:
        lowered = command.lower()
        for salutation in _ACTIVATION_SALUTATIONS:
            if salutation in lowered:
                return salutation
        return None

    async def _activate(self, command: str) -> str:
        """Activa a A.R.I.A con el saludo apropiado y devuelve lo dicho en voz."""
        salutation = self._detect_salutation(command)
        if salutation:
            greeting = f"¡{salutation.capitalize()}! Soy A.R.I.A, ¿en qué puedo ayudarle, jefe?"
        else:
            greeting = greeting_for(datetime.now().astimezone().hour)
        self._active = True
        logger.info("A.R.I.A activada")
        await self._tts.speak(greeting)
        return greeting

    async def _deactivate(self) -> str:
        """Pone a A.R.I.A en espera y devuelve la despedida dicha en voz."""
        self._active = False
        logger.info("A.R.I.A en espera")
        goodbye = "¡Hasta luego! Estaré en espera."
        await self._tts.speak(goodbye)
        return goodbye

    async def _answer(self, command: str) -> str:
        response = await self._orchestrator.ask(command)
        await self._tts.speak(response.content)
        return response.content

    async def run_once(self, *, text: str | None = None) -> str | None:
        """Una interacción completa: escuchar (opcional), responder hablando.

        - Devuelve la respuesta de A.R.I.A como texto.
        - Devuelve None si no había nada que responder (en espera o sin activación).
        - Lanza :class:`ExitConversation` si el usuario pide salir del programa.
        """
        if text is None:
            try:
                text = await self._stt.listen(language=self._language)
            except VoiceError as exc:
                logger.warning("STT falló", extra={"error": str(exc)})
                return None

        if not text or not text.strip():
            return None

        raw = text.strip()
        wake = self._wake_word
        has_wake = wake is not None and raw.lower().startswith(wake)
        command = raw[len(wake) :].strip().strip(" ,;:.") if has_wake else raw

        if self._active:
            if command.lower() in _EXIT_PHRASES:
                raise ExitConversation
            if self._is_deactivation(command):
                return await self._deactivate()
            return await self._answer(command)

        if not has_wake:
            return None

        # "ARIA <saludo>" activa y saluda; "ARIA <petición>" activa, responde
        # y además dispara el gancho de activación (abrir la interfaz).
        greeting = await self._activate(command)
        answered = None
        if command and not self._detect_salutation(command):
            answered = await self._answer(command)
        if self._on_wake is not None:
            await self._on_wake()
        return answered or greeting

    async def run_loop(self) -> None:
        """Escucha y responde indefinidamente hasta que el usuario se despida."""
        logger.info("Asistente de voz activo", extra={"wake_word": self._wake_word})
        while True:
            try:
                await self.run_once()
            except ExitConversation:
                logger.info("Sesión de voz terminada por el usuario")
                break