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

# Palabras que delatan una orden/acción (para vetar a quién no está autorizado).
_ORDER_VERBS = (
    "ejecuta", "ejecutar", "corre", "correr", "abre", "abrir", "crea", "crear",
    "instala", "instalar", "actualiza", "actualizar", "borra", "borrar",
    "elimina", "eliminar", "compila", "compilar", "depura", "depurar",
    "lista", "listar", "muestra", "mostrar", "guarda", "guardar",
)

# Patrones de registro de voz: "acuérdate de mi voz como X" / "soy X"
_REGISTRATION_MARKERS = (
    "acuérdate de mi voz", "acuerdate de mi voz", "reconóceme como",
    "reconceme como", "regístrate mi voz", "registrate mi voz",
)
# Muestras de voz necesarias para tener una huella fiable.
_MIN_SAMPLES_TO_REGISTER = 3


def greeting_for(hour: int, speaker: str | None = None) -> str:
    """Saludo de A.R.I.A según la hora y, si se conoce, el nombre del hablante."""
    if 5 <= hour < 12:
        part = "¡Buenos días!"
    elif 12 <= hour < 20:
        part = "¡Buenas tardes!"
    else:
        part = "¡Buenas noches!"
    if speaker:
        return f"{part} Soy A.R.I.A. ¿En qué puedo ayudarle, {speaker}?"
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
        speaker_manager=None,
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
        self._speaker_manager = speaker_manager
        self._speaker: str | None = None          # quién se identificó por voz
        self._registration: str | None = None     # nombre en proceso de registro
        self._registration_done: bool = False     # muestras suficientes

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
            sal = salutation.capitalize()
            greeting = (
                f"¡{sal}! Soy A.R.I.A. ¿En qué puedo ayudarle, {self._speaker}?"
                if self._speaker
                else f"¡{sal}! Soy A.R.I.A, ¿en qué puedo ayudarle, jefe?"
            )
        else:
            greeting = greeting_for(datetime.now().astimezone().hour, self._speaker)
        self._active = True
        logger.info("A.R.I.A activada", extra={"speaker": self._speaker})
        await self._tts.speak(greeting)
        return greeting

    async def _identify_speaker(self, audio: bytes | None) -> str | None:
        """Identifica por voz a quién habla; devuelve None si no está registrado."""
        if self._speaker_manager is None or not audio:
            return None
        try:
            name, _score = await self._speaker_manager.identify(audio)
        except VoiceError as exc:
            logger.info("No pude identificar la voz", extra={"error": str(exc)})
            return None
        if name:
            logger.info("Hablante identificado", extra={"speaker": name})
        return name

    def _is_order(self, command: str) -> bool:
        lowered = command.lower().lstrip()
        return any(lowered.startswith(verb) for verb in _ORDER_VERBS)

    def _parse_registration_name(self, command: str) -> str | None:
        """Extrae el nombre de un pedido de registro ('... como Samuel' / 'soy Samuel')."""
        lowered = command.lower()
        if any(marker in lowered for marker in _REGISTRATION_MARKERS) and "como" in lowered:
            resto = command.split("como", 1)[1].strip()
            if resto:
                candidate = resto.split()[0].strip(".,;:!")
                if candidate:
                    return candidate.capitalize()
        lowered_words = lowered.split()
        for i, w in enumerate(lowered_words):
            if w == "soy" and i + 1 < len(lowered_words):
                return lowered_words[i + 1].strip(".,;:!").capitalize()
        return None

    async def _deactivate(self) -> str:
        """Pone a A.R.I.A en espera y devuelve la despedida dicha en voz."""
        self._active = False
        self._registration = None
        self._registration_done = False
        logger.info("A.R.I.A en espera")
        goodbye = "¡Hasta luego! Estaré en espera."
        await self._tts.speak(goodbye)
        return goodbye

    async def _answer(self, command: str) -> str:
        # Valida que el hablante está autorizado a dar órdenes (si lo gestionamos).
        if (
            self._speaker_manager is not None
            and self._speaker
            and self._is_order(command)
            and not self._speaker_manager.is_authority(self._speaker)
        ):
            denied = (
                f"Lo siento, {self._speaker}. Usted está identificado pero no "
                "está autorizado a darme órdenes."
            )
            await self._tts.speak(denied)
            return denied
        response = await self._orchestrator.ask(command)
        await self._tts.speak(response.content)
        return response.content

    async def _capture(self) -> tuple[str, bytes | None]:
        """Escucha y devuelve (texto, audio PCM) — audio solo si se puede capturar."""
        audio: bytes | None = None
        capture_fn = getattr(self._stt, "listen_with_audio", None)
        if capture_fn is not None and self._speaker_manager is not None:
            text, audio = await capture_fn(language=self._language)
            audio = bytes(audio) if audio else None
        else:
            text = await self._stt.listen(language=self._language)
        return text, audio

    async def _confirm_registration(self, name: str) -> str:
        """Completa el registro y lo anuncia; marca como autoridad al dueño."""
        is_owner = (
            name.lower()
            == (self._speaker_manager.default_authority or "").lower()
        )
        if is_owner:
            await self._speaker_manager.set_authority(name, enabled=True)
        self._registration = None
        self._registration_done = True
        msg = (
            f"Listo, {name}. Ya tengo su voz guardada"
            + (", y quedará autorizado a darme órdenes." if is_owner else ".")
        )
        await self._tts.speak(msg)
        return msg

    async def _handle_registration(self, name: str, audio: bytes | None) -> str:
        """Gestiona el registro de una huella de voz (varias frases)."""
        if self._speaker_manager is None:
            return "No tengo activado el reconocimiento de voces."
        if not audio:
            denied = "No pude capturar su voz. Inténtelo de nuevo."
            await self._tts.speak(denied)
            return denied

        await self._speaker_manager.add_sample(name, audio)
        n = self._speaker_manager.sample_count(name)
        if n >= _MIN_SAMPLES_TO_REGISTER:
            return await self._confirm_registration(name)

        self._registration = name
        pendientes = _MIN_SAMPLES_TO_REGISTER - n
        pedi = (
            f"Muy bien, {name}. Para memorizar mejor su voz, diga "
            f"{'otra' if n > 1 else 'una'} frase más."
            if pendientes == 1
            else f"Muy bien, {name}. Diga {pendientes} frases más para terminar."
        )
        await self._tts.speak(pedi)
        return pedi

    async def run_once(self, *, text: str | None = None) -> str | None:
        """Una interacción completa: escuchar (opcional), responder hablando.

        - Devuelve la respuesta de A.R.I.A como texto.
        - Devuelve None si no había nada que responder (en espera o sin activación).
        - Lanza :class:`ExitConversation` si el usuario pide salir del programa.
        """
        audio: bytes | None = None
        if text is None:
            try:
                text, audio = await self._capture()
            except VoiceError as exc:
                logger.warning("STT falló", extra={"error": str(exc)})
                return None

        if not text or not text.strip():
            return None

        raw = text.strip()
        wake = self._wake_word
        has_wake = wake is not None and raw.lower().startswith(wake)
        command = raw[len(wake) :].strip().strip(" ,;:.") if has_wake else raw

        # Saber quién está hablando (si tenemos audio y reconocimiento).
        self._speaker = await self._identify_speaker(audio)

        # Registro en curso: cada frase que diga añade una muestra de su voz.
        if self._registration:
            return await self._handle_registration(self._registration, audio)

        if self._speaker_manager is not None:
            reg_name = self._parse_registration_name(command)
            if reg_name and audio:
                return await self._handle_registration(reg_name, audio)
            if reg_name and (not self._speaker_manager.has_profile(reg_name)):
                denied = (
                    "No pude capturar su voz para registrarla. Repítalo, por favor."
                )
                await self._tts.speak(denied)
                return denied

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