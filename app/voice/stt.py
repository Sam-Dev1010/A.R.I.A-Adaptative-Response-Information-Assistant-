"""Speech-to-Text con el micrófono (FASE 5).

Dos motores:
- ``GoogleSTTProvider``: micrófono local (sounddevice) + servicio gratuito de Google.
- ``GroqSTTProvider``: Whisper-turbo por API (~0.3 s, ideal para la interfaz web).
Las importaciones pesadas son perezosas para que el núcleo funcione sin
hardware de audio instalado.
"""
import asyncio
import logging
import time
from dataclasses import dataclass

from app.core.config import Settings, get_settings
from app.voice.base import STTProvider, VoiceError

logger = logging.getLogger("sia.voice")

_DEFAULT_BLOCK_MS = 50


@dataclass
class _CaptureConfig:
    sample_rate: int = 16000
    block_size: int = 800  # 50 ms a 16 kHz
    silence_threshold: float = 0.008  # RMS base bajo el cual se considera silencio
    calibration_seconds: float = 0.4  # se ajusta el umbral al ruido ambiente real
    silence_limit_seconds: float = 0.9  # pausa que corta la captura
    phrase_time_limit: float = 5.0  # duración máxima de la frase


class GoogleSTTProvider(STTProvider):
    """Transcripción de voz usando el reconocedor gratuito de Google."""

    name = "google_stt"

    def __init__(
        self,
        *,
        language: str = "es-ES",
        capture: _CaptureConfig | None = None,
    ) -> None:
        self._language = language
        self._capture = capture or _CaptureConfig()
        self._recognizer = None

    def _get_recognizer(self):
        if self._recognizer is None:
            import speech_recognition as sr

            self._recognizer = sr.Recognizer()
            self._recognizer.energy_threshold = 300
        return self._recognizer

    async def listen(self, *, language: str | None = None) -> str:
        """Graba desde el micrófono y devuelve el texto (o levanta VoiceError)."""
        audio = await asyncio.to_thread(self._record)
        text = await asyncio.to_thread(
            self._transcribe, audio, language or self._language
        )
        logger.info("Voz transcrita", extra={"chars": len(text)})
        return text

    def transcribe_bytes(
        self,
        raw: bytes,
        *,
        sample_rate: int = 16000,
        language: str | None = None,
    ) -> str:
        """Transcribe audio PCM crudo (16 bits) capturado fuera del micrófono.

        Útil para la interfaz web: el navegador graba, el servidor transcribe.
        """
        import speech_recognition as sr

        audio = sr.AudioData(raw, sample_rate, 2)
        return self._transcribe(audio, language or self._language)

    def _record(self):
        """Captura audio del micrófono hasta silencio o límite de tiempo.

        El umbral de silencio se auto-calibra con el ruido ambiente real de los
        primeros instantes, para cortar la frase tan pronto como el usuario
        hace una pausa.
        """
        import sounddevice as sd

        capture = self._capture
        blocks: list[bytes] = []
        quiet_blocks = 0
        calibrating = True
        calibration: list[float] = []
        threshold = capture.silence_threshold
        started = time.monotonic()
        max_seconds = capture.phrase_time_limit
        calibration_blocks = max(
            1, int(capture.calibration_seconds * capture.sample_rate / capture.block_size)
        )

        def callback(indata, frames, time_info, status):
            nonlocal quiet_blocks, calibrating, threshold
            blocks.append(indata.copy().tobytes())
            rms = float((indata.astype("float32") ** 2).mean() ** 0.5)
            if calibrating:
                calibration.append(rms)
                if len(calibration) >= calibration_blocks:
                    calibrating = False
                    ambient = sum(calibration) / len(calibration)
                    threshold = max(capture.silence_threshold, ambient * 2.0)
                    quiet_blocks = 0
                return
            quiet_blocks = quiet_blocks + 1 if rms < threshold else 0

        with sd.InputStream(
            samplerate=capture.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=capture.block_size,
            callback=callback,
        ):
            while True:
                time.sleep(_DEFAULT_BLOCK_MS / 1000)
                elapsed = time.monotonic() - started
                quiet_seconds = quiet_blocks * capture.block_size / capture.sample_rate
                if elapsed >= max_seconds or quiet_seconds >= capture.silence_limit_seconds:
                    break

        raw = b"".join(blocks) or b"\x00\x00" * capture.block_size
        import speech_recognition as sr

        return sr.AudioData(raw, capture.sample_rate, 2)

    def _transcribe(self, audio, language: str) -> str:
        import speech_recognition as sr

        recognizer = self._get_recognizer()
        try:
            return recognizer.recognize_google(audio, language=language)
        except sr.UnknownValueError as exc:
            raise VoiceError("No se entendió el audio") from exc
        except sr.RequestError as exc:
            raise VoiceError(f"El servicio de transcripción no respondió: {exc}") from exc


class GroqSTTProvider(STTProvider):
    """Transcripción con Whisper-turbo vía API compatible OpenAI (Groq).

    ~0.3 s por frase frente a 1-2 s del servicio gratuito de Google: la voz
    de SIA responde casi un segundo antes. Requiere ``STT_GROQ_API_KEY``
    (gratis en https://console.groq.com/keys). Acepta PCM crudo igual que
    ``GoogleSTTProvider.transcribe_bytes`` y lo envuelve en WAV para la API.
    """

    name = "groq_stt"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.groq.com/openai/v1",
        model: str = "whisper-large-v3-turbo",
        language: str = "es-ES",
    ) -> None:
        if not api_key:
            raise VoiceError("GroqSTTProvider necesita una API key")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._language = language

    @staticmethod
    def _wav_bytes(raw: bytes, sample_rate: int) -> bytes:
        """Envuelve PCM crudo (s16le mono) en un contenedor WAV."""
        import io
        import wave

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(raw)
        return buffer.getvalue()

    def _request(self, wav: bytes, language: str) -> str:
        import httpx

        try:
            response = httpx.post(
                f"{self._base_url}/audio/transcriptions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                files={"file": ("audio.wav", wav, "audio/wav")},
                data={
                    "model": self._model,
                    "language": language.split("-")[0].lower(),
                    "response_format": "json",
                    "temperature": "0",
                },
                timeout=15.0,
            )
        except Exception as exc:
            raise VoiceError(f"Groq STT no respondió: {exc}") from exc
        if response.status_code != 200:
            raise VoiceError(
                f"Groq STT respondió {response.status_code}: "
                f"{response.text[:150]}"
            )
        return str(response.json().get("text", "")).strip()

    def transcribe_bytes(
        self,
        raw: bytes,
        *,
        sample_rate: int = 16000,
        language: str | None = None,
    ) -> str:
        """Transcribe audio PCM crudo (16 bits) capturado fuera del micrófono."""
        if not raw:
            raise VoiceError("No se entendió el audio")
        wav = self._wav_bytes(raw, sample_rate)
        texto = self._request(wav, language or self._language)
        logger.info("Voz transcrita con Groq", extra={"chars": len(texto)})
        return texto

    async def listen(self, *, language: str | None = None) -> str:
        raise VoiceError(
            "GroqSTT no captura micrófono: usa la interfaz web o el satélite"
        )


def build_stt_provider(settings: Settings | None = None) -> STTProvider:
    """Construye el motor de transcripción según STT_PROVIDER.

    - ``google`` (defecto): servicio gratuito de Google.
    - ``groq``: Whisper-turbo (~0.3 s). Si no hay key propia, reutiliza
      LLM_FALLBACK_API_KEY cuando el respaldo LLM apunta a Groq.
    Cualquier otro valor o motor roto → Google.
    """
    settings = settings or get_settings()
    mode = (settings.stt_provider or "google").strip().lower()
    idioma = settings.stt_language
    if mode == "groq":
        api_key = (
            settings.stt_groq_api_key.strip()
            or (
                settings.llm_fallback_api_key.strip()
                if "groq" in (settings.llm_fallback_base_url or "").lower()
                else ""
            )
        )
        if api_key:
            try:
                return GroqSTTProvider(
                    api_key=api_key,
                    base_url=settings.stt_groq_base_url,
                    model=settings.stt_groq_model,
                    language=idioma,
                )
            except VoiceError:
                logger.exception("No pude construir GroqSTT; uso Google")
        else:
            logger.warning(
                "STT_PROVIDER=groq sin API key; usando el servicio de Google"
            )
    return GoogleSTTProvider(language=idioma)