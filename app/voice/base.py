"""Contrato de los proveedores de voz (FASES 5-6)."""
from abc import ABC, abstractmethod
from pathlib import Path


class VoiceError(RuntimeError):
    """Error de la capa de voz (captura, transcripción o síntesis)."""


class STTProvider(ABC):
    """Convierte voz en texto (Speech-to-Text)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identificador del proveedor (ej: ``google_stt``)."""

    @abstractmethod
    async def listen(self, *, language: str | None = None) -> str:
        """Captura audio del micrófono y devuelve el texto transcrito.

        Levanta :class:`VoiceError` si no se entiende el audio o falla el servicio.
        """

    async def aclose(self) -> None:
        """Libera los recursos del proveedor (idempotente)."""


class TTSProvider(ABC):
    """Convierte texto en voz (Text-to-Speech)."""

    audio_ext = ".mp3"  # extensión que genera synthesize (para el MIME)

    @property
    @abstractmethod
    def name(self) -> str:
        """Identificador del proveedor (ej: ``edge_tts``)."""

    @abstractmethod
    async def synthesize(self, text: str, output_path: Path | str) -> Path:
        """Genera un archivo de audio con el texto y devuelve su ruta."""

    @abstractmethod
    async def speak(self, text: str, *, output_dir: Path | str | None = None) -> Path:
        """Sintetiza el texto y lo reproduce por el altavoz. Devuelve el audio."""

    async def aclose(self) -> None:
        """Libera los recursos del proveedor (idempotente)."""