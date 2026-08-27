"""Text-to-Speech: Piper local (instantáneo) y Microsoft Edge en la nube.

``PiperTTSProvider`` sintetiza en el propio PC (~0.3 s, sin internet).
``EdgeTTSProvider`` usa voces neuronales de Microsoft sin API key.
La reproducción usa el primer reproductor disponible del sistema, siempre en
modo silencioso (sin ventanas): ``paplay`` (nativo de PipeWire), ``ffplay``
con ``-nodisp``, ``mpv`` con ``--no-video`` o ``aplay``.
"""
import asyncio
import logging
import re
import shlex
import shutil
import subprocess
import uuid
from pathlib import Path

from app.core.config import Settings, get_settings
from app.voice.base import TTSProvider, VoiceError

logger = logging.getLogger("sia.voice")

_DEFAULT_PLAYERS = (
    "paplay",
    "ffplay -nodisp -autoexit -loglevel quiet",
    "mpv --no-video --really-quiet",
    "aplay -q",
)


def play_audio_file(path: Path, player_cmds: tuple[str, ...] | None = None) -> None:
    """Reproduce un archivo de audio con el primer reproductor disponible."""
    for command in player_cmds or _DEFAULT_PLAYERS:
        parts = shlex.split(command)
        if not parts or not shutil.which(parts[0]):
            continue
        try:
            subprocess.run(
                [*parts, str(path)],
                check=False,
                capture_output=True,
                timeout=60,
            )
            return
        except (subprocess.TimeoutExpired, OSError):
            continue
    logger.warning(
        "Ningún reproductor disponible; el audio no se reproduce",
        extra={"file": str(path)},
    )


class EdgeTTSProvider(TTSProvider):
    """Síntesis de voz con voces neuronales de Microsoft Edge.

    Con ``vary_rate`` activado, cada frase se sintetiza con ritmo Y tono
    ligeramente distintos (±2% y ±2 Hz alrededor del base): la entonación
    sube y baja como al hablar de verdad y deja de sonar monocorde.
    """

    name = "edge_tts"

    _VARIACIONES = (-2, 0, 1, -1, 2)  # puntos porcentuales sobre el ritmo base
    _VARIACIONES_HZ = (-2, -1, 1, 0, 2)  # hertz sobre el tono base

    def __init__(
        self,
        *,
        voice: str = "es-MX-DaliaNeural",
        rate: str = "+12%",
        pitch: str = "+12Hz",
        player_cmds: tuple[str, ...] | None = None,
        vary_rate: bool = False,
    ) -> None:
        self._voice = voice
        self._rate = rate
        self._pitch = pitch
        self._player_cmds = player_cmds or _DEFAULT_PLAYERS
        self._vary_rate = vary_rate
        self._variacion_idx = 0
        self._pitch_idx = 1  # fase distinta a la del ritmo: combinaciones variadas

    @property
    def voice(self) -> str:
        return self._voice

    def _siguiente_rate(self) -> str:
        """Ritmo de esta frase: base +/- variación sutil en cascada."""
        if not self._vary_rate:
            return self._rate
        try:
            valor = int(self._rate.rstrip("%"))
        except ValueError:
            return self._rate
        delta = self._VARIACIONES[self._variacion_idx % len(self._VARIACIONES)]
        self._variacion_idx += 1
        return f"{valor + delta:+d}%"

    def _siguiente_pitch(self) -> str:
        """Tono de esta frase: base +/- unos hertz, en cascada independiente."""
        if not self._vary_rate:
            return self._pitch
        match = re.fullmatch(r"([+-]?\d+)Hz", self._pitch.strip())
        if match is None:
            return self._pitch
        base = int(match.group(1))
        delta_hz = self._VARIACIONES_HZ[self._pitch_idx % len(self._VARIACIONES_HZ)]
        self._pitch_idx += 1
        return f"{base + delta_hz:+d}Hz"

    async def synthesize(self, text: str, output_path: Path | str) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import edge_tts

            communicate = edge_tts.Communicate(
                text,
                self._voice,
                rate=self._siguiente_rate(),
                pitch=self._siguiente_pitch(),
            )
            await communicate.save(str(path))
        except Exception as exc:
            raise VoiceError(f"No se pudo sintetizar el audio: {exc}") from exc
        return path

    async def speak(self, text: str, *, output_dir: Path | str | None = None) -> Path:
        directory = Path(output_dir) if output_dir else Path("data/tts")
        path = await self.synthesize(text, directory / f"{uuid.uuid4().hex}.mp3")
        await asyncio.to_thread(self._play, path)
        return path

    def _play(self, path: Path) -> None:
        """Reproduce el audio con el primer reproductor disponible, sin ventanas."""
        play_audio_file(path, self._player_cmds)


class PiperTTSProvider(TTSProvider):
    """Síntesis local con Piper: instantánea (~0.3 s) y sin internet."""

    name = "piper_tts"
    audio_ext = ".wav"

    def __init__(
        self,
        model_path: Path | str,
        *,
        player_cmds: tuple[str, ...] | None = None,
    ) -> None:
        self._model_path = Path(model_path)
        self._player_cmds = player_cmds or _DEFAULT_PLAYERS

    @property
    def model_path(self) -> Path:
        return self._model_path

    def _run_piper(self, text: str, path: Path) -> None:
        binary = shutil.which("piper")
        if not binary:
            raise VoiceError("Piper no está instalado en el sistema")
        result = subprocess.run(
            [binary, "--model", str(self._model_path), "--output-file", str(path)],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0 or not path.exists():
            raise VoiceError(
                f"piper: {result.stderr.decode(errors='replace')[:200]}"
            )

    async def synthesize(self, text: str, output_path: Path | str) -> Path:
        if not self._model_path.exists():
            raise VoiceError(f"Modelo de Piper no encontrado: {self._model_path}")
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            await asyncio.to_thread(self._run_piper, text, path)
        except VoiceError:
            raise
        except Exception as exc:
            raise VoiceError(f"No se pudo sintetizar el audio: {exc}") from exc
        return path

    async def speak(self, text: str, *, output_dir: Path | str | None = None) -> Path:
        directory = Path(output_dir) if output_dir else Path("data/tts")
        path = await self.synthesize(text, directory / f"{uuid.uuid4().hex}.wav")
        await asyncio.to_thread(play_audio_file, path, self._player_cmds)
        return path


class GoogleTranslateTTSProvider(TTSProvider):
    """TTS gratuito de Google Translate: rápido, sin API key (≤200 chars/petición)."""

    name = "gtts"
    audio_ext = ".mp3"

    _URL = "https://translate.google.com/translate_tts"
    _MAX_CHARS = 200

    def __init__(
        self,
        *,
        language: str = "es",
        player_cmds: tuple[str, ...] | None = None,
    ) -> None:
        self._language = language
        self._player_cmds = player_cmds or _DEFAULT_PLAYERS

    @staticmethod
    def _split_for_request(text: str) -> list[str]:
        """Trocea el texto en peticiones de ≤200 chars cortando por espacios."""
        pieces: list[str] = []
        restante = text.strip()
        while len(restante) > GoogleTranslateTTSProvider._MAX_CHARS:
            corte = restante.rfind(" ", 0, GoogleTranslateTTSProvider._MAX_CHARS)
            if corte <= 0:
                corte = GoogleTranslateTTSProvider._MAX_CHARS
            pieces.append(restante[:corte].strip())
            restante = restante[corte:].strip()
        if restante:
            pieces.append(restante)
        return pieces

    async def synthesize(self, text: str, output_path: Path | str) -> Path:
        import httpx

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _fetch() -> bytes:
            audio = b""
            for piece in self._split_for_request(text):
                response = httpx.get(
                    self._URL,
                    params={
                        "ie": "UTF-8",
                        "q": piece,
                        "tl": self._language,
                        "client": "tw-ob",
                    },
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=10.0,
                )
                if response.status_code != 200 or not response.content:
                    raise VoiceError(f"gtts respondió {response.status_code}")
                audio += response.content
            return audio

        try:
            data = await asyncio.to_thread(_fetch)
        except VoiceError:
            raise
        except Exception as exc:
            raise VoiceError(f"No se pudo sintetizar con gtts: {exc}") from exc
        path.write_bytes(data)
        return path

    async def speak(self, text: str, *, output_dir: Path | str | None = None) -> Path:
        directory = Path(output_dir) if output_dir else Path("data/tts")
        path = await self.synthesize(text, directory / f"{uuid.uuid4().hex}.mp3")
        await asyncio.to_thread(play_audio_file, path, self._player_cmds)
        return path


class RaceTTSProvider(TTSProvider):
    """Lanza varios proveedores a la vez y usa el primero que termine bien."""

    name = "race_tts"
    audio_ext = ".mp3"

    def __init__(
        self,
        *providers: TTSProvider,
        player_cmds: tuple[str, ...] | None = None,
    ) -> None:
        if not providers:
            raise ValueError("RaceTTSProvider necesita al menos un proveedor")
        self._providers = providers
        self._player_cmds = player_cmds or _DEFAULT_PLAYERS

    async def synthesize(self, text: str, output_path: Path | str) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        temporales = {
            provider: out.with_name(
                f"{out.stem}_{provider.name}{provider.audio_ext}"
            )
            for provider in self._providers
        }
        tasks = {
            asyncio.create_task(
                provider.synthesize(text, temporal)
            ): (provider, temporal)
            for provider, temporal in temporales.items()
        }
        pending = set(tasks)
        ultimo_error: Exception | None = None
        try:
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    try:
                        task.result()
                    except Exception as exc:  # noqa: BLE001 - la carrera continúa
                        ultimo_error = exc
                        continue
                    _ganador, ruta_ganadora = tasks[task]
                    shutil.copyfile(ruta_ganadora, out)
                    return out
        finally:
            for task in pending:
                task.cancel()
            for temporal in temporales.values():
                await asyncio.to_thread(temporal.unlink, True)
        raise VoiceError(
            f"Ningún proveedor de voz respondió: {ultimo_error}"
        ) from ultimo_error

    async def speak(self, text: str, *, output_dir: Path | str | None = None) -> Path:
        directory = Path(output_dir) if output_dir else Path("data/tts")
        path = await self.synthesize(text, directory / f"{uuid.uuid4().hex}.mp3")
        await asyncio.to_thread(play_audio_file, path, self._player_cmds)
        return path


def build_tts_provider(settings: Settings | None = None) -> TTSProvider:
    """Construye el motor de voz según TTS_PROVIDER.

    ``auto`` usa Edge (voz neuronal, la más parecida a una persona);
    ``gtts``/``edge``/``piper`` fuerzan un motor concreto.
    Google Translate ya no participa del modo automático porque suena robótica.
    """
    settings = settings or get_settings()
    mode = (settings.tts_provider or "auto").strip().lower() or "auto"
    edge = EdgeTTSProvider(
        voice=settings.tts_voice,
        rate=settings.tts_rate,
        pitch=settings.tts_pitch,
        vary_rate=True,  # entonación viva: cada frase con ritmo y tono sutilmente distintos
    )
    if mode == "piper":
        if shutil.which("piper") and settings.piper_model.expanduser().exists():
            logger.info("TTS local activo: Piper (%s)", settings.piper_model)
            return PiperTTSProvider(settings.piper_model)
        logger.warning("Piper no disponible (%s); usando Edge", settings.piper_model)
        return edge
    if mode == "gtts":
        return GoogleTranslateTTSProvider()
    # auto y edge: voz neuronal de Microsoft (natural, tono humano)
    return edge