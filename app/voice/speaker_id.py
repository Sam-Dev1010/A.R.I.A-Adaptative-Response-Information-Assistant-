"""Identificación del hablante (quién le habla a A.R.I.A) — 100 % local.

Usa :mod:`speakeronnx` (modelo WeSpeaker/ECAPA en onnxruntime, solo CPU). Cada
persona registra su "huella de voz" (varios embeddings) y A.R.I.A puede:

- **Identificar (1:N)**: saber qué usuario registrado está hablando.
- **Verificar (1:1)**: confirmar que el hablante está autorizado a dar órdenes.

Todo se procesa y se guarda en el disco local (privacidad total, sin nube).
El modelo se carga de forma perezosa para no penalizar el arranque cuando la
identificación no está en uso.
"""
import asyncio
import json
import logging
from pathlib import Path

import numpy as np

from app.voice.base import VoiceError

logger = logging.getLogger("sia.voice")

# Modelo por defecto de speakeronnx (WeSpeaker ResNet34, ~26 MB, Apache-2.0).
_DEFAULT_MODEL = "wespeaker-resnet34"
# Sample rate que espera el extractor (16 kHz), mismo que usa el micrófono STT.
_SAMPLE_RATE = 16000

# Similitud coseno mínima para considerar a la misma persona y descartar a
# desconocidos. Para identificación 1:N conviene un umbral alto: con voz real,
# la misma persona ronda 0.7-0.9, mientras que otras rondan 0.3-0.5.
_DEFAULT_THRESHOLD = 0.68
# Autoridad por defecto si no se completa el registro del dueño.
_DEFAULT_AUTHORITY = "Samuel"


def raw_pcm_to_f32(raw: bytes) -> np.ndarray:
    """Convierte PCM crudo (s16le mono a 16 kHz) en un array float32 [-1, 1]."""
    if not raw:
        raise VoiceError("No hay audio para analizar")
    data = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
    if data.size < _SAMPLE_RATE:  # menos de un segundo: poco útil para una huella
        raise VoiceError("Audio demasiado corto para identificar la voz")
    return data


class SpeakerIdManager:
    """Registra, identifica y verifica hablantes por su voz (local)."""

    def __init__(
        self,
        storage_dir: Path,
        *,
        model: str = _DEFAULT_MODEL,
        threshold: float = _DEFAULT_THRESHOLD,
        default_authority: str = _DEFAULT_AUTHORITY,
    ) -> None:
        self._storage_dir = Path(storage_dir)
        self._model = model
        self._threshold = threshold
        self._default_authority = default_authority
        self._embedder = None
        self._profiles: dict[str, dict] = self._load()

    # ---------------------------------------------------------------- modelo

    def _get_embedder(self):
        """Carga el extractor de huellas (perezoso, la primera llamada baixa el modelo)."""
        if self._embedder is None:
            from speakeronnx import SpeakerEmbedder  # import perezoso

            self._embedder = SpeakerEmbedder(model=self._model)
        return self._embedder

    # ------------------------------------------------------------- persistencia

    def _storage_file(self) -> Path:
        return self._storage_dir / "speakers.json"

    def _load(self) -> dict[str, dict]:
        path = self._storage_file()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 — nunca tumbar ARIA por un archivo corrupto
            logger.warning("No pude leer speaker profiles: %s", exc)
            return {}
        profiles: dict[str, dict] = {}
        for name, entry in data.items():
            if not isinstance(entry, dict):
                continue
            embeddings = [
                np.asarray(e, dtype="float32")
                for e in entry.get("embeddings", [])
                if isinstance(e, list)
            ]
            if not embeddings:
                continue
            profiles[name] = {
                "embeddings": embeddings,
                "is_authority": bool(entry.get("is_authority", False)),
                "kind": entry.get("kind", "sample"),
            }
        return profiles

    def _save(self) -> None:
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            name: {
                "embeddings": [e.tolist() for e in entry["embeddings"]],
                "is_authority": entry["is_authority"],
                "kind": entry.get("kind", "sample"),
            }
            for name, entry in self._profiles.items()
        }
        tmp = self._storage_file().with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._storage_file())

    # ------------------------------------------------------------------- API

    @property
    def speaker_names(self) -> list[str]:
        return list(self._profiles)

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def default_authority(self) -> str:
        return self._default_authority

    def is_authority(self, name: str) -> bool:
        entry = self._profiles.get(name)
        return bool(entry and entry.get("is_authority", False))

    def has_profile(self, name: str) -> bool:
        return name in self._profiles and bool(self._profiles[name]["embeddings"])

    def sample_count(self, name: str) -> int:
        """Número de muestras de voz registradas para ``name``."""
        entry = self._profiles.get(name)
        return len(entry["embeddings"]) if entry else 0

    def _embed(self, raw: bytes) -> np.ndarray:
        """De PCM crudo a embedding normalizado (bloqueante, para hilo)."""
        audio = raw_pcm_to_f32(raw)
        return self._get_embedder().embed(audio)

    async def add_sample(self, name: str, raw: bytes, *, kind: str = "sample") -> None:
        """Añade una muestra de voz al perfil de ``name`` (puede crearlo)."""
        embedding = await asyncio.to_thread(self._embed, raw)
        entry = self._profiles.setdefault(
            name, {"embeddings": [], "is_authority": False, "kind": kind}
        )
        entry["embeddings"].append(embedding)
        self._save()
        logger.info(
            "Huella de voz añadida",
            extra={"speaker": name, "n": len(entry["embeddings"])},
        )

    async def set_authority(self, name: str, *, enabled: bool = True) -> None:
        """Marca un perfil como autorizado a dar órdenes (o lo revoca)."""
        if name not in self._profiles:
            raise VoiceError(f"No hay un perfil guardado para {name}")
        self._profiles[name]["is_authority"] = enabled
        self._save()

    async def identify(self, raw: bytes) -> tuple[str | None, float]:
        """Identifica (1:N) qué usuario registrado habla. Devuelve (nombre, score)."""
        probe = await asyncio.to_thread(self._embed, raw)
        return await self._identify_embedding(probe)

    async def _identify_embedding(self, probe: np.ndarray) -> tuple[str | None, float]:
        best_name: str | None = None
        best_score = 0.0
        for name, entry in self._profiles.items():
            if not entry["embeddings"]:
                continue
            for ref in entry["embeddings"]:
                score = float(np.dot(probe, ref))  # embeddings L2-normalizados → coseno
                if score > best_score:
                    best_score = score
                    best_name = name
        if best_name is not None and best_score >= self._threshold:
            return best_name, round(best_score, 3)
        return None, round(best_score, 3)

    async def can_control(self, raw: bytes) -> tuple[bool, str | None, float]:
        """Verifica que quien habla está autorizado a dar órdenes (1:1 + rol).

        Devuelve (autorizado, nombre_detectado, score).
        """
        name, score = await self.identify(raw)
        if name is None:
            return False, None, score
        return self.is_authority(name), name, score
