"""Tests de identificación del hablante (speaker_id)."""
from pathlib import Path

import numpy as np
import pytest

from app.voice.base import VoiceError
from app.voice.speaker_id import SpeakerIdManager, raw_pcm_to_f32

SR = 16000


def _pcm(seed: int, *, seconds: float = 2.0) -> bytes:
    """Audio PCM sintético determinista (16 kHz, s16le, mono)."""
    n = int(SR * seconds)
    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, n, endpoint=False)
    f0 = 100 + (seed % 5) * 40  # cada semilla → un tono distinto
    wave = np.sign(np.sin(2 * np.pi * f0 * t)) * 0.8
    wave += 0.1 * rng.standard_normal(n)
    wave = wave / (np.max(np.abs(wave)) + 1e-9)
    return (wave * 32767).astype("<i2").tobytes()


class _FakeEmbedder:
    """Embedding determinista donde el coseno refleja la semejanza de la señal.

    Un downsample normalizado: dos audios con la misma forma de onda dan coseno
    ≈ 1, y formas distintas dan coseno bajo — imita a un extractor de huella
    real sin descargar el modelo.
    """

    def embed(self, audio: np.ndarray) -> np.ndarray:
        step = max(1, len(audio) // 128)
        reduced = audio[::step].astype("float32")
        norm = np.linalg.norm(reduced)
        if norm == 0:
            return np.zeros(128, dtype="float32")
        return reduced / norm


def _manager(tmp_path: Path, **kwargs) -> SpeakerIdManager:
    mgr = SpeakerIdManager(tmp_path, **kwargs)
    mgr._get_embedder = lambda: _FakeEmbedder()
    return mgr


@pytest.mark.asyncio
async def test_raw_pcm_to_f32_ranges(tmp_path):
    pcm = _pcm(0)
    arr = raw_pcm_to_f32(pcm)
    assert arr.shape == (SR * 2,)
    assert arr.min() >= -1.0 and arr.max() <= 1.0
    with pytest.raises(VoiceError):
        raw_pcm_to_f32(b"\x00\x00" * 100)  # menos de un segundo


@pytest.mark.asyncio
async def test_identify_known_speaker(tmp_path):
    mgr = _manager(tmp_path, default_authority="Samuel")
    await mgr.add_sample("Samuel", _pcm(1))
    await mgr.add_sample("Samuel", _pcm(2))

    name, score = await mgr.identify(_pcm(1))

    assert name == "Samuel"
    assert score >= mgr.threshold


@pytest.mark.asyncio
async def test_identify_distinguishes_two_speakers(tmp_path):
    mgr = _manager(tmp_path)
    await mgr.add_sample("Samuel", _pcm(1))
    await mgr.add_sample("Ana", _pcm(4))

    name, _ = await mgr.identify(_pcm(4))

    assert name == "Ana"


@pytest.mark.asyncio
async def test_identify_unknown_raises_none(tmp_path):
    mgr = _manager(tmp_path)
    await mgr.add_sample("Ana", _pcm(4))

    name, _ = await mgr.identify(_pcm(50))  # semilla sin registrar

    assert name is None


@pytest.mark.asyncio
async def test_authority_and_can_control(tmp_path):
    mgr = _manager(tmp_path, default_authority="Samuel")
    await mgr.add_sample("Samuel", _pcm(1))
    await mgr.add_sample("Ana", _pcm(4))
    await mgr.set_authority("Samuel", enabled=True)

    allowed, name, _ = await mgr.can_control(_pcm(1))
    assert allowed is True
    assert name == "Samuel"

    allowed2, name2, _ = await mgr.can_control(_pcm(4))
    assert allowed2 is False
    assert name2 == "Ana"


@pytest.mark.asyncio
async def test_persistence_reload(tmp_path):
    mgr = _manager(tmp_path, default_authority="Samuel")
    await mgr.add_sample("Samuel", _pcm(1))
    await mgr.set_authority("Samuel", enabled=True)

    reloaded = SpeakerIdManager(tmp_path, default_authority="Samuel")
    reloaded._get_embedder = lambda: _FakeEmbedder()

    name, _ = await reloaded.identify(_pcm(1))
    assert name == "Samuel"
    assert reloaded.sample_count("Samuel") == 1
    assert reloaded.is_authority("Samuel") is True


@pytest.mark.asyncio
async def test_set_authority_unknown_raises(tmp_path):
    mgr = _manager(tmp_path)
    with pytest.raises(VoiceError):
        await mgr.set_authority("Nadie", enabled=True)


def test_build_speaker_manager_disabled(tmp_path, monkeypatch):
    from app.ai.factory import build_speaker_manager
    from app.core.config import Settings

    settings = Settings(speaker_id_enabled=False, data_dir=tmp_path)
    assert build_speaker_manager(settings) is None


def test_build_speaker_manager_enabled(tmp_path, monkeypatch):
    from app.ai.factory import build_speaker_manager
    from app.core.config import Settings

    settings = Settings(
        speaker_id_enabled=True,
        speaker_id_dir=tmp_path,
        speaker_id_threshold=0.6,
        speaker_default_authority="",
        aria_creator_name="Samuel",
    )
    mgr = build_speaker_manager(settings)
    assert mgr is not None
    assert mgr.threshold == 0.6
    assert mgr.default_authority == "Samuel"
