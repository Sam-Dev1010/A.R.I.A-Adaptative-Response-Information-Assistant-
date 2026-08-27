"""Tests del satélite de voz ESP32 (/ws/satelite)."""
import pytest
from fastapi.testclient import TestClient

from app.ai.orchestrator import AssistantOrchestrator
from app.ai.providers.base import LLMProvider
from app.ai.schemas import ChatResponse, TokenUsage
from app.main import create_app
from app.voice.base import TTSProvider
from app.voice.stt import GoogleSTTProvider
from app.web.satelite_ws import SateliteConnection


class ScriptedProvider(LLMProvider):
    name = "scripted"
    model = "scripted-model"

    def __init__(self, script) -> None:
        self._script = list(script)

    async def chat(self, messages, tools=None) -> ChatResponse:
        content, _ = self._script.pop(0)
        return ChatResponse(
            content=content,
            model=self.model,
            provider=self.name,
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1),
        )


class FakeSTT(GoogleSTTProvider):
    def transcribe_bytes(self, raw, *, sample_rate=16000, language=None):
        return "¿qué hora es?"


class FakeTTS(TTSProvider):
    name = "fake_tts"
    audio_ext = ".mp3"

    async def synthesize(self, text, output_path):
        from pathlib import Path

        Path(output_path).write_bytes(b"MP3FAKE")
        return Path(output_path)

    async def speak(self, text, *, output_dir=None):
        raise AssertionError("el satélite no usa speak")


def _con_factory(monkeypatch, script=(("Son las tres", None),), **kwargs):
    """Crea la app con el satélite sustituido por uno de prueba."""

    def factory(ws, orchestrator):
        return SateliteConnection(
            ws, orchestrator, stt=FakeSTT(), tts=FakeTTS(), **kwargs
        )

    monkeypatch.setattr("app.main.SateliteConnection", factory)
    return TestClient(create_app(orchestrator=AssistantOrchestrator(ScriptedProvider(list(script)))))


def test_satelite_push_to_talk_completo(monkeypatch):
    monkeypatch.setattr("app.web.satelite_ws._a_pcm", lambda ruta: b"\x00\x01" * 8)
    client = _con_factory(monkeypatch)

    with client.websocket_connect("/ws/satelite") as ws:
        # El ESP32 manda PCM crudo mientras se pulsa el botón…
        ws.send_bytes(b"\x00\x00" * 16)
        ws.send_bytes(b"\x00\x00" * 16)
        # …y avisa al soltarlo.
        ws.send_json({"type": "fin"})

        assert ws.receive_json() == {"type": "estado", "valor": "pensando"}
        assert ws.receive_json() == {"type": "estado", "valor": "hablando"}
        pcm = ws.receive_bytes()
        fin = ws.receive_json()

    assert len(pcm) == 16  # los 2 frames que devuelve el conversor falso
    assert fin["type"] == "audio_end"


def test_satelite_silencio_avisa_y_sigue_vivo(monkeypatch):
    client = _con_factory(monkeypatch)

    with client.websocket_connect("/ws/satelite") as ws:
        ws.send_json({"type": "fin"})  # soltó el botón sin hablar

        aviso = ws.receive_json()
        assert aviso["type"] == "error"
        assert "silencio" in aviso["message"]

        # La conexión queda viva para el siguiente intento.
        ws.send_json({"type": "fin"})
        assert ws.receive_json()["type"] == "error"


def test_satelite_exige_token(monkeypatch):
    from fastapi import WebSocketDisconnect

    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "access_token", "secreto")
    client = TestClient(
        create_app(orchestrator=AssistantOrchestrator(ScriptedProvider([])))
    )

    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws/satelite") as ws:
        aviso = ws.receive_json()
        assert "token" in aviso["message"].lower()
        ws.receive_json()
