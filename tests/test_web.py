"""Tests de la GUI web: chat por WebSocket y confirmaciones (FASE 8)."""
import base64
from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

from app.ai.orchestrator import AssistantOrchestrator
from app.ai.providers.base import LLMProvider
from app.ai.schemas import ChatResponse, TokenUsage, ToolCall
from app.main import create_app
from app.tools.base import BaseTool, ToolPermission
from app.tools.registry import ToolRegistry
from app.voice.base import TTSProvider
from app.voice.stt import GoogleSTTProvider
from app.web.interface_ws import InterfaceConnection


class ScriptedProvider(LLMProvider):
    name = "scripted"
    model = "scripted-model"

    def __init__(self, script) -> None:
        self._script = list(script)

    async def chat(self, messages, tools=None) -> ChatResponse:
        content, tool_calls = self._script.pop(0)
        return ChatResponse(
            content=content,
            model=self.model,
            provider=self.name,
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1),
            tool_calls=tool_calls,
        )


class FakeConfirmTool(BaseTool):
    name = "open_thing"
    description = "Abre algo."
    permission = ToolPermission.CONFIRM
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
    }

    async def _run(self, url: str, **kwargs) -> str:
        return f"abierto {url}"


def _client(provider) -> TestClient:
    return TestClient(create_app(orchestrator=AssistantOrchestrator(provider)))


def test_index_page_served():
    client = TestClient(create_app(orchestrator=AssistantOrchestrator(ScriptedProvider([]))))

    response = client.get("/")

    assert response.status_code == 200
    assert "A.R.I.A" in response.text


def test_interface_page_served():
    client = TestClient(create_app(orchestrator=AssistantOrchestrator(ScriptedProvider([]))))

    response = client.get("/")

    assert response.status_code == 200
    assert "ws/interface" in response.text


def test_chat_page_kept_at_slash_chat():
    client = TestClient(create_app(orchestrator=AssistantOrchestrator(ScriptedProvider([]))))

    response = client.get("/chat")

    assert response.status_code == 200
    assert "ws/chat" in response.text


class FakeAudioSTT(GoogleSTTProvider):
    """STT que devuelve un texto fijo sin tocar el micrófono."""

    def transcribe_bytes(self, raw, *, sample_rate=16000, language=None):
        return "¿qué hora es?"


class FakeAudioTTS(TTSProvider):
    name = "fake_audio_tts"

    async def synthesize(self, text, output_path):
        from pathlib import Path

        Path(output_path).write_bytes(b"MP3DATA")
        return Path(output_path)

    async def speak(self, text, *, output_dir=None):
        raise AssertionError("la interfaz no usa speak")


class RecordingTTS(TTSProvider):
    name = "recording_tts"

    def __init__(self) -> None:
        self.pieces: list[str] = []

    async def synthesize(self, text, output_path):
        from pathlib import Path

        self.pieces.append(text)
        Path(output_path).write_bytes(f"AUDIO:{text}".encode())
        return Path(output_path)

    async def speak(self, text, *, output_dir=None):
        raise AssertionError("la interfaz no usa speak")


def _recibir_hasta_fin(ws) -> list[dict]:
    """Recibe mensajes del WebSocket hasta el cierre del turno de habla."""
    mensajes = []
    while True:
        mensaje = ws.receive_json()
        mensajes.append(mensaje)
        if mensaje["type"] == "audio_end":
            return mensajes


def test_ws_interface_voice_roundtrip(monkeypatch):
    provider = ScriptedProvider([("Son las tres", None)])

    def factory(ws, orchestrator):
        return InterfaceConnection(
            ws, orchestrator, stt=FakeAudioSTT(), tts=FakeAudioTTS()
        )

    monkeypatch.setattr("app.main.InterfaceConnection", factory)
    monkeypatch.setattr("app.web.interface_ws._ffmpeg_available", lambda: True)
    monkeypatch.setattr(InterfaceConnection, "_decode_to_pcm", staticmethod(lambda data: b"\x00" * 320))
    client = TestClient(create_app(orchestrator=AssistantOrchestrator(provider)))

    with client.websocket_connect("/ws/interface") as ws:
        ws.send_json({"type": "audio", "data": base64.b64encode(b"webm").decode()})
        mensajes = _recibir_hasta_fin(ws)

    # La voz va primero: los chunks salen sin esperar al texto completo.
    assert mensajes[0]["type"] == "thinking"
    assert mensajes[1]["type"] == "speaking"
    chunks = [m for m in mensajes if m["type"] == "audio_chunk"]
    textos = [m for m in mensajes if m["type"] == "response"]
    assert len(chunks) == 1
    assert chunks[0]["mime"] == "audio/mpeg"
    assert base64.b64decode(chunks[0]["audio"]) == b"MP3DATA"
    assert len(textos) == 1
    assert textos[0]["text"] == "Son las tres"
    assert mensajes[-1]["type"] == "audio_end"


def test_ws_interface_splits_response_into_ordered_chunks(monkeypatch):
    provider = ScriptedProvider(
        [("Son las tres en punto de la tarde. Todo sigue funcionando bien.", None)]
    )
    tts = RecordingTTS()

    def factory(ws, orchestrator):
        return InterfaceConnection(ws, orchestrator, stt=FakeAudioSTT(), tts=tts)

    monkeypatch.setattr("app.main.InterfaceConnection", factory)
    monkeypatch.setattr("app.web.interface_ws._ffmpeg_available", lambda: True)
    monkeypatch.setattr(InterfaceConnection, "_decode_to_pcm", staticmethod(lambda data: b"\x00" * 320))
    client = TestClient(create_app(orchestrator=AssistantOrchestrator(provider)))

    with client.websocket_connect("/ws/interface") as ws:
        ws.send_json({"type": "audio", "data": base64.b64encode(b"webm").decode()})
        mensajes = _recibir_hasta_fin(ws)

    assert tts.pieces == [
        "Son las tres en punto de la tarde.",
        "Todo sigue funcionando bien.",
    ]
    chunks = [m for m in mensajes if m["type"] == "audio_chunk"]
    assert len(chunks) == 2
    assert all(m["last"] is False for m in chunks)
    assert base64.b64decode(chunks[0]["audio"]) == b"AUDIO:Son las tres en punto de la tarde."
    assert base64.b64decode(chunks[1]["audio"]) == b"AUDIO:Todo sigue funcionando bien."
    texto = next(m for m in mensajes if m["type"] == "response")
    assert texto["text"] == "Son las tres en punto de la tarde. Todo sigue funcionando bien."
    assert mensajes[-1]["type"] == "audio_end"


def test_ws_interface_text_message(monkeypatch):
    """El texto escrito llega por WebSocket y usa el mismo pipeline que la voz."""
    provider = ScriptedProvider([("Texto recibido", None)])
    tts = RecordingTTS()

    def factory(ws, orchestrator):
        return InterfaceConnection(ws, orchestrator, stt=FakeAudioSTT(), tts=tts)

    monkeypatch.setattr("app.main.InterfaceConnection", factory)
    client = TestClient(create_app(orchestrator=AssistantOrchestrator(provider)))

    with client.websocket_connect("/ws/interface") as ws:
        ws.send_json({"type": "text", "data": "\u00bfc\u00f3mo est\u00e1s?"})
        mensajes = _recibir_hasta_fin(ws)

    tipos = [m["type"] for m in mensajes]
    assert tipos[0] == "thinking"
    assert "speaking" in tipos
    assert tts.pieces == ["Texto recibido"]
    respuesta = next(m for m in mensajes if m["type"] == "response")
    assert respuesta["text"] == "Texto recibido"
    assert mensajes[-1]["type"] == "audio_end"


def test_ws_chat_basic_roundtrip():
    provider = ScriptedProvider([("Hola desde SIA", None)])
    client = _client(provider)

    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"type": "message", "text": "hola"})
        data = ws.receive_json()

    assert data["type"] == "response"
    assert data["content"] == "Hola desde SIA"


def test_ws_rechaza_token_invalido(monkeypatch):
    from fastapi import WebSocketDisconnect

    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "access_token", "secreto")
    client = _client(ScriptedProvider([]))

    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws/interface") as ws:
        aviso = ws.receive_json()
        assert aviso["type"] == "error"
        assert "token" in aviso["message"].lower()
        # Tras el aviso el servidor cierra la conexión.
        ws.receive_json()


def test_ws_acepta_token_valido(monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "access_token", "secreto")
    provider = ScriptedProvider([("Acceso concedido", None)])
    client = _client(provider)

    with client.websocket_connect("/ws/chat?token=secreto") as ws:
        ws.send_json({"type": "message", "text": "hola"})
        data = ws.receive_json()

    assert data["type"] == "response"
    assert data["content"] == "Acceso concedido"


def test_acceso_local_omite_token(monkeypatch):
    """La propia PC (localhost) nunca necesita token; el celular sí."""
    from types import SimpleNamespace

    from app.core.config import get_settings
    from app.main import _acceso_permitido

    monkeypatch.setattr(get_settings(), "access_token", "secreto")

    def conexion(host, query=""):
        return SimpleNamespace(
            client=SimpleNamespace(host=host), query_params={"token": query}
        )

    # PC: localhost entra sin token y con cualquier valor.
    assert _acceso_permitido(conexion("127.0.0.1"))
    assert _acceso_permitido(conexion("::1"))
    # Celular u otro equipo: sin token correcto no entra.
    assert _acceso_permitido(conexion("192.168.1.40", "secreto"))
    assert not _acceso_permitido(conexion("192.168.1.40", "malo"))
    assert not _acceso_permitido(conexion("100.64.0.5"))  # IP Tailscale


def test_ws_confirm_approved_runs_tool():
    call = ToolCall(id="call-1", name="open_thing", arguments={"url": "https://x.com"})
    provider = ScriptedProvider([("", [call]), ("Listo", None)])
    registry = ToolRegistry()
    registry.register(FakeConfirmTool())
    client = TestClient(
        create_app(orchestrator=AssistantOrchestrator(provider, registry=registry))
    )

    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"type": "message", "text": "abre x"})
        confirm = ws.receive_json()

        assert confirm["type"] == "confirm"
        assert confirm["tool"] == "open_thing"
        assert confirm["arguments"] == {"url": "https://x.com"}

        ws.send_json(
            {
                "type": "confirm_response",
                "request_id": confirm["request_id"],
                "approved": True,
            }
        )
        data = ws.receive_json()

    assert data["type"] == "response"
    assert data["content"] == "Listo"


def test_ws_confirm_denied_reports_to_llm():
    call = ToolCall(id="call-1", name="open_thing", arguments={"url": "https://x.com"})
    provider = ScriptedProvider([("", [call]), ("Entendido", None)])
    registry = ToolRegistry()
    registry.register(FakeConfirmTool())
    client = TestClient(
        create_app(orchestrator=AssistantOrchestrator(provider, registry=registry))
    )

    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"type": "message", "text": "abre x"})
        confirm = ws.receive_json()
        ws.send_json(
            {
                "type": "confirm_response",
                "request_id": confirm["request_id"],
                "approved": False,
            }
        )
        data = ws.receive_json()

    assert data["type"] == "response"
    assert data["content"] == "Entendido"