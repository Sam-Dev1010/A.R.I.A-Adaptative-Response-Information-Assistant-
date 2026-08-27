"""Tests del satélite de presencia ESP32 (/ws/presencia)."""
import asyncio

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.tools.presence_tools import GetPresenceTool
from app.voice.base import TTSProvider
from app.web.presencia_ws import (
    PresenciaConnection,
    frase_de_bienvenida,
    get_estado_presencia,
)


class FakeTTS(TTSProvider):
    name = "fake_tts"
    audio_ext = ".mp3"

    async def synthesize(self, text, output_path):
        from pathlib import Path

        Path(output_path).write_bytes(b"MP3FAKE")
        return Path(output_path)

    async def speak(self, text, *, output_dir=None):
        raise AssertionError("el saludo usa hablar_a_todas (synthesize)")


@pytest.fixture(autouse=True)
def estado_limpio():
    get_estado_presencia().reiniciar()
    yield
    get_estado_presencia().reiniciar()


def _client_factory(monkeypatch, **kwargs):
    def factory(ws):
        return PresenciaConnection(ws, **kwargs)

    monkeypatch.setattr("app.main.PresenciaConnection", factory)
    return TestClient(create_app())


def test_llegada_y_salida_actualizan_el_estado(monkeypatch):
    client = _client_factory(monkeypatch)

    with client.websocket_connect("/ws/presencia") as ws:
        ws.send_json({"type": "presencia", "presente": True, "rssi": -58})
        assert ws.receive_json() == {"type": "ok", "presente": True}
        ws.send_json({"type": "presencia", "presente": False})
        assert ws.receive_json() == {"type": "ok", "presente": False}

    estado = get_estado_presencia()
    assert estado.presente is False


def test_mensajes_desconocidos_no_rompen_el_canal(monkeypatch):
    client = _client_factory(monkeypatch)

    with client.websocket_connect("/ws/presencia") as ws:
        ws.send_json({"type": "ping"})
        ws.send_json({"type": "presencia", "presente": True})
        # Solo el reporte válido produce confirmación; el ping se ignora.
        assert ws.receive_json() == {"type": "ok", "presente": True}


def test_llegada_dispara_saludo_si_alguien_escucha(monkeypatch):
    llamadas: list[str] = []

    async def fake_hablar(texto, tts):
        llamadas.append(texto)
        return 1

    monkeypatch.setattr("app.web.presencia_ws.hablar_a_todas", fake_hablar)
    monkeypatch.setattr("app.web.presencia_ws.hay_alguien_escuchando", lambda: True)
    client = _client_factory(monkeypatch, tts=FakeTTS())

    with client.websocket_connect("/ws/presencia") as ws:
        ws.send_json({"type": "presencia", "presente": True})
        ws.receive_json()

    assert len(llamadas) == 1
    assert "jefe" in llamadas[0].lower()


def test_llegada_sin_oyentes_no_saluda(monkeypatch):
    llamadas: list[str] = []

    async def fake_hablar(texto, tts):
        llamadas.append(texto)
        return 1

    monkeypatch.setattr("app.web.presencia_ws.hablar_a_todas", fake_hablar)
    monkeypatch.setattr("app.web.presencia_ws.hay_alguien_escuchando", lambda: False)
    client = _client_factory(monkeypatch, tts=FakeTTS())

    with client.websocket_connect("/ws/presencia") as ws:
        ws.send_json({"type": "presencia", "presente": True})
        ws.receive_json()

    assert llamadas == []


def test_frase_de_bienvenida_cambia_con_la_hora():
    import time as _time

    class RelojFalso:
        def __init__(self, hora):
            self.hora = hora

        def __call__(self):
            return _time.struct_time((2026, 1, 1, self.hora, 0, 0, 0, 1, -1))

    mañana = frase_de_bienvenida(RelojFalso(8))
    tarde = frase_de_bienvenida(RelojFalso(15))
    noche = frase_de_bienvenida(RelojFalso(23))
    assert mañana.startswith("Buenos días")
    assert tarde.startswith(("Bienvenido", "Ya te siento", "Qué bueno"))
    assert noche.startswith(("Buenas noches", "Llegaste"))


# --------------------------- herramienta -----------------------------------


def test_get_presence_sin_satelite():
    texto = asyncio.run(GetPresenceTool().execute())
    assert "No hay satélite de presencia" in texto


def test_get_presence_en_casa():
    get_estado_presencia().marcar(True, rssi=-52)
    texto = asyncio.run(GetPresenceTool().execute())
    assert "EN CASA" in texto
    assert '"rssi": -52' in texto


def test_get_presence_fuera():
    get_estado_presencia().marcar(True)
    get_estado_presencia().marcar(False)
    texto = asyncio.run(GetPresenceTool().execute())
    assert "FUERA" in texto
