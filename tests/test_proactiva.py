"""Tests de la charla espontánea (SIA habla sin que le pregunten)."""
from datetime import UTC, datetime
from pathlib import Path

from app.ai.proactive import ProactiveSpeaker
from app.ai.schemas import ChatResponse, TokenUsage
from app.core.config import Settings
from app.memory.manager import MemoryManager
from app.web import interface_ws


class FakeProvider:
    def __init__(self, texto: str = "¿Sabes qué? Estuve investigando los agujeros negros.") -> None:
        self.texto = texto
        self.ultimas_instrucciones: list[str] = []

    async def chat(self, messages, tools=None):
        self.ultimas_instrucciones.append(messages[-1].content)
        return ChatResponse(
            content=self.texto,
            model="fake",
            provider="fake",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1),
        )


class FakeTTS:
    name = "fake_tts"
    audio_ext = ".mp3"

    async def synthesize(self, text: str, output_path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mp3-falso")
        return path


class FakeWS:
    def __init__(self) -> None:
        self.mensajes: list[dict] = []

    async def send_json(self, mensaje: dict) -> None:
        self.mensajes.append(mensaje)


class FakeConexion:
    def __init__(self, ocupada: bool = False) -> None:
        self.ws = FakeWS()
        self.ocupada = ocupada


def _speaker(tmp_path, provider=None, **ajustes):
    memory = MemoryManager(tmp_path / "m.db")
    memory.open()
    opciones = {
        "aria_creator_name": "Samuel",
        "proactive_min_minutes": 1,
        "proactive_max_minutes": 2,
        "proactive_max_per_hour": 3,
        "proactive_quiet_start": 23,
        "proactive_quiet_end": 8,
    }
    opciones.update(ajustes)
    settings = Settings(**opciones)
    provider = provider or FakeProvider()
    speaker = ProactiveSpeaker(
        provider,
        memory,
        FakeTTS(),
        settings,
        reloj=lambda: datetime(2026, 8, 24, 12, 0, tzinfo=UTC),  # mediodía: en horario
        sorteo=lambda a, b: (a + b) / 2,
    )
    return speaker, provider, memory


def test_horario_silencio_cruza_medianoche(tmp_path):
    speaker, _, _ = _speaker(tmp_path)
    # 23:00–08:00 de silencio
    assert not speaker._en_horario(datetime(2026, 8, 24, 23, 30, tzinfo=UTC))
    assert not speaker._en_horario(datetime(2026, 8, 25, 3, 0, tzinfo=UTC))
    assert not speaker._en_horario(datetime(2026, 8, 25, 7, 59, tzinfo=UTC))
    assert speaker._en_horario(datetime(2026, 8, 24, 8, 0, tzinfo=UTC))
    assert speaker._en_horario(datetime(2026, 8, 24, 22, 59, tzinfo=UTC))


async def test_habla_novedad_y_la_marca_compartida(tmp_path, monkeypatch):
    speaker, _, memory = _speaker(tmp_path)
    memory.remember("ITER es el mayor experimento de fusión", origin="web")
    conexion = FakeConexion()
    monkeypatch.setattr(interface_ws, "_conexiones", {conexion})

    hablo = await speaker._quizas_hablar()

    assert hablo
    tipos = [m["type"] for m in conexion.ws.mensajes]
    assert tipos == ["speaking", "audio_chunk", "audio_end"]
    chunk = conexion.ws.mensajes[1]
    assert chunk["last"] is True and chunk["audio"] == "bXAzLWZhbHNv"
    assert memory.pending_discoveries() == []  # ya se lo contó


async def test_no_interrumpe_si_la_interfaz_esta_ocupada(tmp_path, monkeypatch):
    speaker, _, memory = _speaker(tmp_path)
    memory.remember("Dato curioso pendiente", origin="auto")
    ocupadas = {FakeConexion(ocupada=True)}
    monkeypatch.setattr(interface_ws, "_conexiones", ocupadas)

    assert not await speaker._quizas_hablar()
    assert all(not c.ws.mensajes for c in ocupadas)
    assert len(memory.pending_discoveries()) == 1  # no lo marcó como contado


async def test_presupuesto_por_hora(tmp_path, monkeypatch):
    speaker, _, _ = _speaker(tmp_path)  # presupuesto por defecto: 3 por hora
    monkeypatch.setattr(interface_ws, "_conexiones", set())

    for i in range(3):
        conexion = FakeConexion()
        interface_ws._conexiones.clear()
        interface_ws._conexiones.add(conexion)
        hablo = await speaker._quizas_hablar()
        assert hablo, f"comentario {i + 1} debería salir"

    interface_ws._conexiones.clear()
    interface_ws._conexiones.add(FakeConexion())
    assert not await speaker._quizas_hablar()  # cuarto comentario: fuera de presupuesto


async def test_rotacion_de_temas_sin_novedades(tmp_path, monkeypatch):
    provider = FakeProvider("Hola, ¿cómo va tu día?")
    speaker, _, memory = _speaker(tmp_path, provider=provider)
    monkeypatch.setattr(interface_ws, "_conexiones", {FakeConexion()})

    assert await speaker._quizas_hablar()

    instruccion = provider.ultimas_instrucciones[0]
    assert "espontáneo" in instruccion or "Tema:" in instruccion
    assert "NO repitas" in instruccion
    assert memory.pending_discoveries() == []
