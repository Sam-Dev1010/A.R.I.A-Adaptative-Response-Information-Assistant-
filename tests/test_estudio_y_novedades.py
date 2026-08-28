"""Tests: origen de los hechos, novedades por contar y estudio profundo."""
import sqlite3

import pytest

from app.ai.orchestrator import AssistantOrchestrator
from app.ai.schemas import ChatResponse
from app.memory.manager import MemoryManager
from app.tools.study_tool import DeepStudyTool


class FakeProvider:
    name = "fake"

    def __init__(self, respuestas):
        self.respuestas = list(respuestas)

    @property
    def model(self):
        return "fake-model"

    async def chat(self, messages, tools=None):
        contenido = self.respuestas.pop(0)
        return ChatResponse(content=contenido, model=self.model, provider=self.name)


# --- memoria: origen y descubrimientos --------------------------------------

def test_remember_guarda_origen(tmp_path):
    memory = MemoryManager(tmp_path / "m.db").open()
    memory.remember("El usuario tiene una gata", origin="auto")
    memory.remember("Júpiter tiene 95 lunas conocidas", origin="web")
    memory.remember("Dato dicho directamente")

    conn = sqlite3.connect(tmp_path / "m.db")
    filas = dict(conn.execute("SELECT content, origin FROM facts").fetchall())
    assert filas["El usuario tiene una gata"] == "auto"
    assert filas["Júpiter tiene 95 lunas conocidas"] == "web"
    assert filas["Dato dicho directamente"] == ""


def test_base_antigua_se_migra_sin_perder_hechos(tmp_path):
    """Una sia.db creada antes de 'origin' se actualiza sola al abrirla."""
    db = tmp_path / "vieja.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        );
        INSERT INTO facts (content, created_at) VALUES ('hecho antiguo', 'x');
        """
    )
    conn.commit()
    conn.close()

    memory = MemoryManager(db).open()
    assert memory.facts() == ["hecho antiguo"]
    # El hecho viejo no es descubrimiento autónomo (origin vacío).
    assert memory.pending_discoveries() == []


def test_descubrimientos_pendientes_y_marca_compartido(tmp_path):
    memory = MemoryManager(tmp_path / "m.db").open()
    memory.remember("dato personal del usuario", origin="")
    memory.remember("descubrió A sobre júpiter", origin="web")
    memory.remember("dedujo B de la charla", origin="auto")

    pendientes = memory.pending_discoveries(limit=5)
    assert set(pendientes) == {"descubrió A sobre júpiter", "dedujo B de la charla"}

    memory.mark_shared(pendientes)
    assert memory.pending_discoveries() == []
    # Los hechos siguen existiendo como recuerdos normales.
    assert len(memory.facts()) == 3


# --- orquestador: menciona novedades al abrir sesión -------------------------

def test_novedades_se_inyectan_y_se_consumen(tmp_path):
    provider = FakeProvider(["listo"])
    memory = MemoryManager(tmp_path / "m.db").open()
    memory.remember("investigó la misión Europa Clipper", origin="web")

    primera = AssistantOrchestrator(provider, memory=memory)
    assert primera.novedades == ["investigó la misión Europa Clipper"]
    system_prompt = primera._system_content()
    assert "Europa Clipper" in system_prompt

    # Ya quedaron marcadas: una sesión nueva no las recibe como novedades.
    segunda = AssistantOrchestrator(provider, memory=memory)
    assert segunda.novedades == []
    assert "estuve investigando" not in segunda._system_content()


# --- deep_study --------------------------------------------------------------

async def _search_falso(query):
    return [
        {
            "title": f"Resultado para {query}",
            "snippet": f"Datos concretos sobre {query}.",
            "url": "https://ejemplo.com",
        }
    ]


@pytest.mark.asyncio
async def test_deep_study_estudia_guarda_y_responde(tmp_path):
    provider = FakeProvider(
        [
            "¿qué es la fusión nuclear?\n¿cómo funciona un tokamak?\n¿cuándo será comercial?",
            (
                "La fusión nuclear une átomos ligeros liberando energía, igual que el Sol. "
                "Los tokamaks confinan el plasma magnéticamente y se espera electricidad "
                "comercial hacia la década de 2040."
            ),
            (
                "La fusión nuclear libera energía uniendo átomos como en el Sol\n"
                "ITER es el mayor experimento de fusión del mundo"
            ),
        ]
    )
    memory = MemoryManager(tmp_path / "m.db").open()
    herramienta = DeepStudyTool(provider, memory, _search_falso)

    salida = await herramienta.execute(topic="fusión nuclear")

    assert "Estudio de 'fusión nuclear'" in salida
    assert "tokamak" in salida.lower() or "fusión" in salida.lower()
    hechos = memory.facts()
    assert len(hechos) == 2
    assert all("fusión" in h or "ITER" in h for h in hechos)
    conn = sqlite3.connect(tmp_path / "m.db")
    origenes = [row[0] for row in conn.execute("SELECT origin FROM facts").fetchall()]
    assert origenes == ["web", "web"]


@pytest.mark.asyncio
async def test_deep_study_sin_resultados_web_avisa(tmp_path):
    provider = FakeProvider(["pregunta uno suficientemente larga"])

    async def search_vacio(query):
        return []

    memoria = MemoryManager(tmp_path / "m.db").open()
    herramienta = DeepStudyTool(provider, memoria, search_vacio)

    salida = await herramienta.execute(topic="tema inexistente xyz")

    assert "No encontré información web" in salida
    assert memoria.facts() == []


@pytest.mark.asyncio
async def test_deep_study_registrado_en_el_orquestador(tmp_path):
    from unittest.mock import patch

    from app.ai.factory import build_orchestrator

    with patch("app.ai.factory.build_provider", return_value=FakeProvider([])):
        orchestrator = build_orchestrator(
            __import__("app.core.config", fromlist=["Settings"]).Settings(
                data_dir=tmp_path,
                # El cerebro neural puede enunciar un proveedor LLM (None si no
                # hay API key); al desactivarlo se garantiza la ruta provider+memoria.
                neural_enabled=False,
            ),
            memory=MemoryManager(tmp_path / "m.db").open(),
        )
    assert orchestrator.registry.get("deep_study") is not None
