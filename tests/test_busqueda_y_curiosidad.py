"""Tests de búsqueda web (DuckDuckGo) y curiosidad autónoma."""

import httpx
import pytest

from app.ai.auto_curiosity import CuriosityEngine
from app.ai.schemas import ChatResponse
from app.memory.manager import MemoryManager
from app.tools.network_tools import (
    WebSearchTool,
    _url_real,
    parsear_resultados_ddg,
)

_HTML_DDG = """
<div class="result results_links results_links_deep web-result">
  <h2 class="result__title">
    <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fejemplo.com%2Farticulo&amp;rut=abc">Título uno</a>
  </h2>
  <a class="result__snippet" href="#">Resumen del <b>primer</b> resultado.</a>
</div>
<div class="result">
  <h2><a rel="nofollow" class="result__a" href="https://directo.com/pagina">Título dos</a></h2>
  <a class="result__snippet">Segundo resumen.</a>
</div>
"""

_HTML_LITE = """
<table>
  <tr><td>1.</td><td><a rel="nofollow" href="https://lite.ejemplo.com/a">Resultado lite</a></td></tr>
  <tr><td></td><td class="result-snippet">Fragmento lite con info.</td></tr>
</table>
"""


def test_url_real_decodifica_redirect_ddg():
    assert (
        _url_real("//duckduckgo.com/l/?uddg=https%3A%2F%2Fejemplo.com%2Fa&rut=1")
        == "https://ejemplo.com/a"
    )
    assert _url_real("https://normal.com") == "https://normal.com"


def test_parsear_resultados_ddg_vista_html():
    resultados = parsear_resultados_ddg(_HTML_DDG)
    assert len(resultados) == 2
    assert resultados[0]["title"] == "Título uno"
    assert resultados[0]["url"] == "https://ejemplo.com/articulo"
    assert "primer" in resultados[0]["snippet"]
    assert resultados[1]["snippet"] == "Segundo resumen."


def test_parsear_resultados_ddg_vista_lite():
    resultados = parsear_resultados_ddg(_HTML_LITE)
    assert len(resultados) == 1
    assert resultados[0]["title"] == "Resultado lite"
    assert resultados[0]["snippet"].startswith("Fragmento lite")


def _ddg_tool(handler) -> WebSearchTool:
    return WebSearchTool(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_web_search_tool_formatea_para_el_llm():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_HTML_DDG)

    salida = await _ddg_tool(handler).execute(query="james webb telescopio")

    assert "Título uno" in salida
    assert "Resumen del primer resultado." in salida
    assert "https://ejemplo.com/articulo" in salida


@pytest.mark.asyncio
async def test_web_search_tool_sin_resultados():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html><body>nada</body></html>")

    salida = await _ddg_tool(handler).execute(query="xyzq")

    assert "Sin resultados" in salida


@pytest.mark.asyncio
async def test_web_search_tool_bloqueo_da_mensaje_amable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="blocked")

    salida = await _ddg_tool(handler).execute(query="cualquier cosa")

    assert "Error buscando" in salida


class FakeProvider:
    """Proveedor mínimo que devuelve respuestas en orden."""

    name = "fake"

    def __init__(self, respuestas):
        self.respuestas = list(respuestas)

    @property
    def model(self):
        return "fake-model"

    async def chat(self, messages, tools=None):
        contenido = self.respuestas.pop(0)
        return ChatResponse(
            content=contenido, model=self.model, provider=self.name
        )


class FakeMemory(MemoryManager):
    pass


def _memory(tmp_path):
    return MemoryManager(tmp_path / "curiosity.db").open()


@pytest.mark.asyncio
async def test_curiosidad_investiga_y_guarda(tmp_path):
    provider = FakeProvider(
        [
            "¿qué es la nave espacial Juno y qué descubrió?",
            "La sonda Juno estudia la atmósfera de Júpiter desde 2016\nNADA",
        ]
    )

    async def search(query):
        return [
            {"title": "Juno", "snippet": "Sonda de la NASA orbitando Júpiter.", "url": "x"},
        ]

    memory = _memory(tmp_path)
    motor = CuriosityEngine(
        provider, memory, search, cooldown_seconds=0.0, max_por_hora=5
    )

    aprendido = await motor.research(
        "acabo de ver un documental de júpiter increíble",
        "qué bien, júpiter es fascinante",
    )

    assert len(aprendido) == 1
    assert "Juno" in aprendido[0]
    assert memory.facts() == aprendido


@pytest.mark.asyncio
async def test_curiosidad_nada_interesante_no_busca(tmp_path):
    provider = FakeProvider(["NADA"])
    busquedas = []

    async def search(query):
        busquedas.append(query)
        return []

    motor = CuriosityEngine(provider, _memory(tmp_path), search, cooldown_seconds=0)

    aprendido = await motor.research("ponme la hora", "son las tres")

    assert aprendido == []
    assert busquedas == []


@pytest.mark.asyncio
async def test_curiosidad_respeta_cooldown(tmp_path):
    provider = FakeProvider(
        [
            "¿qué descubrió la sonda juno en júpiter?",
            "Idea duradera suficientemente larga",
            "¿otra pregunta de investigación interesante?",
        ]
    )
    llamadas = []

    async def search(query):
        llamadas.append(query)
        return [{"title": "t", "snippet": "s", "url": "u"}]

    motor = CuriosityEngine(provider, _memory(tmp_path), search, cooldown_seconds=3600)

    await motor.research("un tema muy interesante para investigar hoy", "sí, cuéntame más")
    segunda = await motor.research("otro tema distinto e interesante también", "va de nuevo")

    assert len(llamadas) == 1
    assert segunda == []


@pytest.mark.asyncio
async def test_curiosidad_tope_por_hora(tmp_path):
    provider = FakeProvider(
        ["pregunta una bastante larga", "Idea duradera suficientemente larga"]
    )
    busquedas_hechas = []

    async def search(query):
        busquedas_hechas.append(query)
        return [{"title": "t", "snippet": "s", "url": "u"}]

    motor = CuriosityEngine(
        provider, _memory(tmp_path), search, cooldown_seconds=0, max_por_hora=1
    )

    await motor.research("un tema muy interesante para investigar hoy", "cuéntame")
    # El tope horario corta aunque el proveedor tenga respuestas.
    provider.respuestas.insert(0, "otra pregunta larga e interesante")
    resultado = await motor.research("otro tema distinto e interesante también", "va")

    assert len(busquedas_hechas) == 1
    assert resultado == []
