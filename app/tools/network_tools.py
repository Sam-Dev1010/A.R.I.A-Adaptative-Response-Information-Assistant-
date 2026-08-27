"""Herramientas de red de SIA (FASE 9): clima, Wikipedia y búsqueda web.

Todo sin API key: Open-Meteo (clima), Wikipedia (resúmenes) y DuckDuckGo
(búsqueda general en internet).
"""
import html as _html
import logging
import re
from typing import ClassVar
from urllib.parse import parse_qs, quote, unquote, urlparse

import httpx

from app.tools.base import BaseTool, ToolPermission

logger = logging.getLogger("sia.tools")

_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
_WIKIPEDIA_API = "https://es.wikipedia.org/w/api.php"
_WIKIPEDIA_SUMMARY = "https://es.wikipedia.org/api/rest_v1/page/summary/{title}"
_DDG_HTML = "https://html.duckduckgo.com/html/"
_DDG_LITE = "https://lite.duckduckgo.com/lite/"
_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"

_WMO_CODES = {
    0: "despejado",
    1: "mayormente despejado",
    2: "parcialmente nublado",
    3: "nublado",
    45: "niebla",
    48: "niebla helada",
    51: "llovizna ligera",
    53: "llovizna",
    55: "llovizna fuerte",
    61: "lluvia ligera",
    63: "lluvia",
    65: "lluvia fuerte",
    71: "nieve ligera",
    73: "nieve",
    75: "nieve fuerte",
    80: "chubascos",
    81: "chubascos fuertes",
    95: "tormenta",
    96: "tormenta con granizo",
}


class GetWeatherTool(BaseTool):
    """Clima actual de una ciudad vía Open-Meteo (sin API key)."""

    name = "get_weather"
    description = "Devuelve el clima actual de una ciudad (temperatura, humedad, viento)."
    permission = ToolPermission.SAFE
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "Nombre de la ciudad (ej: Madrid)"}
        },
        "required": ["city"],
    }

    def __init__(
        self,
        *,
        timeout_seconds: float = 15.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(timeout=timeout_seconds, transport=transport)

    async def _run(self, city: str, **kwargs) -> str:
        try:
            location = await self._geocode(city)
            if location is None:
                return f"No encontré la ciudad: {city}"
            return await self._current_weather(location)
        except httpx.HTTPError as exc:
            logger.warning("Error en get_weather", extra={"error": str(exc)})
            return f"Error consultando el clima: {exc}"

    async def _geocode(self, city: str) -> dict | None:
        response = await self._client.get(
            _GEOCODING_URL,
            params={"name": city, "count": 1, "language": "es"},
        )
        response.raise_for_status()
        results = response.json().get("results") or []
        return results[0] if results else None

    async def _current_weather(self, location: dict) -> str:
        response = await self._client.get(
            _WEATHER_URL,
            params={
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "current": (
                    "temperature_2m,relative_humidity_2m,apparent_temperature,"
                    "weather_code,wind_speed_10m"
                ),
            },
        )
        response.raise_for_status()
        current = response.json()["current"]
        code = _WMO_CODES.get(current["weather_code"], "desconocido")
        name = location.get("name") or location.get("admin1") or location.get("country_code", "")
        return (
            f"Clima en {name}, {location.get('country', '')}: {code}, "
            f"{current['temperature_2m']}°C (sensación "
            f"{current['apparent_temperature']}°C), humedad "
            f"{current['relative_humidity_2m']}%, viento "
            f"{current['wind_speed_10m']} km/h."
        )


class SearchWikipediaTool(BaseTool):
    """Busca en Wikipedia (español) y resume el primer resultado."""

    name = "search_wikipedia"
    description = "Busca en Wikipedia en español y resume el primer resultado."
    permission = ToolPermission.SAFE
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Término a buscar (ej: 'Iron Man').",
            }
        },
        "required": ["query"],
    }

    def __init__(
        self,
        *,
        timeout_seconds: float = 15.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(timeout=timeout_seconds, transport=transport)

    async def _run(self, query: str, **kwargs) -> str:
        try:
            title = await self._search(query)
            if title is None:
                return f"No encontré resultados en Wikipedia para: {query}"
            return await self._summary(title)
        except httpx.HTTPError as exc:
            logger.warning("Error en search_wikipedia", extra={"error": str(exc)})
            return f"Error consultando Wikipedia: {exc}"

    async def _search(self, query: str) -> str | None:
        response = await self._client.get(
            _WIKIPEDIA_API,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 1,
            },
        )
        response.raise_for_status()
        hits = response.json().get("query", {}).get("search", [])
        return hits[0]["title"] if hits else None

    async def _summary(self, title: str) -> str:
        response = await self._client.get(_WIKIPEDIA_SUMMARY.format(title=quote(title)))
        response.raise_for_status()
        data = response.json()
        extract = data.get("extract")
        if not extract:
            return f"No hay resumen disponible para: {title}"
        url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
        return f"{title}: {extract[:1500]}\nFuente: {url}"


# --- búsqueda web general (DuckDuckGo, sin API key) -----------------------

_TAG_RE = re.compile(r"<[^>]+>")
_RESULT_LINK_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL | re.IGNORECASE
)
_SNIPPET_RE = re.compile(
    r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL | re.IGNORECASE
)
_LITE_ROW_RE = re.compile(
    r'<a[^>]+rel="nofollow"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL | re.IGNORECASE
)


def _limpiar_html(fragmento: str) -> str:
    """Quita etiquetas y entidades; texto plano legible."""
    return re.sub(r"\s{2,}", " ", _TAG_RE.sub("", _html.unescape(fragmento))).strip()


def _url_real(enlace: str) -> str:
    """Resuelve los enlaces-redirección de DuckDuckGo (uddg=<url codificada>)."""
    if "duckduckgo.com/l/" in enlace:
        query = urlparse(enlace).query
        destino = parse_qs(query).get("uddg", [""])[0]
        return unquote(destino) or enlace
    return enlace


def parsear_resultados_ddg(html: str, limite: int = 5) -> list[dict]:
    """Extrae resultados del HTML de DuckDuckGo (vista html o lite)."""
    resultados: list[dict] = []
    enlaces = _RESULT_LINK_RE.findall(html)
    fragmentos = [_limpiar_html(s) for s in _SNIPPET_RE.findall(html)]
    for i, (enlace, titulo) in enumerate(enlaces):
        if i < len(fragmentos):
            resumen = fragmentos[i]
        else:
            resumen = ""
        resultados.append(
            {
                "title": _limpiar_html(titulo),
                "snippet": resumen,
                "url": _url_real(_html.unescape(enlace)),
            }
        )
        if len(resultados) >= limite:
            return resultados

    # Vista lite: filas de tabla con el enlace y el snippet por separado.
    if not resultados:
        for enlace, titulo in _LITE_ROW_RE.findall(html):
            if any(r["title"] == _limpiar_html(titulo) for r in resultados):
                continue
            resultados.append(
                {
                    "title": _limpiar_html(titulo),
                    "snippet": "",
                    "url": _url_real(_html.unescape(enlace)),
                }
            )
            if len(resultados) >= limite:
                break
        for i, snippet in enumerate(re.finditer(r'class="result-snippet">(.*?)</td>', html, re.DOTALL)):
            if i < len(resultados):
                resultados[i]["snippet"] = _limpiar_html(snippet.group(1))
    return resultados


class WebSearchTool(BaseTool):
    """Búsqueda general en internet vía DuckDuckGo (sin API key).

    Devuelve títulos, resúmenes y URLs listos para que el LLM los lea.
    """

    name = "web_search"
    description = (
        "Busca en internet (DuckDuckGo) datos actuales o desconocidos."
    )
    permission = ToolPermission.SAFE
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Consulta de búsqueda."}
        },
        "required": ["query"],
    }

    def __init__(
        self,
        *,
        timeout_seconds: float = 15.0,
        max_results: int = 5,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(timeout=timeout_seconds, transport=transport)
        self._max_results = max_results

    async def _run(self, query: str, **kwargs) -> str:
        try:
            resultados = await self._buscar(query)
        except httpx.HTTPError as exc:
            logger.warning("Error en web_search", extra={"error": str(exc)})
            return f"Error buscando en internet: {exc}"
        if not resultados:
            return f"Sin resultados web para: {query}"
        lineas = []
        for i, r in enumerate(resultados, 1):
            linea = f"{i}. {r['title']}"
            if r["snippet"]:
                linea += f" — {r['snippet']}"
            lineas.append(linea + f"\n   {r['url']}")
        return "\n\n".join(lineas)

    async def search(self, query: str) -> list[dict]:
        """Resultados estructurados [{title, snippet, url}] (para la curiosidad)."""
        try:
            return await self._buscar(query)
        except httpx.HTTPError as exc:
            logger.warning("web_search falló", extra={"error": str(exc)})
            return []

    async def _buscar(self, query: str) -> list[dict]:
        # La vista html acepta POST (más fiable); si falla, la vista lite por GET.
        try:
            respuesta = await self._client.post(
                _DDG_HTML,
                data={"q": query},
                headers={"User-Agent": _UA},
                follow_redirects=True,
            )
            respuesta.raise_for_status()
            resultados = parsear_resultados_ddg(respuesta.text, self._max_results)
            if resultados:
                return resultados
        except httpx.HTTPError as exc:
            logger.debug("DDG html falló (%s); probando lite", exc)

        respuesta = await self._client.get(
            _DDG_LITE,
            params={"q": query},
            headers={"User-Agent": _UA},
            follow_redirects=True,
        )
        respuesta.raise_for_status()
        return parsear_resultados_ddg(respuesta.text, self._max_results)