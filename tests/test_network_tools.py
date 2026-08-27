"""Tests de las herramientas de red (FASE 9)."""
import httpx
import pytest

from app.tools.network_tools import GetWeatherTool, SearchWikipediaTool


def _weather_tool(handler) -> GetWeatherTool:
    return GetWeatherTool(transport=httpx.MockTransport(handler))


def _weather_handler():
    def handler(request: httpx.Request) -> httpx.Response:
        if "/v1/search" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "name": "Madrid",
                            "country": "España",
                            "latitude": 40.42,
                            "longitude": -3.7,
                        }
                    ]
                },
            )
        if "/v1/forecast" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "current": {
                        "temperature_2m": 22.5,
                        "apparent_temperature": 21.0,
                        "relative_humidity_2m": 45,
                        "weather_code": 0,
                        "wind_speed_10m": 12.3,
                    }
                },
            )
        return httpx.Response(404)

    return handler


@pytest.mark.asyncio
async def test_get_weather_returns_human_text():
    tool = _weather_tool(_weather_handler())

    output = await tool.execute(city="Madrid")

    assert "Madrid" in output
    assert "22.5°C" in output
    assert "despejado" in output
    assert "45%" in output


@pytest.mark.asyncio
async def test_get_weather_unknown_city():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    tool = _weather_tool(handler)

    output = await tool.execute(city="Atlántida")

    assert "no encontré" in output.lower()


@pytest.mark.asyncio
async def test_get_weather_network_error_is_friendly():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    tool = _weather_tool(handler)

    output = await tool.execute(city="Madrid")

    assert "Error" in output
    assert "500" in output


@pytest.mark.asyncio
async def test_search_wikipedia_returns_summary():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api.php"):
            return httpx.Response(
                200,
                json={"query": {"search": [{"title": "Iron Man"}]}},
            )
        if "/page/summary/Iron%20Man" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "title": "Iron Man",
                    "extract": "Iron Man es un superhéroe ficticio.",
                    "content_urls": {
                        "desktop": {"page": "https://es.wikipedia.org/wiki/Iron_Man"}
                    },
                },
            )
        return httpx.Response(404)

    tool = SearchWikipediaTool(transport=httpx.MockTransport(handler))

    output = await tool.execute(query="Iron Man")

    assert "Iron Man" in output
    assert "superhéroe" in output
    assert "wikipedia.org" in output


@pytest.mark.asyncio
async def test_search_wikipedia_no_results():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"query": {"search": []}})

    tool = SearchWikipediaTool(transport=httpx.MockTransport(handler))

    output = await tool.execute(query="xyzzy")

    assert "no encontré" in output.lower()