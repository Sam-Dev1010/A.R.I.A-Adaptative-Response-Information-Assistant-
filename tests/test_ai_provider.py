"""Tests del proveedor OpenAI-compatible con httpx.MockTransport."""
import json

import httpx
import pytest

from app.ai.providers.base import LLMError
from app.ai.providers.openai_compatible import OpenAICompatibleProvider
from app.ai.schemas import ChatMessage, ChatRole, ToolCall


def _provider(handler, **kwargs):
    transport = httpx.MockTransport(handler)
    return OpenAICompatibleProvider(
        base_url="https://api.example.com/v1",
        model="modelo-test",
        transport=transport,
        **kwargs,
    )


def _sample_response(content: str = "Hola") -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
    }


def _chat_payload():
    return [ChatMessage(role=ChatRole.USER, content="Hola")]


@pytest.mark.asyncio
async def test_chat_success():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, json=_sample_response())

    provider = _provider(handler)

    response = await provider.chat(_chat_payload())

    assert response.content == "Hola"
    assert response.model == "modelo-test"
    assert response.provider == "openai_compatible"
    assert response.usage.total_tokens == 15
    await provider.aclose()


@pytest.mark.asyncio
async def test_chat_sends_correct_payload_and_auth():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        captured["body"] = request.read()
        return httpx.Response(200, json=_sample_response())

    provider = _provider(handler, api_key="sk-clave-secreta")

    await provider.chat(_chat_payload())

    assert captured["headers"]["authorization"] == "Bearer sk-clave-secreta"
    body = captured["body"]
    assert b'"model":"modelo-test"' in body
    assert b'"role":"user"' in body
    assert b'"content":"Hola"' in body
    await provider.aclose()


@pytest.mark.asyncio
async def test_chat_without_api_key_has_no_auth_header():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        return httpx.Response(200, json=_sample_response())

    provider = _provider(handler)

    await provider.chat(_chat_payload())
    await provider.aclose()


@pytest.mark.asyncio
async def test_chat_error_status_raises_llm_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": "API key inválida"}},
        )

    provider = _provider(handler)

    with pytest.raises(LLMError, match="401"):
        await provider.chat(_chat_payload())
    await provider.aclose()


@pytest.mark.asyncio
async def test_chat_malformed_body_raises_llm_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"sin": "choices"})

    provider = _provider(handler)

    with pytest.raises(LLMError, match="Formato"):
        await provider.chat(_chat_payload())
    await provider.aclose()


@pytest.mark.asyncio
async def test_chat_without_usage_returns_none_usage():
    def handler(request: httpx.Request) -> httpx.Response:
        data = _sample_response()
        data.pop("usage")
        return httpx.Response(200, json=data)

    provider = _provider(handler)

    response = await provider.chat(_chat_payload())

    assert response.content == "Hola"
    assert response.usage is not None
    assert response.usage.total_tokens == 0
    await provider.aclose()


@pytest.mark.asyncio
async def test_chat_merges_extra_body_reasoning_effort():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read())
        return httpx.Response(200, json=_sample_response())

    provider = _provider(handler, extra_body={"reasoning_effort": "low"})

    await provider.chat(_chat_payload())

    assert captured["body"]["reasoning_effort"] == "low"
    await provider.aclose()


@pytest.mark.asyncio
async def test_chat_without_tools_payload_has_no_tools_key():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        assert "tools" not in body
        return httpx.Response(200, json=_sample_response())

    provider = _provider(handler)

    await provider.chat(_chat_payload())
    await provider.aclose()


# --- Herramientas (FASE 3) ---


def _tool_response(tool_calls: list[dict]) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                }
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }


@pytest.mark.asyncio
async def test_chat_sends_tools_payload():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.read()
        return httpx.Response(200, json=_sample_response())

    provider = _provider(handler)
    tool_spec = {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Hora actual",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    from app.tools.schemas import ToolSpec

    await provider.chat(_chat_payload(), tools=[ToolSpec.model_validate(tool_spec)])

    body = captured["body"]
    assert b'"tools":' in body
    assert b'"name":"get_time"' in body
    await provider.aclose()


@pytest.mark.asyncio
async def test_chat_parses_tool_calls():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_tool_response(
                [
                    {
                        "id": "call-abc",
                        "type": "function",
                        "function": {
                            "name": "get_time",
                            "arguments": '{"formato": "largo"}',
                        },
                    }
                ]
            ),
        )

    provider = _provider(handler)

    response = await provider.chat(_chat_payload())

    assert response.tool_calls is not None
    assert response.tool_calls[0].id == "call-abc"
    assert response.tool_calls[0].name == "get_time"
    assert response.tool_calls[0].arguments == {"formato": "largo"}
    await provider.aclose()


@pytest.mark.asyncio
async def test_chat_malformed_tool_arguments_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_tool_response(
                [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": "{rotos"},
                    }
                ]
            ),
        )

    provider = _provider(handler)

    with pytest.raises(LLMError, match="tool_calls"):
        await provider.chat(_chat_payload())
    await provider.aclose()


@pytest.mark.asyncio
async def test_chat_serializes_tool_calls_in_wire_format():
    """Al reenviar tool_calls del asistente, usa el formato wire de OpenAI."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read())
        return httpx.Response(200, json=_sample_response())

    provider = _provider(handler)
    call = ToolCall(id="call-1", name="get_time", arguments={"formato": "largo"})
    messages = [
        ChatMessage(role=ChatRole.USER, content="¿Qué hora es?"),
        ChatMessage(role=ChatRole.ASSISTANT, content="", tool_calls=[call]),
    ]

    await provider.chat(messages)

    wire_call = captured["body"]["messages"][1]["tool_calls"][0]
    assert wire_call["type"] == "function"
    assert wire_call["id"] == "call-1"
    assert wire_call["function"]["name"] == "get_time"
    assert json.loads(wire_call["function"]["arguments"]) == {"formato": "largo"}
    await provider.aclose()


# --- Streaming (SSE) ---


def _sse_body(chunks: list[dict]) -> bytes:
    lineas = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
    lineas.append("data: [DONE]\n\n")
    return "".join(lineas).encode()


@pytest.mark.asyncio
async def test_stream_chat_emits_deltas_and_final_event():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read())
        return httpx.Response(
            200,
            content=_sse_body(
                [
                    {"choices": [{"delta": {"content": "Hola"}}]},
                    {"choices": [{"delta": {"content": " mundo"}}]},
                ]
            ),
        )

    provider = _provider(handler)

    eventos = [evento async for evento in provider.stream_chat(_chat_payload())]

    assert captured["body"]["stream"] is True
    assert [tipo for tipo, _ in eventos] == ["delta", "delta", "final"]
    assert eventos[0][1] == "Hola"
    assert eventos[1][1] == " mundo"
    final = eventos[-1][1]
    assert final.content == "Hola mundo"
    assert final.model == "modelo-test"
    await provider.aclose()


@pytest.mark.asyncio
async def test_stream_chat_accumulates_fragmented_tool_calls():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse_body(
                [
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call-9",
                                            "function": {
                                                "name": "get_",
                                                "arguments": '{"forma',
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {
                                                "name": "time",
                                                "arguments": 'to": "corto"}',
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                ]
            ),
        )

    provider = _provider(handler)

    eventos = [evento async for evento in provider.stream_chat(_chat_payload())]

    tipo, final = eventos[-1]
    assert tipo == "final"
    assert final.tool_calls is not None
    assert final.tool_calls[0].id == "call-9"
    assert final.tool_calls[0].name == "get_time"
    assert final.tool_calls[0].arguments == {"formato": "corto"}
    await provider.aclose()


@pytest.mark.asyncio
async def test_stream_chat_error_status_raises_llm_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "límite"}})

    provider = _provider(handler)

    with pytest.raises(LLMError, match="429"):
        async for _evento in provider.stream_chat(_chat_payload()):
            pass
    await provider.aclose()