"""Tests del ciclo LLM → herramientas → LLM en el orquestador."""
from typing import ClassVar

import pytest

from app.ai.orchestrator import AssistantOrchestrator, ToolLoopError
from app.ai.providers.base import LLMProvider
from app.ai.schemas import ChatMessage, ChatResponse, ChatRole, TokenUsage, ToolCall
from app.tools.base import BaseTool, ToolPermission
from app.tools.policy import PermissionDenied, ToolPolicy
from app.tools.registry import ToolRegistry


class FakeEchoTool(BaseTool):
    name = "echo"
    description = "Devuelve el texto."
    permission = ToolPermission.SAFE
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def __init__(self) -> None:
        self.executions: list[dict] = []

    async def _run(self, text: str, **kwargs) -> str:
        self.executions.append({"text": text})
        return f"eco: {text}"


class FakeConfirmTool(BaseTool):
    name = "open_thing"
    description = "Abre algo."
    permission = ToolPermission.CONFIRM

    async def _run(self, **kwargs) -> str:
        return "abierto"


class ScriptedProvider(LLMProvider):
    """Devuelve respuestas en orden; cada entrada: (content, tool_calls)."""

    name = "scripted"
    model = "scripted-model"

    def __init__(self, script: list[tuple[str, list[ToolCall] | None]]) -> None:
        self._script = list(script)
        self.calls: list[list[ChatMessage]] = []

    async def chat(self, messages, tools=None) -> ChatResponse:
        self.calls.append(messages)
        content, tool_calls = self._script.pop(0)
        return ChatResponse(
            content=content,
            model=self.model,
            provider=self.name,
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1),
            tool_calls=tool_calls,
        )


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(FakeEchoTool())
    registry.register(FakeConfirmTool())
    return registry


@pytest.mark.asyncio
async def test_tool_loop_executes_tool_and_returns_final_answer():
    call = ToolCall(id="call-1", name="echo", arguments={"text": "hola"})
    provider = ScriptedProvider(
        [("", [call]), ("El eco dice: eco: hola", None)]
    )
    orchestrator = AssistantOrchestrator(provider, registry=_registry())

    response = await orchestrator.ask("repite hola")

    assert response.content == "El eco dice: eco: hola"
    # El proveedor recibió el resultado de la herramienta en la 2ª llamada
    second_call_messages = provider.calls[1]
    assert second_call_messages[-1].role == ChatRole.TOOL
    assert second_call_messages[-1].content == "eco: hola"
    assert second_call_messages[-1].tool_call_id == "call-1"
    # El mensaje intermedio del asistente lleva los tool_calls
    assert second_call_messages[-2].role == ChatRole.ASSISTANT
    assert second_call_messages[-2].tool_calls == [call]
    # El historial persistente queda limpio: solo user + respuesta final
    assert [m.role for m in orchestrator.history] == [
        ChatRole.USER,
        ChatRole.ASSISTANT,
    ]


@pytest.mark.asyncio
async def test_unknown_tool_reports_error_to_llm():
    call = ToolCall(id="call-1", name="no_existe", arguments={})
    provider = ScriptedProvider([("", [call]), ("No pude", None)])
    orchestrator = AssistantOrchestrator(provider, registry=_registry())

    await orchestrator.ask("haz algo")

    tool_message = provider.calls[1][-1]
    assert tool_message.role == ChatRole.TOOL
    assert "desconocida" in tool_message.content


@pytest.mark.asyncio
async def test_permission_denied_without_confirmation_raises():
    call = ToolCall(id="call-1", name="open_thing", arguments={})
    provider = ScriptedProvider([("", [call])])
    orchestrator = AssistantOrchestrator(
        provider,
        registry=_registry(),
        policy=ToolPolicy(),
    )

    with pytest.raises(PermissionDenied):
        await orchestrator.ask("abre algo")


@pytest.mark.asyncio
async def test_confirmation_declined_returns_denied_to_llm():
    call = ToolCall(id="call-1", name="open_thing", arguments={})
    provider = ScriptedProvider([("", [call]), ("Vale", None)])
    orchestrator = AssistantOrchestrator(
        provider,
        registry=_registry(),
        confirm=lambda _call: _no(),
    )

    await orchestrator.ask("abre algo")

    tool_message = provider.calls[1][-1]
    assert tool_message.role == ChatRole.TOOL
    assert "denegó" in tool_message.content


async def _no():
    return False


@pytest.mark.asyncio
async def test_confirmation_accepted_executes_tool():
    call = ToolCall(id="call-1", name="open_thing", arguments={})
    provider = ScriptedProvider([("", [call]), ("Hecho", None)])
    orchestrator = AssistantOrchestrator(
        provider,
        registry=_registry(),
        confirm=lambda _call: _yes(),
    )

    response = await orchestrator.ask("abre algo")

    assert response.content == "Hecho"
    assert provider.calls[1][-1].content == "abierto"


async def _yes():
    return True


@pytest.mark.asyncio
async def test_tool_error_is_reported_to_llm():
    class BoomTool(BaseTool):
        name = "boom"
        description = "Falla siempre."
        permission = ToolPermission.SAFE

        async def _run(self, **kwargs) -> str:
            raise RuntimeError("fallo interno")

    registry = _registry()
    registry.register(BoomTool())
    call = ToolCall(id="call-1", name="boom", arguments={})
    provider = ScriptedProvider([("", [call]), ("Lo siento", None)])
    orchestrator = AssistantOrchestrator(provider, registry=registry)

    await orchestrator.ask("prueba")

    tool_message = provider.calls[1][-1]
    assert "fallo interno" in tool_message.content


@pytest.mark.asyncio
async def test_tool_loop_limit_raises():
    call = ToolCall(id="call-1", name="echo", arguments={"text": "x"})
    script = [("", [call])] * 6  # el proveedor siempre pide herramientas
    provider = ScriptedProvider(script)
    orchestrator = AssistantOrchestrator(provider, registry=_registry())

    with pytest.raises(ToolLoopError):
        await orchestrator.ask("bucle")


@pytest.mark.asyncio
async def test_no_tools_when_registry_is_none():
    provider = ScriptedProvider([("Hola", None)])
    orchestrator = AssistantOrchestrator(provider)

    response = await orchestrator.ask("hola")

    assert response.content == "Hola"
    assert orchestrator.tool_count == 0