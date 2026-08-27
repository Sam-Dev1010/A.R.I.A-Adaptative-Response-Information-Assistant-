"""Tests del AssistantOrchestrator con un proveedor falso."""
import pytest

from app.ai.orchestrator import AssistantOrchestrator
from app.ai.providers.base import LLMProvider
from app.ai.schemas import ChatMessage, ChatResponse, ChatRole, TokenUsage


class FakeProvider(LLMProvider):
    """Proveedor determinista para tests: responde con el contenido del último mensaje."""

    name = "fake"
    model = "fake-model"

    def __init__(self, reply: str = "Hola, soy SIA.") -> None:
        self._reply = reply
        self.calls: list[list[ChatMessage]] = []

    async def chat(
        self, messages: list[ChatMessage], tools=None
    ) -> ChatResponse:
        self.calls.append(messages)
        return ChatResponse(
            content=self._reply,
            model=self.model,
            provider=self.name,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        )


@pytest.mark.asyncio
async def test_ask_returns_provider_response():
    provider = FakeProvider(reply="¡A tus órdenes!")
    orchestrator = AssistantOrchestrator(provider)

    response = await orchestrator.ask("Hola")

    assert response.content == "¡A tus órdenes!"
    assert response.provider == "fake"
    assert response.usage.total_tokens == 15


@pytest.mark.asyncio
async def test_ask_sends_system_prompt_and_user_message():
    provider = FakeProvider()
    orchestrator = AssistantOrchestrator(provider, system_prompt="Eres SIA.")

    await orchestrator.ask("¿Qué hora es?")

    messages = provider.calls[0]
    assert len(messages) == 2
    assert messages[0].role == ChatRole.SYSTEM
    assert messages[0].content.startswith("Eres SIA.")
    assert "Contexto temporal" in messages[0].content
    assert messages[1] == ChatMessage(role=ChatRole.USER, content="¿Qué hora es?")


@pytest.mark.asyncio
async def test_ask_records_exchange_in_history():
    provider = FakeProvider()
    orchestrator = AssistantOrchestrator(provider)

    await orchestrator.ask("Primero")
    await orchestrator.ask("Segundo")

    assert orchestrator.history == [
        ChatMessage(role=ChatRole.USER, content="Primero"),
        ChatMessage(role=ChatRole.ASSISTANT, content="Hola, soy SIA."),
        ChatMessage(role=ChatRole.USER, content="Segundo"),
        ChatMessage(role=ChatRole.ASSISTANT, content="Hola, soy SIA."),
    ]


@pytest.mark.asyncio
async def test_ask_includes_history_in_next_call():
    provider = FakeProvider()
    orchestrator = AssistantOrchestrator(provider)

    await orchestrator.ask("Recuérdame esto")
    await orchestrator.ask("¿Qué te dije?")

    messages = provider.calls[1]
    assert [m.role for m in messages] == [
        ChatRole.SYSTEM,
        ChatRole.USER,
        ChatRole.ASSISTANT,
        ChatRole.USER,
    ]


@pytest.mark.asyncio
async def test_history_is_bounded_and_keeps_pairs():
    provider = FakeProvider()
    orchestrator = AssistantOrchestrator(provider, max_history_messages=2)

    for i in range(5):
        await orchestrator.ask(f"Mensaje {i}")

    assert len(orchestrator.history) == 2
    assert orchestrator.history[0].content == "Mensaje 4"
    assert orchestrator.history[1].role == ChatRole.ASSISTANT


@pytest.mark.asyncio
async def test_reset_clears_history():
    provider = FakeProvider()
    orchestrator = AssistantOrchestrator(provider)

    await orchestrator.ask("Hola")
    orchestrator.reset()

    assert orchestrator.history == []
    await orchestrator.ask("Nuevo")
    assert len(provider.calls[1]) == 2  # solo system + user


@pytest.mark.asyncio
async def test_empty_message_is_rejected():
    orchestrator = AssistantOrchestrator(FakeProvider())

    with pytest.raises(ValueError):
        await orchestrator.ask("   ")

@pytest.mark.asyncio
async def test_system_message_includes_current_datetime():
    provider = FakeProvider()
    orchestrator = AssistantOrchestrator(provider)

    await orchestrator.ask("¿Qué hora es?")

    system = provider.calls[0][0]
    assert system.role == ChatRole.SYSTEM
    assert "Contexto temporal" in system.content
    assert "sin usar herramientas" in system.content


# --- Streaming ---


class StreamingProvider(LLMProvider):
    """Proveedor que emite la respuesta en tres fragmentos."""

    name = "streaming"
    model = "stream-model"

    def __init__(self, fragmentos: list[str]) -> None:
        self._fragmentos = fragmentos

    async def chat(self, messages, tools=None) -> ChatResponse:
        raise AssertionError("ask_stream no debe usar chat()")

    async def stream_chat(self, messages, tools=None):
        for fragmento in self._fragmentos:
            yield ("delta", fragmento)
        yield (
            "final",
            ChatResponse(
                content="".join(self._fragmentos),
                model=self.model,
                provider=self.name,
            ),
        )


@pytest.mark.asyncio
async def test_ask_stream_yields_fragments_and_records_history():
    orchestrator = AssistantOrchestrator(
        StreamingProvider(["Son las ", "tres en ", "punto."])
    )

    fragmentos = [f async for f in orchestrator.ask_stream("¿Qué hora es?")]

    assert fragmentos == ["Son las ", "tres en ", "punto."]
    assert orchestrator.history[-1] == ChatMessage(
        role=ChatRole.ASSISTANT, content="Son las tres en punto."
    )
    assert orchestrator.history[-2].content == "¿Qué hora es?"


@pytest.mark.asyncio
async def test_ask_stream_rejects_empty_message():
    orchestrator = AssistantOrchestrator(StreamingProvider(["x"]))

    with pytest.raises(ValueError):
        async for _ in orchestrator.ask_stream("   "):
            pass
