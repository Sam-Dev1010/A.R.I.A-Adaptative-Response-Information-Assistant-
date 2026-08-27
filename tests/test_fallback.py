"""Tests del failover entre proveedores (FASE 10)."""
import pytest

from app.ai.providers.base import LLMError, LLMProvider
from app.ai.providers.fallback import FallbackProvider
from app.ai.schemas import ChatMessage, ChatResponse, ChatRole, TokenUsage


class FakeProvider(LLMProvider):
    name = "fake"

    def __init__(self, model: str, fail: bool = False) -> None:
        self._model = model
        self._fail = fail
        self.calls = 0

    @property
    def model(self) -> str:
        return self._model

    async def chat(self, messages, tools=None) -> ChatResponse:
        self.calls += 1
        if self._fail:
            raise LLMError("proveedor caído")
        return ChatResponse(
            content=f"desde {self._model}",
            model=self._model,
            provider=self.name,
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1),
        )


def _messages() -> list[ChatMessage]:
    return [ChatMessage(role=ChatRole.USER, content="hola")]


@pytest.mark.asyncio
async def test_uses_primary_when_healthy():
    primary = FakeProvider("gemini")
    fallback = FakeProvider("groq")
    provider = FallbackProvider(primary, fallback)

    response = await provider.chat(_messages())

    assert response.content == "desde gemini"
    assert primary.calls == 1
    assert fallback.calls == 0
    assert provider.switched is False
    assert provider.active is primary


@pytest.mark.asyncio
async def test_switches_to_fallback_on_error():
    primary = FakeProvider("gemini", fail=True)
    fallback = FakeProvider("groq")
    provider = FallbackProvider(primary, fallback)

    response = await provider.chat(_messages())

    assert response.content == "desde groq"
    assert provider.switched is True
    assert provider.active is fallback
    assert provider.model == "groq"


@pytest.mark.asyncio
async def test_stays_on_fallback_after_switch():
    primary = FakeProvider("gemini", fail=True)
    fallback = FakeProvider("groq")
    provider = FallbackProvider(primary, fallback)

    await provider.chat(_messages())
    response = await provider.chat(_messages())

    assert response.content == "desde groq"
    assert primary.calls == 1  # el principal no se vuelve a intentar
    assert fallback.calls == 2


@pytest.mark.asyncio
async def test_raises_when_fallback_also_fails():
    primary = FakeProvider("gemini", fail=True)
    fallback = FakeProvider("groq", fail=True)
    provider = FallbackProvider(primary, fallback)

    with pytest.raises(LLMError):
        await provider.chat(_messages())


@pytest.mark.asyncio
async def test_close_closes_both():
    closed = []

    class ClosingProvider(FakeProvider):
        async def aclose(self) -> None:
            closed.append(self.model)

    provider = FallbackProvider(ClosingProvider("gemini"), ClosingProvider("groq"))

    await provider.aclose()

    assert closed == ["gemini", "groq"]