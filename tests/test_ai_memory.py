"""Tests de la integración memoria ↔ orquestador (FASE 4)."""
import pytest

from app.ai.orchestrator import AssistantOrchestrator
from app.ai.providers.base import LLMProvider
from app.ai.schemas import ChatMessage, ChatResponse, ChatRole, TokenUsage
from app.memory.manager import MemoryManager


class CapturingProvider(LLMProvider):
    """Proveedor que captura los mensajes que recibe."""

    name = "capture"
    model = "capture-model"

    def __init__(self) -> None:
        self.calls: list[list[ChatMessage]] = []

    async def chat(self, messages, tools=None) -> ChatResponse:
        self.calls.append(messages)
        last_user = next(
            (m.content for m in reversed(messages) if m.role is ChatRole.USER), ""
        )
        return ChatResponse(
            content=f"Respuesta a: {last_user}",
            model=self.model,
            provider=self.name,
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1),
        )


def _memory(tmp_path) -> MemoryManager:
    return MemoryManager(tmp_path / "sia.db").open()


@pytest.mark.asyncio
async def test_orchestrator_seeds_history_from_memory(tmp_path):
    memory = _memory(tmp_path)
    memory.add_exchange("¿Cómo me llamo?", "Samuel")
    provider = CapturingProvider()
    orchestrator = AssistantOrchestrator(provider, memory=memory)

    await orchestrator.ask("¿Qué hablamos?")

    sent = provider.calls[0]
    roles = [m.role for m in sent]
    assert roles == [ChatRole.SYSTEM, ChatRole.USER, ChatRole.ASSISTANT, ChatRole.USER]
    assert sent[1].content == "¿Cómo me llamo?"
    assert sent[3].content == "¿Qué hablamos?"


@pytest.mark.asyncio
async def test_orchestrator_persists_exchange(tmp_path):
    memory = _memory(tmp_path)
    orchestrator = AssistantOrchestrator(CapturingProvider(), memory=memory)

    await orchestrator.ask("Hola")

    assert [m.content for m in memory.recent_messages()] == [
        "Hola",
        "Respuesta a: Hola",
    ]


@pytest.mark.asyncio
async def test_facts_injected_in_system_prompt(tmp_path):
    memory = _memory(tmp_path)
    memory.remember("me llamo Samuel")
    provider = CapturingProvider()
    orchestrator = AssistantOrchestrator(provider, memory=memory)

    await orchestrator.ask("Hola")

    system_message = provider.calls[0][0]
    assert system_message.role == ChatRole.SYSTEM
    assert "me llamo Samuel" in system_message.content
    assert "Recuerdos" in system_message.content


@pytest.mark.asyncio
async def test_no_facts_no_recuerdos_block(tmp_path):
    memory = _memory(tmp_path)
    provider = CapturingProvider()
    orchestrator = AssistantOrchestrator(provider, memory=memory)

    await orchestrator.ask("Hola")

    assert "Recuerdos" not in provider.calls[0][0].content


@pytest.mark.asyncio
async def test_without_memory_no_persistence(tmp_path):
    db_path = tmp_path / "sia.db"
    orchestrator = AssistantOrchestrator(CapturingProvider())

    await orchestrator.ask("Hola")

    assert not db_path.exists()


@pytest.mark.asyncio
async def test_memory_continuity_across_orchestrators(tmp_path):
    """Un segundo orquestador retoma la conversación del primero."""
    memory = _memory(tmp_path)
    first = AssistantOrchestrator(CapturingProvider(), memory=memory)
    await first.ask("Me llamo Samuel")

    second = AssistantOrchestrator(CapturingProvider(), memory=memory)
    assert [m.content for m in second.history] == [
        "Me llamo Samuel",
        "Respuesta a: Me llamo Samuel",
    ]