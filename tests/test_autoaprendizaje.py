"""Tests de identidad (personalidad) y autoaprendizaje de A.R.I.A."""
import asyncio

import pytest

from app.ai.orchestrator import AssistantOrchestrator
from app.ai.personality import build_personality_prompt
from app.ai.schemas import ChatMessage, ChatResponse
from app.memory.manager import MemoryManager


class ScriptedProvider:
    name = "scripted"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[list[ChatMessage]] = []

    @property
    def model(self):
        return "scripted-model"

    async def chat(self, messages, tools=None):
        self.calls.append(list(messages))
        content, tool_calls = self.responses.pop(0)
        return ChatResponse(
            content=content,
            model=self.model,
            provider=self.name,
            tool_calls=tool_calls,
        )

    async def stream_chat(self, messages, tools=None):
        content, _ = self.responses.pop(0)
        yield ("delta", content)
        yield (
            "final",
            ChatResponse(content=content, model=self.model, provider=self.name),
        )


def test_personalidad_menciona_al_creador():
    prompt = build_personality_prompt("Samuel")
    assert "Samuel" in prompt
    assert "A.R.I.A" in prompt
    assert build_personality_prompt() == build_personality_prompt("Samuel")


def test_prompt_por_defecto_usa_la_personalidad():
    provider = ScriptedProvider([("Hola", None)])
    orchestrator = AssistantOrchestrator(provider)
    assert "Samuel" in orchestrator.system_prompt
    assert orchestrator.auto_learner is None


@pytest.mark.asyncio
async def test_autoaprendizaje_guarda_hechos(tmp_path):
    db = tmp_path / "mem.db"
    memory = MemoryManager(db).open()
    provider = ScriptedProvider(
        [
            ("Hola Samuel", None),
            ("El usuario estudia robótica\nNADA\n- El usuario vive en Colombia", None),
        ]
    )
    aprendidos = []
    learner_calls = 0

    async def learner(user_text, assistant_text):
        nonlocal learner_calls
        learner_calls += 1
        from app.ai.self_learner import SelfLearner

        sl = SelfLearner(provider, memory)
        hechos = await sl.learn(user_text, assistant_text)
        aprendidos.extend(hechos)

    # La extracción real la hace SelfLearner con el proveedor:
    orchestrator = AssistantOrchestrator(provider, memory=memory, auto_learner=learner)
    await orchestrator.ask("te cuento que estudio robótica en la universidad")
    await asyncio.sleep(0.05)

    assert learner_calls == 1
    facts = memory.facts()
    assert any("robótica" in f for f in facts)


@pytest.mark.asyncio
async def test_autoaprendizaje_no_se_dispara_con_mensajes_cortos():
    llamadas = []

    async def learner(user_text, assistant_text):
        llamadas.append((user_text, assistant_text))

    provider = ScriptedProvider([("Ok", None)])
    orchestrator = AssistantOrchestrator(provider, auto_learner=learner)
    await orchestrator.ask("hola")
    await asyncio.sleep(0.01)

    assert llamadas == []


@pytest.mark.asyncio
async def test_self_learner_extrae_y_persiste(tmp_path):
    from app.ai.self_learner import SelfLearner

    memory = MemoryManager(tmp_path / "m.db").open()
    provider = ScriptedProvider(
        [
            ("El usuario tiene una gata llamada Luna\nEl usuario prefiere café", None),
            ("NADA", None),
        ]
    )
    learner = SelfLearner(provider, memory)

    hechos = await learner.learn(
        "mi gata se llama luna y siempre tomo café", "qué lindo, anotado"
    )

    assert len(hechos) == 2
    assert memory.facts() == hechos

    # Idempotente: repetir no duplica (UNIQUE en la tabla facts).
    await learner.learn("mi gata se llama luna", "ya lo sé")
    assert len(memory.facts()) == 2


@pytest.mark.asyncio
async def test_self_learner_responde_nada_sin_hechos(tmp_path):
    from app.ai.self_learner import SelfLearner

    memory = MemoryManager(tmp_path / "m.db").open()
    provider = ScriptedProvider([("NADA", None)])
    learner = SelfLearner(provider, memory)

    hechos = await learner.learn("qué hora es", "son las tres")

    assert hechos == []
    assert memory.facts() == []
