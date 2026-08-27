"""Tests de la memoria persistente (FASE 4)."""

from app.ai.schemas import ChatRole
from app.memory.manager import MemoryManager


def _memory(tmp_path, **kwargs) -> MemoryManager:
    return MemoryManager(tmp_path / "sia.db", **kwargs).open()


# --- Mensajes y sesiones ---


def test_add_exchange_and_recent_messages(tmp_path):
    memory = _memory(tmp_path)
    memory.add_exchange("Hola", "¡Hola!")

    messages = memory.recent_messages()
    assert len(messages) == 2
    assert messages[0].role == ChatRole.USER and messages[0].content == "Hola"
    assert messages[1].role == ChatRole.ASSISTANT and messages[1].content == "¡Hola!"


def test_recent_messages_empty_without_data(tmp_path):
    assert _memory(tmp_path).recent_messages() == []


def test_recent_messages_returns_even_pairs(tmp_path):
    memory = _memory(tmp_path)
    for i in range(3):
        memory.add_exchange(f"Pregunta {i}", f"Respuesta {i}")

    messages = memory.recent_messages(limit=3)  # 3 impar → 2
    assert len(messages) == 2
    assert messages[0].content == "Pregunta 2"


def test_recent_messages_takes_latest_session(tmp_path):
    memory = _memory(tmp_path)
    memory.start_session("sesión vieja")
    memory.add_exchange("viejo", "antiguo")
    memory.start_session("sesión nueva")

    messages = memory.recent_messages()
    assert [m.content for m in messages] == ["viejo", "antiguo"]


def test_memory_persists_across_reopens(tmp_path):
    db_path = tmp_path / "sia.db"
    memory = MemoryManager(db_path).open()
    memory.add_exchange("Hola", "¡Hola!")
    memory.remember("me llamo Samuel")
    memory.close()

    memory = MemoryManager(db_path).open()
    assert [m.content for m in memory.recent_messages()] == ["Hola", "¡Hola!"]
    assert memory.facts() == ["me llamo Samuel"]


def test_prunes_old_sessions(tmp_path):
    memory = _memory(tmp_path, max_sessions=2)
    for i in range(3):
        memory.start_session(f"s{i}")
        memory.add_exchange(f"m{i}", f"r{i}")

    stats = memory.stats()
    assert stats["sessions"] == 2
    messages = memory.recent_messages(limit=100)
    assert messages[-1].content == "r2"


def test_prunes_messages_per_session(tmp_path):
    memory = _memory(tmp_path, max_messages_per_session=4)
    for i in range(5):
        memory.add_exchange(f"p{i}", f"r{i}")

    assert memory.count_messages() == 4
    assert [m.content for m in memory.recent_messages(limit=10)] == ["p3", "r3", "p4", "r4"]


# --- Hechos ---


def test_remember_and_facts(tmp_path):
    memory = _memory(tmp_path)
    memory.remember("me gusta el café")

    assert memory.facts() == ["me gusta el café"]


def test_remember_is_idempotent(tmp_path):
    memory = _memory(tmp_path)
    memory.remember("un dato")
    memory.remember("un dato")

    assert memory.facts() == ["un dato"]


def test_remember_ignores_empty(tmp_path):
    memory = _memory(tmp_path)
    memory.remember("   ")

    assert memory.facts() == []


def test_forget(tmp_path):
    memory = _memory(tmp_path)
    memory.remember("un dato")

    assert memory.forget("un dato") is True
    assert memory.facts() == []
    assert memory.forget("un dato") is False


def test_prunes_facts(tmp_path):
    memory = _memory(tmp_path, max_facts=2)
    for i in range(4):
        memory.remember(f"dato {i}")

    assert memory.facts() == ["dato 2", "dato 3"]


def test_clear(tmp_path):
    memory = _memory(tmp_path)
    memory.add_exchange("a", "b")
    memory.remember("c")

    memory.clear()

    assert memory.stats() == {"sessions": 0, "messages": 0, "facts": 0}