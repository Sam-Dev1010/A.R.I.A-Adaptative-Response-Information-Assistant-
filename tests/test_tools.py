"""Tests del sistema de herramientas: base, registro, permisos y builtins."""
from typing import ClassVar

import pytest

from app.tools.base import BaseTool, ToolPermission
from app.tools.builtins import GetSystemInfoTool, GetTimeTool, OpenWebsiteTool
from app.tools.policy import PermissionDenied, ToolPolicy
from app.tools.registry import ToolRegistry


class EchoTool(BaseTool):
    name = "echo"
    description = "Devuelve el texto recibido."
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def _run(self, text: str, **kwargs) -> str:
        return f"eco: {text}"


class ConfirmTool(BaseTool):
    name = "confirm_tool"
    description = "Herramienta que requiere confirmación."
    permission = ToolPermission.CONFIRM

    async def _run(self, **kwargs) -> str:
        return "ejecutada"


class RestrictedTool(BaseTool):
    name = "restricted_tool"
    description = "Herramienta restringida."
    permission = ToolPermission.RESTRICTED

    async def _run(self, **kwargs) -> str:
        return "ejecutada"


# --- Registry ---


def test_register_and_get():
    registry = ToolRegistry()
    tool = EchoTool()
    registry.register(tool)

    assert registry.get("echo") is tool
    assert len(registry) == 1


def test_register_duplicate_raises():
    registry = ToolRegistry()
    registry.register(EchoTool())

    with pytest.raises(ValueError, match="duplicada"):
        registry.register(EchoTool())


def test_register_empty_name_raises():
    registry = ToolRegistry()

    class NoNameTool(BaseTool):
        async def _run(self, **kwargs) -> str:
            return ""

    with pytest.raises(ValueError, match="nombre"):
        registry.register(NoNameTool())


def test_get_unknown_returns_none():
    assert ToolRegistry().get("nope") is None


def test_specs_format():
    registry = ToolRegistry()
    registry.register(EchoTool())

    spec = registry.specs()[0]
    assert spec.type == "function"
    assert spec.function.name == "echo"
    assert spec.function.parameters["required"] == ["text"]


# --- Policy ---


def test_policy_allows_safe_by_default():
    ToolPolicy().check(EchoTool())


def test_policy_disabled_tool_raises():
    policy = ToolPolicy(allowed={"echo"})
    policy.check(EchoTool())

    with pytest.raises(PermissionDenied, match="deshabilitada"):
        policy.check(ConfirmTool())


def test_policy_confirm_requires_confirmation():
    policy = ToolPolicy()
    with pytest.raises(PermissionDenied, match="confirmación"):
        policy.check(ConfirmTool())

    policy.check(ConfirmTool(), confirmed=True)


def test_policy_auto_confirm_skips_question():
    policy = ToolPolicy(auto_confirm={"confirm_tool"})
    policy.check(ConfirmTool())


def test_policy_restricted_never_auto_confirmed():
    policy = ToolPolicy(auto_confirm={"restricted_tool"})
    with pytest.raises(PermissionDenied):
        policy.check(RestrictedTool())

    policy.check(RestrictedTool(), confirmed=True)


# --- Builtins ---


@pytest.mark.asyncio
async def test_get_time_returns_text():
    output = await GetTimeTool().execute()
    assert len(output) > 10
    assert any(c.isdigit() for c in output)


@pytest.mark.asyncio
async def test_get_system_info_returns_json():
    output = await GetSystemInfoTool().execute()
    assert "hostname" in output
    assert "os" in output


@pytest.mark.asyncio
async def test_open_website_calls_browser(monkeypatch):
    opened = []

    def fake_open(url):
        opened.append(url)

    monkeypatch.setattr("app.tools.builtins.webbrowser.open", fake_open)

    output = await OpenWebsiteTool().execute(url="https://example.com")

    assert opened == ["https://example.com"]
    assert "example.com" in output


# --- Herramientas de memoria (FASE 4) ---


@pytest.mark.asyncio
async def test_remember_tool_stores_fact(tmp_path):
    from app.memory.manager import MemoryManager
    from app.tools.memory_tools import RememberFactTool

    memory = MemoryManager(tmp_path / "sia.db").open()
    output = await RememberFactTool(memory).execute(fact="me llamo Samuel")

    assert output == "Recordado: me llamo Samuel"
    assert memory.facts() == ["me llamo Samuel"]


@pytest.mark.asyncio
async def test_forget_tool_removes_fact(tmp_path):
    from app.memory.manager import MemoryManager
    from app.tools.memory_tools import ForgetFactTool

    memory = MemoryManager(tmp_path / "sia.db").open()
    memory.remember("me llamo Samuel")

    output = await ForgetFactTool(memory).execute(fact="me llamo Samuel")

    assert "Olvidado" in output
    assert memory.facts() == []