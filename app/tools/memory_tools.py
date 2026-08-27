"""Herramientas de memoria a largo plazo (FASE 4)."""
from typing import ClassVar

from app.memory.manager import MemoryManager
from app.tools.base import BaseTool, ToolPermission


class RememberFactTool(BaseTool):
    """Guarda un dato o preferencia del usuario para recordarlo a futuro."""

    name = "remember"
    description = "Guarda un dato o preferencia del usuario para recordarlo en el futuro."
    permission = ToolPermission.SAFE
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "fact": {
                "type": "string",
                "description": "Dato a recordar, en forma de frase (ej: 'me llamo Samuel').",
            }
        },
        "required": ["fact"],
    }

    def __init__(self, memory: MemoryManager) -> None:
        self._memory = memory

    async def _run(self, fact: str, **kwargs) -> str:
        self._memory.remember(fact)
        return f"Recordado: {fact}"


class ForgetFactTool(BaseTool):
    """Elimina un recuerdo concreto (requiere confirmación)."""

    name = "forget"
    description = "Elimina un recuerdo concreto que SIA haya guardado antes."
    permission = ToolPermission.CONFIRM
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "fact": {
                "type": "string",
                "description": "Texto exacto del recuerdo a eliminar.",
            }
        },
        "required": ["fact"],
    }

    def __init__(self, memory: MemoryManager) -> None:
        self._memory = memory

    async def _run(self, fact: str, **kwargs) -> str:
        if self._memory.forget(fact):
            return f"Olvidado: {fact}"
        return f"No tenía guardado ese recuerdo: {fact}"