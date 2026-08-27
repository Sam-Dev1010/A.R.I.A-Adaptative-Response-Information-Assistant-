"""Modelos de datos de la capa de IA (FASES 2-3)."""
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ChatRole(StrEnum):
    """Roles soportados en una conversación con el LLM."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """Solicitud del LLM de ejecutar una herramienta."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    """Un mensaje individual dentro de la conversación.

    Campos opcionales según el rol:
    - ``tool_call_id``: obligatorio en mensajes de rol TOOL.
    - ``tool_calls``: presente en mensajes de rol ASSISTANT que piden herramientas.
    """

    role: ChatRole
    content: str = ""
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


class TokenUsage(BaseModel):
    """Consumo de tokens reportado por el proveedor."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ChatResponse(BaseModel):
    """Respuesta del LLM a una llamada de chat."""

    content: str
    model: str
    provider: str
    usage: TokenUsage | None = None
    tool_calls: list[ToolCall] | None = None