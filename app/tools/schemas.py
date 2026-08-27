"""Modelos de datos del sistema de herramientas (FASE 3)."""
from typing import Any

from pydantic import BaseModel, Field


class FunctionSpec(BaseModel):
    """Descripción de una función para la API del LLM (formato OpenAI)."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )


class ToolSpec(BaseModel):
    """Especificación de una herramienta enviada al LLM."""

    type: str = "function"
    function: FunctionSpec


class ToolResult(BaseModel):
    """Resultado de ejecutar una herramienta, listo para devolver al LLM."""

    tool_call_id: str
    name: str
    ok: bool
    output: str