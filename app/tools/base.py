"""Base del sistema de herramientas (FASE 3)."""
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import ClassVar

from app.tools.schemas import FunctionSpec, ToolSpec


class ToolPermission(StrEnum):
    """Nivel de riesgo de una herramienta:

    - ``SAFE``: solo lectura/inofensiva; se ejecuta sin preguntar.
    - ``CONFIRM``: puede tener efectos; requiere confirmación del usuario.
    - ``RESTRICTED``: potencialmente peligrosa; requiere confirmación siempre.
    """

    SAFE = "safe"
    CONFIRM = "confirm"
    RESTRICTED = "restricted"


class BaseTool(ABC):
    """Contrato de una herramienta ejecutable por el LLM.

    Las subclases definen ``name``, ``description``, ``permission``,
    ``parameters`` (JSON Schema) e implementan ``_run``.
    """

    name: str = ""
    description: str = ""
    permission: ToolPermission = ToolPermission.SAFE
    parameters: ClassVar[dict] = {}

    def to_spec(self) -> ToolSpec:
        return ToolSpec(
            function=FunctionSpec(
                name=self.name,
                description=self.description,
                parameters=self.parameters,
            )
        )

    async def execute(self, **kwargs) -> str:
        """Punto de entrada: valida y ejecuta la herramienta."""
        return await self._run(**kwargs)

    @abstractmethod
    async def _run(self, **kwargs) -> str:
        """Implementación concreta. Devuelve texto listo para el LLM."""