"""Política de permisos de las herramientas (FASE 3)."""
from app.tools.base import BaseTool, ToolPermission


class PermissionDenied(Exception):
    """Lanzada cuando una llamada de herramienta no está permitida."""

    def __init__(self, tool_name: str, reason: str) -> None:
        self.tool_name = tool_name
        self.reason = reason
        super().__init__(f"Herramienta {tool_name!r}: {reason}")


class ToolPolicy:
    """Decide si una herramienta puede ejecutarse.

    - ``allowed``: herramientas habilitadas (None = todas).
    - ``auto_confirm``: herramientas CONFIRM que no piden confirmación.
    """

    def __init__(
        self,
        *,
        allowed: set[str] | None = None,
        auto_confirm: set[str] | None = None,
    ) -> None:
        self._allowed = allowed
        self._auto_confirm = auto_confirm or set()

    def check(self, tool: BaseTool, *, confirmed: bool = False) -> None:
        """Lanza :class:`PermissionDenied` si la ejecución no está permitida."""
        if self._allowed is not None and tool.name not in self._allowed:
            raise PermissionDenied(tool.name, "herramienta deshabilitada")

        if tool.permission is ToolPermission.SAFE:
            return
        if confirmed:
            return
        if tool.permission is ToolPermission.CONFIRM and tool.name in self._auto_confirm:
            return

        raise PermissionDenied(tool.name, "requiere confirmación del usuario")