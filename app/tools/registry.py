"""Registro central de herramientas (FASE 3)."""
from app.tools.base import BaseTool
from app.tools.schemas import FunctionSpec, ToolSpec


def _compactar_parametros(params: dict) -> dict:
    """Reduce el esquema JSON de parámetros a lo esencial (tipo + enum).

    Los nombres de los parámetros ya se explican solos; las descripciones
    y restricciones extra engordan cada request del LLM y lo vuelven lento.
    """
    if not isinstance(params, dict):
        return {"type": "object", "properties": {}}
    props = params.get("properties")
    compactas: dict = {}
    if isinstance(props, dict):
        for nombre, esquema in props.items():
            limpio: dict = {}
            if isinstance(esquema, dict):
                if "type" in esquema:
                    limpio["type"] = esquema["type"]
                if "enum" in esquema:
                    limpio["enum"] = esquema["enum"]
                if "items" in esquema:
                    limpio["items"] = _compactar_parametros(esquema["items"]).get(
                        "properties"
                    ) or esquema["items"]
            compactas[nombre] = limpio
    compacto = {"type": "object", "properties": compactas}
    requeridos = params.get("required")
    if requeridos:
        compacto["required"] = list(requeridos)
    return compacto


class ToolRegistry:
    """Colección de herramientas registradas, accesibles por nombre único."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if not tool.name:
            raise ValueError("Las herramientas necesitan un nombre no vacío")
        if tool.name in self._tools:
            raise ValueError(f"Herramienta duplicada: {tool.name!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def all(self) -> list[BaseTool]:
        return sorted(self._tools.values(), key=lambda tool: tool.name)

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                type="function",
                function=FunctionSpec(
                    name=tool.name,
                    description=tool.description,
                    parameters=_compactar_parametros(tool.parameters),
                ),
            )
            for tool in self.all()
        ]

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())