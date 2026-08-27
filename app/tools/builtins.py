"""Herramientas integradas de SIA (FASE 3)."""
import json
import os
import platform
import socket
import webbrowser
from datetime import datetime
from typing import ClassVar

from app.tools.base import BaseTool, ToolPermission


class GetTimeTool(BaseTool):
    """Fecha y hora actuales en la zona horaria local."""

    name = "get_time"
    description = "Devuelve la fecha y hora actuales en la zona horaria local."
    permission = ToolPermission.SAFE

    async def _run(self, **kwargs) -> str:
        now = datetime.now().astimezone()
        return now.strftime("%A, %d de %B de %Y — %H:%M:%S %Z")


class GetSystemInfoTool(BaseTool):
    """Información básica y de solo lectura del sistema."""

    name = "get_system_info"
    description = "Devuelve información básica del sistema (SO, hostname, CPU, Python)."
    permission = ToolPermission.SAFE

    async def _run(self, **kwargs) -> str:
        info = {
            "hostname": socket.gethostname(),
            "os": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        }
        return json.dumps(info, ensure_ascii=False, indent=2)


class OpenWebsiteTool(BaseTool):
    """Abre una URL en el navegador del sistema (requiere confirmación)."""

    name = "open_website"
    description = "Abre una URL en el navegador web del sistema."
    permission = ToolPermission.CONFIRM
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL completa (ej: https://example.com)",
            }
        },
        "required": ["url"],
    }

    async def _run(self, url: str, **kwargs) -> str:
        webbrowser.open(url)
        return f"Abrí {url} en el navegador."


BUILTIN_TOOLS: list[type[BaseTool]] = [
    GetTimeTool,
    GetSystemInfoTool,
    OpenWebsiteTool,
]