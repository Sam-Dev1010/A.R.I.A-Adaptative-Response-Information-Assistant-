"""Herramientas de desarrollo: ejecutar comandos en el PC del usuario.

Con ``run_command`` SIA puede hacer vida de desarrolladora: correr pruebas,
usar git, compilar, instalar dependencias, revisar logs... El comando se
ejecuta dentro del HOME del usuario y siempre pide confirmación
(RESTRICTED), salvo que la interfaz o el usuario la hayan aprobado.
"""
import asyncio
import subprocess
from pathlib import Path
from typing import ClassVar

from app.tools.base import BaseTool, ToolPermission
from app.tools.file_tools import safe_path

_TIMEOUT_DEFAULT = 60.0
_TIMEOUT_MIN = 5.0
_TIMEOUT_MAX = 240.0
_STDOUT_LIMIT = 6000
_STDERR_LIMIT = 2000


def _truncar(texto: str, limite: int) -> str:
    if len(texto) <= limite:
        return texto
    return texto[:limite] + f"\n…(truncado, {len(texto)} caracteres en total)"


class RunCommandTool(BaseTool):
    """Ejecuta un comando de shell con salida capturada y tiempo límite."""

    name = "run_command"
    description = (
        "Ejecuta un comando bash en el PC (dentro del home): pruebas, git, "
        "instalar dependencias, tareas con sudo/root… Devuelve código de "
        "salida, stdout y stderr."
    )
    permission = ToolPermission.RESTRICTED
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Comando a ejecutar ('pytest -q', 'git status'…).",
            },
            "cwd": {
                "type": "string",
                "description": "Carpeta de trabajo, relativa al home. Vacío = home.",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Espera máxima 5-240 s (defecto 60).",
            },
        },
        "required": ["command"],
    }

    async def _run(
        self,
        command: str,
        cwd: str = "",
        timeout_seconds: float | None = None,
        **kwargs,
    ) -> str:
        command = command.strip()
        if not command:
            return "No me diste ningún comando."
        if "\x00" in command:
            return "Comando inválido."

        workdir = Path.home().resolve()
        if cwd.strip():
            workdir = safe_path(cwd)
            if not workdir.is_dir():
                return f"La carpeta de trabajo no existe: {workdir}"

        try:
            timeout = float(timeout_seconds or _TIMEOUT_DEFAULT)
        except (TypeError, ValueError):
            timeout = _TIMEOUT_DEFAULT
        timeout = max(_TIMEOUT_MIN, min(_TIMEOUT_MAX, timeout))

        try:
            # En un hilo: el proceso no debe bloquear el event loop del server.
            proc = await asyncio.to_thread(
                subprocess.run,
                ["bash", "-lc", command],
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return (
                f"$ {command}\n[en {workdir}]\n"
                f"El comando excedió {timeout:.0f} s y lo cancelé."
            )
        except OSError as exc:
            return f"No pude ejecutar el comando: {exc}"

        partes = [
            f"$ {command}",
            f"[en {workdir}]",
            f"código de salida: {proc.returncode}",
        ]
        if proc.stdout.strip():
            partes.append(f"SALIDA:\n{_truncar(proc.stdout.strip(), _STDOUT_LIMIT)}")
        if proc.stderr.strip():
            partes.append(f"ERRORES:\n{_truncar(proc.stderr.strip(), _STDERR_LIMIT)}")
        return "\n".join(partes)
