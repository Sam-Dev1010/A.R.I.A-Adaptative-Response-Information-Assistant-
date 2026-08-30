"""Herramientas de archivos: crear, listar y borrar en el PC del usuario.

Todas las rutas se resuelven contra el HOME del usuario y se valida que
queden dentro de él: SIA no puede tocar nada fuera de su carpeta personal.
"""
import shutil
from pathlib import Path
from typing import ClassVar

from app.tools.base import BaseTool, ToolPermission

_LIST_LIMIT = 50
_READ_LIMIT = 8000  # caracteres máximos leídos de un archivo por llamada


def safe_path(raw: str) -> Path:
    """Resuelve una ruta y garantiza que quede dentro del HOME del usuario.

    Las rutas absolutas y con ~ se usan tal cual. Las relativas se resuelven
    contra el directorio de trabajo actual (CWD) si apunta a algo existente
    ahí; en caso contrario, contra el HOME. Así "lee el archivo README.md"
    funciona desde la carpeta del proyecto y también desde el home.
    """
    home = Path.home().resolve()
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        cwd_candidate = Path.cwd() / candidate
        if cwd_candidate.exists():
            candidate = cwd_candidate
        else:
            candidate = home / candidate
    resolved = candidate.resolve()
    if resolved != home and home not in resolved.parents:
        raise ValueError(
            f"Solo puedo operar dentro de tu carpeta personal ({home})"
        )
    return resolved


class CreateFolderTool(BaseTool):
    """Crea una carpeta (y las carpetas padre si hacen falta)."""

    name = "create_folder"
    description = "Crea una carpeta en el PC (rutas relativas desde el home)."
    permission = ToolPermission.CONFIRM
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Ruta de la carpeta."}
        },
        "required": ["path"],
    }

    async def _run(self, path: str, **kwargs) -> str:
        target = safe_path(path)
        target.mkdir(parents=True, exist_ok=True)
        return f"Carpeta lista: {target}"


class CreateFileTool(BaseTool):
    """Crea un archivo de texto con contenido opcional."""

    name = "create_file"
    description = "Crea un archivo de texto; no sobrescribe salvo overwrite=true."
    permission = ToolPermission.CONFIRM
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Ruta del archivo (relativa al home si es relativa).",
            },
            "content": {"type": "string", "description": "Contenido del archivo."},
            "overwrite": {"type": "boolean"},
        },
        "required": ["path"],
    }

    async def _run(
        self, path: str, content: str = "", overwrite: bool = False, **kwargs
    ) -> str:
        target = safe_path(path)
        if target.exists() and not overwrite:
            return (
                f"{target} ya existe y no lo sobrescribí. "
                "Si de verdad quieres reemplazarlo, pídemelo."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        action = "sobrescrito" if overwrite else "creado"
        return f"Archivo {action}: {target} ({len(content)} caracteres)"


class ReadFileTool(BaseTool):
    """Lee un archivo de texto o código (trunca archivos enormes)."""

    name = "read_file"
    description = (
        "Lee un archivo de texto o código del PC (código fuente, configs, logs…)."
    )
    permission = ToolPermission.SAFE
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Ruta del archivo (relativa al home si es relativa).",
            }
        },
        "required": ["path"],
    }

    async def _run(self, path: str, **kwargs) -> str:
        target = safe_path(path)
        if not target.is_file():
            return f"No existe el archivo: {target}"
        data = target.read_bytes()
        if b"\x00" in data[:4096]:
            return f"{target.name} parece un archivo binario: solo leo texto o código."
        text = data.decode("utf-8", errors="replace")
        total_lineas = text.count("\n") + 1
        contenido = text[:_READ_LIMIT]
        truncado = (
            f"\n\n…(mostrados {len(contenido)} de {len(text)} caracteres)"
            if len(text) > _READ_LIMIT
            else ""
        )
        return f"{target} ({total_lineas} líneas):\n\n{contenido}{truncado}"


class ListFilesTool(BaseTool):
    """Lista el contenido de una carpeta."""

    name = "list_files"
    description = "Lista archivos y carpetas de una ruta (vacío = carpeta personal)."
    permission = ToolPermission.SAFE
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Carpeta a listar (opcional)."}
        },
        "required": [],
    }

    async def _run(self, path: str = "", **kwargs) -> str:
        target = safe_path(path) if path.strip() else Path.home()
        if not target.is_dir():
            return f"{target} no existe o no es una carpeta."
        entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        lines = [
            f"[carpeta] {e.name}" if e.is_dir() else f"[archivo] {e.name}"
            for e in entries[:_LIST_LIMIT]
        ]
        if len(entries) > _LIST_LIMIT:
            lines.append(f"…y {len(entries) - _LIST_LIMIT} más")
        header = f"{target}: {len(entries)} elementos"
        return header + ("\n" + "\n".join(lines) if lines else " (vacía)")


class DeletePathTool(BaseTool):
    """Borra un archivo o carpeta (permanente, sin papelera)."""

    name = "delete_path"
    description = (
        "Borra permanentemente un archivo o carpeta del PC del usuario. "
        "¡Peligroso! Pide confirmación siempre."
    )
    permission = ToolPermission.RESTRICTED
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Ruta a borrar."}
        },
        "required": ["path"],
    }

    async def _run(self, path: str, **kwargs) -> str:
        target = safe_path(path)
        home = Path.home().resolve()
        if target == home:
            raise ValueError("No voy a borrar tu carpeta personal completa.")
        if target.is_dir():
            shutil.rmtree(target)
            return f"Carpeta borrada: {target}"
        if target.exists():
            target.unlink()
            return f"Archivo borrado: {target}"
        return f"No existe nada en {target}."
