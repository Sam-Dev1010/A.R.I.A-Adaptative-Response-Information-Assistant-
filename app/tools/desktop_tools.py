"""Herramientas de escritorio: abrir aplicaciones instaladas en el PC.

Se lanzan a través de sus entradas ``.desktop`` (estándar freedesktop.org),
las mismas que usa el menú de aplicaciones: SIA solo puede abrir programas
registrados en el sistema, nunca binarios arbitrarios.
"""
import asyncio
import os
import shlex
import shutil
import subprocess
import webbrowser
from pathlib import Path
from typing import ClassVar

from app.tools.base import BaseTool, ToolPermission

_LOCAL_APPLICATIONS = Path.home() / ".local" / "share" / "applications"


def _desktop_dirs() -> list[Path]:
    """Directorios XDG donde viven las entradas .desktop del sistema."""
    dirs = [_LOCAL_APPLICATIONS]
    xdg = os.environ.get("XDG_DATA_DIRS", "/usr/local/share:/usr/share")
    dirs.extend(Path(part) / "applications" for part in xdg.split(":") if part.strip())
    return dirs


def _parse_field(path: Path, field: str) -> str:
    """Lee un campo (Name=, Exec=...) de la sección [Desktop Entry]."""
    try:
        in_entry = False
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if line == "[Desktop Entry]":
                    in_entry = True
                elif line.startswith("[") and line.endswith("]"):
                    in_entry = False
                elif in_entry and line.lower().startswith(f"{field.lower()}="):
                    return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return ""


class OpenAppTool(BaseTool):
    """Abre una aplicación instalada por nombre (vía su launcher .desktop)."""

    name = "open_app"
    description = "Abre una app del PC por nombre ('firefox', 'spotify'…)."
    permission = ToolPermission.CONFIRM
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "app": {"type": "string", "description": "Nombre de la app."}
        },
        "required": ["app"],
    }

    @staticmethod
    def find_launcher(query: str) -> tuple[Path, str] | None:
        """Busca un launcher .desktop cuyo id o Name coincida con la consulta."""
        q = query.strip().lower()
        if not q:
            return None
        seen: set[str] = set()
        partial: tuple[Path, str] | None = None
        for directory in _desktop_dirs():
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.desktop")):
                if path.stem in seen:
                    continue
                seen.add(path.stem)
                display = _parse_field(path, "Name") or path.stem
                stem = path.stem.lower()
                name = display.lower()
                if q == stem or q == name:
                    return path, display
                if partial is None and (q in stem or q in name):
                    partial = (path, display)
        return partial

    async def _run(self, app: str, **kwargs) -> str:
        match = self.find_launcher(app)
        if match is None:
            installed = sorted(
                {_parse_field(p, "Name") or p.stem
                 for d in _desktop_dirs() if d.is_dir()
                 for p in d.glob("*.desktop")}
            )
            sample = ", ".join(installed[:25])
            extra = f" …y {len(installed) - 25} más" if len(installed) > 25 else ""
            return (
                f"No encontré una aplicación llamada {app!r}. "
                f"Algunas instaladas: {sample}{extra}"
            )

        path, display = match
        self._launch(path)
        return f"Abrí {display}."

    @staticmethod
    def _launch(path: Path) -> None:
        """Lanza la app sin bloquear y sin acoplar el proceso a SIA."""
        gio = shutil.which("gio")
        if gio:
            cmd = [gio, "launch", str(path)]
        else:
            exec_line = _parse_field(path, "Exec")
            if not exec_line:
                raise RuntimeError(f"El launcher {path.name} no tiene campo Exec")
            # Se descartan los códigos de campo (%f, %u, ...) del estándar.
            cmd = [part for part in shlex.split(exec_line) if not part.startswith("%")]
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )


_EDITORES_POR_DEFECTO = ("code", "codium", "cursor", "zed", "subl", "kate", "gedit")

_EDITOR_ALIASES = {
    "vscode": "code",
    "vs code": "code",
    "visual studio code": "code",
    "code": "code",
    "codium": "codium",
    "vscodium": "codium",
    "cursor": "cursor",
    "zed": "zed",
    "sublime": "subl",
    "sublime text": "subl",
}


class OpenFolderTool(BaseTool):
    """Abre una carpeta o proyecto en VS Code u otro editor/IDE instalado."""

    name = "open_folder"
    description = "Abre una carpeta o proyecto en un editor de código (VS Code…)."
    permission = ToolPermission.CONFIRM
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "folder": {
                "type": "string",
                "description": "Ruta de la carpeta (relativa al home si es relativa).",
            },
            "editor": {
                "type": "string",
                "description": "Editor deseado ('vs code', 'zed'…). Vacío = el mejor.",
            },
        },
        "required": ["folder"],
    }

    @staticmethod
    def resolve_folder(folder: str) -> Path:
        """Convierte la consulta en ruta absoluta (~ y relativas van al home)."""
        path = Path(folder.strip().strip('"').strip("'")).expanduser()
        if not path.is_absolute():
            path = Path.home() / path
        return path

    def _command_for(self, editor: str, path: Path) -> list[str] | None:
        """Comando para abrir ``path`` en un editor; None si no hay ninguno."""
        query = editor.strip().lower()
        if not query:
            for candidate in _EDITORES_POR_DEFECTO:
                binary = shutil.which(candidate)
                if binary:
                    return [binary, str(path)]
            return None
        binario_alias = _EDITOR_ALIASES.get(query)
        if binario_alias:
            found = shutil.which(binario_alias)
            if found:
                return [found, str(path)]
        # Editor sin CLI conocida: buscar su launcher .desktop y usar su Exec.
        match = OpenAppTool.find_launcher(query)
        if match is not None:
            exec_line = _parse_field(match[0], "Exec")
            if exec_line:
                return self._with_path_argument(exec_line, path)
        return None

    @staticmethod
    def _with_path_argument(exec_line: str, path: Path) -> list[str]:
        """Sustituye los códigos de campo (%f %u %F %U) del Exec por la ruta."""
        cmd: list[str] = []
        replaced = False
        for part in shlex.split(exec_line):
            if part.startswith("%"):
                if not replaced:
                    cmd.append(str(path))
                    replaced = True
                continue
            cmd.append(part)
        if not replaced:
            cmd.append(str(path))
        return cmd

    async def _run(self, folder: str, editor: str = "", **kwargs) -> str:
        path = self.resolve_folder(folder)
        if not path.is_dir():
            return (
                f"No encontré la carpeta {folder!r}. Dame una ruta absoluta "
                f"o relativa a tu home ({Path.home()})."
            )
        command = self._command_for(editor, path)
        if command is not None:
            self._launch(command)
            return f"Abrí {path} en {Path(command[0]).name}."
        if editor.strip():
            return (
                f"No encontré el editor {editor!r}. Prueba con 'vs code', "
                "'vscodium', 'zed', 'kate' o 'sublime'."
            )
        xdg_open = shutil.which("xdg-open")
        if xdg_open:
            self._launch([xdg_open, str(path)])
            return (
                f"No tengo ningún editor de código instalado; abrí {path} en el "
                "gestor de archivos. Instala VS Code y podré abrirlo ahí."
            )
        return "No hay editores de código ni gestor de archivos disponibles."

    @staticmethod
    def _launch(command: list[str]) -> None:
        """Lanza el editor sin bloquear y sin acoplar el proceso a SIA."""
        subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )


class PlayMusicTool(BaseTool):
    """Reproduce música: audio real con yt-dlp+mpv, o búsqueda en el navegador."""

    name = "play_music"
    description = (
        "Reproduce una canción por nombre o artista (audio real si hay "
        "yt-dlp+mpv; si no, abre la búsqueda en YouTube/Spotify)."
    )
    permission = ToolPermission.CONFIRM
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "song": {"type": "string", "description": "Canción o artista."},
            "service": {
                "type": "string",
                "enum": ["youtube", "youtube_music", "spotify"],
            },
        },
        "required": ["song"],
    }

    async def _run(self, song: str, service: str = "youtube", **kwargs) -> str:
        played = await self._play_audio(song)
        if played:
            return f"Reproduciendo '{song}'."
        webbrowser.open(self._search_url(song, service))
        return (
            f"Busqué '{song}' en {service}. Para que suene sola, instala "
            f"yt-dlp y mpv (sudo dnf install yt-dlp mpv)."
        )

    @staticmethod
    async def _play_audio(song: str) -> bool:
        """Reproduce el primer resultado de YouTube con mpv. False si no hay tools."""
        yt_dlp, mpv = shutil.which("yt-dlp"), shutil.which("mpv")
        if not yt_dlp or not mpv:
            return False
        proc = await asyncio.create_subprocess_exec(
            yt_dlp,
            "-f",
            "bestaudio",
            "--no-playlist",
            "--get-url",
            f"ytsearch1:{song}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        url = stdout.decode(errors="replace").strip().splitlines()
        if proc.returncode != 0 or not url or not url[0].startswith("http"):
            return False
        await asyncio.create_subprocess_exec(
            mpv,
            "--no-video",
            "--no-terminal",
            "--really-quiet",
            url[0],
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
        return True

    @staticmethod
    def _search_url(song: str, service: str) -> str:
        from urllib.parse import quote_plus

        q = quote_plus(song)
        urls = {
            "youtube": f"https://www.youtube.com/results?search_query={q}",
            "youtube_music": f"https://music.youtube.com/search?q={q}",
            "spotify": f"https://open.spotify.com/search/{q}",
        }
        return urls.get(service, urls["youtube"])


class MediaControlTool(BaseTool):
    """Controla la reproducción del sistema (playerctl): play/pause/volumen…"""

    name = "media_control"
    description = (
        "Controla el multimedia del sistema: play, pause, pista siguiente/"
        "anterior, qué suena y volumen."
    )
    permission = ToolPermission.SAFE
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["play", "pause", "play-pause", "next", "previous", "status", "volume"],
            },
            "value": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "Volumen 0-100 (solo action=volume).",
            },
        },
        "required": ["action"],
    }

    async def _run(self, action: str, value: int | None = None, **kwargs) -> str:
        playerctl = shutil.which("playerctl")
        if not playerctl:
            return (
                "No puedo controlar la reproducción: playerctl no está "
                "instalado (sudo dnf install playerctl)."
            )
        if action == "volume":
            if value is None:
                return "Dime un nivel de volumen entre 0 y 100."
            level = max(0, min(value, 100))
            cmd = [playerctl, "volume", f"{level / 100:.2f}"]
        else:
            cmd = [playerctl, action]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        except (subprocess.SubprocessError, OSError):
            return "El control multimedia no respondió."
        output = (stdout or b"").decode(errors="replace").strip()
        if action == "status":
            return f"Estado: {output or 'nada en reproducción'}."
        if action == "volume":
            return f"Volumen ajustado a {level}%."
        labels = {"play-pause": "reproducción alternada", "next": "pista siguiente",
                  "previous": "pista anterior"}
        return f"Hecho: {labels.get(action, action)}."
