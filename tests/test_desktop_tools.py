"""Tests de las herramientas open_app y open_folder (escritorio)."""
from pathlib import Path

import pytest

from app.tools.desktop_tools import OpenAppTool, OpenFolderTool, _parse_field


@pytest.fixture
def fake_apps(tmp_path: Path, monkeypatch) -> Path:
    """Crea launchers .desktop falsos y redirige la búsqueda a ellos."""
    apps = tmp_path / "applications"
    apps.mkdir()
    (apps / "firefox.desktop").write_text(
        "[Desktop Entry]\nName=Firefox\nExec=firefox %u\nType=Application\n",
        encoding="utf-8",
    )
    (apps / "org.gnome.Calculator.desktop").write_text(
        "[Desktop Entry]\nName=Calculadora\nExec=gnome-calculator\nType=Application\n",
        encoding="utf-8",
    )
    (apps / "jetbrains-phpstorm.desktop").write_text(
        "[Desktop Entry]\nName=PhpStorm\nExec=phpstorm %f\nType=Application\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_DATA_DIRS", str(tmp_path))
    monkeypatch.setattr("app.tools.desktop_tools._LOCAL_APPLICATIONS", apps / "no-existe")
    return apps


def test_find_launcher_exact_by_stem(fake_apps):
    match = OpenAppTool.find_launcher("firefox")

    assert match is not None
    assert match[0].name == "firefox.desktop"


def test_find_launcher_exact_by_display_name(fake_apps):
    match = OpenAppTool.find_launcher("calculadora")

    assert match is not None
    assert match[0].name == "org.gnome.Calculator.desktop"


def test_find_launcher_partial_match(fake_apps):
    match = OpenAppTool.find_launcher("fire")

    assert match is not None
    assert match[0].stem == "firefox"


def test_find_launcher_unknown_returns_none(fake_apps):
    assert OpenAppTool.find_launcher("noexiste") is None


def test_parse_field_reads_name_and_exec(fake_apps):
    path = fake_apps / "firefox.desktop"

    assert _parse_field(path, "Name") == "Firefox"
    assert _parse_field(path, "Exec") == "firefox %u"
    assert _parse_field(path, "Icon") == ""


async def test_run_opens_app_with_gio(fake_apps, monkeypatch):
    launched = {}

    def fake_popen(cmd, **kwargs):
        launched["cmd"] = cmd

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/gio" if name == "gio" else None)
    monkeypatch.setattr("subprocess.Popen", fake_popen)

    result = await OpenAppTool().execute(app="firefox")

    assert result == "Abrí Firefox."
    assert launched["cmd"] == ["/usr/bin/gio", "launch", str(fake_apps / "firefox.desktop")]


async def test_run_falls_back_to_exec_without_gio(fake_apps, monkeypatch):
    launched = {}
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr("subprocess.Popen", lambda cmd, **kw: launched.update(cmd=cmd))

    await OpenAppTool().execute(app="calculadora")

    assert launched["cmd"] == ["gnome-calculator"]


async def test_run_unknown_app_lists_installed(fake_apps):
    result = await OpenAppTool().execute(app="totamente-inexistente")

    assert "No encontré" in result
    assert "Firefox" in result and "Calculadora" in result


# --- open_folder ---


def test_open_folder_resolves_relative_paths_against_home():
    ruta = OpenFolderTool.resolve_folder("Documentos/sia")

    assert ruta == Path.home() / "Documentos" / "sia"


async def test_open_folder_uses_requested_editor(tmp_path, monkeypatch):
    launched = {}
    carpeta = tmp_path / "proyecto"
    carpeta.mkdir()
    monkeypatch.setattr(
        "shutil.which", lambda name: "/usr/bin/code" if name == "code" else None
    )
    monkeypatch.setattr("subprocess.Popen", lambda cmd, **kw: launched.update(cmd=cmd))

    result = await OpenFolderTool().execute(folder=str(carpeta), editor="vs code")

    assert result == f"Abrí {carpeta} en code."
    assert launched["cmd"] == ["/usr/bin/code", str(carpeta)]


async def test_open_folder_picks_first_installed_editor(tmp_path, monkeypatch):
    launched = {}
    carpeta = tmp_path / "proyecto"
    carpeta.mkdir()
    monkeypatch.setattr(
        "shutil.which", lambda name: "/usr/bin/codium" if name == "codium" else None
    )
    monkeypatch.setattr("subprocess.Popen", lambda cmd, **kw: launched.update(cmd=cmd))

    result = await OpenFolderTool().execute(folder=str(carpeta))

    assert launched["cmd"] == ["/usr/bin/codium", str(carpeta)]
    assert "codium" in result


async def test_open_folder_via_desktop_launcher(fake_apps, tmp_path, monkeypatch):
    """Un editor sin CLI conocida se abre por su .desktop sustituyendo %f."""
    launched = {}
    carpeta = tmp_path / "proyecto"
    carpeta.mkdir()
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr("subprocess.Popen", lambda cmd, **kw: launched.update(cmd=cmd))

    await OpenFolderTool().execute(folder=str(carpeta), editor="phpstorm")

    assert launched["cmd"] == ["phpstorm", str(carpeta)]


async def test_open_folder_missing_reports_error(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)

    result = await OpenFolderTool().execute(folder="/no/existe/xyz")

    assert "No encontré la carpeta" in result


async def test_open_folder_unknown_editor_suggests_options(monkeypatch, tmp_path):
    carpeta = tmp_path / "proyecto"
    carpeta.mkdir()
    monkeypatch.setattr("shutil.which", lambda name: None)

    result = await OpenFolderTool().execute(folder=str(carpeta), editor="inexistente")

    assert "No encontré el editor" in result
