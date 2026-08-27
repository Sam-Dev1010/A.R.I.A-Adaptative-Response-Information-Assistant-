"""Tests de las herramientas de desarrollador: read_file y run_command."""
from pathlib import Path

import pytest

from app.tools.dev_tools import RunCommandTool
from app.tools.file_tools import ReadFileTool

# --- read_file ---------------------------------------------------------------

async def test_read_file_devuelve_contenido(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    archivo = tmp_path / "proyecto" / "main.py"
    archivo.parent.mkdir(parents=True)
    archivo.write_text("def hola():\n    return 'mundo'\n", encoding="utf-8")

    salida = await ReadFileTool().execute(path="proyecto/main.py")

    assert "def hola():" in salida
    assert "líneas" in salida


async def test_read_file_rechaza_binarios_y_fantasma(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / "app.bin").write_bytes(b"\x00\x01\x02binario")
    tool = ReadFileTool()

    binario = await tool.execute(path="app.bin")
    fantasma = await tool.execute(path="no_existe.txt")

    assert "binario" in binario
    assert "No existe" in fantasma


async def test_read_file_trunca_archivos_enormes(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    grande = tmp_path / "grande.txt"
    grande.write_text("x" * 20_000, encoding="utf-8")

    salida = await ReadFileTool().execute(path="grande.txt")

    assert "truncado" in salida or "mostrados" in salida


# --- run_command -------------------------------------------------------------

async def test_run_command_ejecuta_y_devuelve_salida(tmp_path):
    tool = RunCommandTool()

    salida = await tool.execute(command="echo hola-sia")

    assert "hola-sia" in salida
    assert "código de salida: 0" in salida


async def test_run_command_reporta_errores(tmp_path):
    tool = RunCommandTool()

    salida = await tool.execute(command="ls /esta-ruta-no-existe-xyz 2>&1")

    assert "código de salida:" in salida
    assert "0" not in salida.split("código de salida:")[1].splitlines()[0]


async def test_run_command_respeta_cwd_dentro_del_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    proyecto = tmp_path / "proyectos" / "mi-app"
    proyecto.mkdir(parents=True)
    (proyecto / "README.md").write_text("# mi app")
    tool = RunCommandTool()

    salida = await tool.execute(command="ls", cwd="proyectos/mi-app")

    assert "README.md" in salida


async def test_run_command_rechaza_cwd_fuera_del_home():
    tool = RunCommandTool()

    with pytest.raises(ValueError, match="carpeta personal"):
        await tool.execute(command="echo x", cwd="/etc")


async def test_run_command_timeout_cancela():
    tool = RunCommandTool()

    salida = await tool.execute(command="sleep 30", timeout_seconds=5)

    assert "cancelé" in salida or "excedió" in salida


def test_run_command_es_restricted():
    from app.tools.base import ToolPermission

    assert RunCommandTool.permission == ToolPermission.RESTRICTED


def test_edge_tts_varia_pitch_y_rate():
    from app.voice.tts import EdgeTTSProvider

    tts = EdgeTTSProvider(rate="+3%", pitch="+0Hz", vary_rate=True)
    rates = {tts._siguiente_rate() for _ in range(6)}
    pitches = {tts._siguiente_pitch() for _ in range(6)}

    assert len(rates) > 1, "el ritmo debe variar entre frases"
    assert len(pitches) > 1, "el tono debe variar entre frases"
    assert all(r.startswith(("+", "-")) and r.endswith("%") for r in rates)
    assert all(p.endswith("Hz") for p in pitches)


def test_edge_tts_sin_vary_mantiene_valores():
    from app.voice.tts import EdgeTTSProvider

    tts = EdgeTTSProvider(rate="+3%", pitch="+0Hz", vary_rate=False)
    assert tts._siguiente_rate() == "+3%"
    assert tts._siguiente_pitch() == "+0Hz"
