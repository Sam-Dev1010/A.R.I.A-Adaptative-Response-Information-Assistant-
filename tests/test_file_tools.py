"""Tests de las herramientas de archivos (crear/listar/borrar)."""
from pathlib import Path

import pytest

from app.tools.file_tools import (
    CreateFileTool,
    CreateFolderTool,
    DeletePathTool,
    ListFilesTool,
    safe_path,
)


def test_safe_path_resolves_relative_to_home():
    resolved = safe_path("Documentos/x")

    assert resolved == Path.home() / "Documentos" / "x"


def test_safe_path_rejects_outside_home():
    with pytest.raises(ValueError, match="carpeta personal"):
        safe_path("/etc/passwd")
    with pytest.raises(ValueError, match="carpeta personal"):
        safe_path("../../etc")


async def test_create_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    result = await CreateFolderTool().execute(path="Documentos/SIA/pruebas")

    assert "Carpeta lista" in result
    assert (tmp_path / "Documentos" / "SIA" / "pruebas").is_dir()


async def test_create_file_and_refuse_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    tool = CreateFileTool()

    created = await tool.execute(path="notas.txt", content="hola mundo")

    assert "creado" in created
    assert (tmp_path / "notas.txt").read_text(encoding="utf-8") == "hola mundo"

    refused = await tool.execute(path="notas.txt", content="otro")

    assert "ya existe" in refused
    assert (tmp_path / "notas.txt").read_text(encoding="utf-8") == "hola mundo"

    overwritten = await tool.execute(path="notas.txt", content="nuevo", overwrite=True)

    assert "sobrescrito" in overwritten


async def test_list_files(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / "carpeta").mkdir()
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")

    result = await ListFilesTool().execute()

    assert "[carpeta] carpeta" in result
    assert "[archivo] a.txt" in result


async def test_delete_path(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    victim = tmp_path / "borrar"
    victim.mkdir()
    (victim / "dentro.txt").write_text("x", encoding="utf-8")

    result = await DeletePathTool().execute(path="borrar")

    assert "Carpeta borrada" in result
    assert not victim.exists()


async def test_delete_refuses_home_itself(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    with pytest.raises(ValueError, match="carpeta personal"):
        await DeletePathTool().execute(path=".")
