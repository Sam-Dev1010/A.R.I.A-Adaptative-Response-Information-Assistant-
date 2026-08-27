"""Tests del sincronizador interfaz web → app móvil."""
import importlib.util
from pathlib import Path

_RUTA = Path(__file__).parent.parent / "scripts" / "sincronizar_movil.py"


def _cargar():
    spec = importlib.util.spec_from_file_location("sincronizar_movil", _RUTA)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_sincronizar_genera_index_apto_para_la_app(tmp_path, monkeypatch):
    sia_sync = _cargar()
    destino = tmp_path / "index.html"
    monkeypatch.setattr(sia_sync, "DESTINO", destino)

    resultado = sia_sync.sincronizar()

    assert resultado == destino
    html = destino.read_text(encoding="utf-8")
    # El WebSocket apunta al servidor configurado en el celular…
    linea = next(l for l in html.splitlines() if "new WebSocket" in l)
    esperado = 'ws = new WebSocket(Servidor.websocket() + (token ? "?token=" + token : ""));'
    assert linea.strip() == esperado
    # …movil.js está inyectado y las líneas de navegador se quitaron.
    assert '<script src="movil.js"></script>' in html
    assert 'rel="manifest"' not in html


def test_sincronizar_falla_si_cambia_interface(tmp_path, monkeypatch):
    sia_sync = _cargar()
    origen = tmp_path / "interface.html"
    origen.write_text("<html><head></head></html>", encoding="utf-8")
    monkeypatch.setattr(sia_sync, "ORIGEN", origen)
    monkeypatch.setattr(sia_sync, "DESTINO", tmp_path / "index.html")

    try:
        sia_sync.sincronizar()
    except SystemExit as exc:
        assert "No encontré" in str(exc)
    else:
        raise AssertionError("debía fallar por falta del punto de inyección")
