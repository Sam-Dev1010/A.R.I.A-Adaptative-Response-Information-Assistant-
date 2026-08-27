"""Tests del lanzador móvil: IP local y certificado TLS autofirmado."""
import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

_RUTA = Path(__file__).parent.parent / "scripts" / "sia_movil.py"


def _cargar():
    """Importa scripts/sia_movil.py como módulo (no es paquete)."""
    spec = importlib.util.spec_from_file_location("sia_movil", _RUTA)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_ip_local_devuelve_ipv4():
    sia_movil = _cargar()

    ip = sia_movil.ip_local()

    assert ip.count(".") == 3
    assert not ip.startswith("127.")


def test_generar_certificado_reusa_los_existentes(tmp_path):
    sia_movil = _cargar()
    cert = tmp_path / "sia.crt"
    key = tmp_path / "sia.key"
    cert.write_text("EXISTENTE")
    key.write_text("EXISTENTE")

    sia_movil.generar_certificado(cert, key, "192.168.1.50")

    assert cert.read_text() == "EXISTENTE"


@pytest.mark.skipif(shutil.which("openssl") is None, reason="requiere openssl")
def test_generar_certificado_con_openssl(tmp_path):
    sia_movil = _cargar()
    cert = tmp_path / "sia.crt"
    key = tmp_path / "sia.key"

    creado = sia_movil._cert_con_openssl(cert, key, "192.168.1.50")

    assert creado
    assert "BEGIN CERTIFICATE" in cert.read_text()
    assert "PRIVATE KEY" in key.read_text()
    detalle = subprocess.run(
        ["openssl", "x509", "-in", str(cert), "-noout", "-ext", "subjectAltName"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "192.168.1.50" in detalle.stdout
