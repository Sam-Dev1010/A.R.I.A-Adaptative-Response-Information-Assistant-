"""Sirve SIA por HTTPS en la red local para usarlo desde el CELULAR.

Los navegadores solo permiten el micrófono en páginas HTTPS (o localhost):
este script genera un certificado autofirmado, detecta la IP local de la PC
y levanta el servidor con TLS.

    python scripts/sia_movil.py

Después, con el celular en la MISMA red WiFi:
    1. Abre https://<IP-DE-LA-PC>:8000
    2. Acepta el aviso de certificado autofirmado (una sola vez)
    3. Opcional: menú del navegador → "Añadir a pantalla de inicio"
"""
import argparse
import datetime
import ipaddress
import shutil
import socket
import subprocess
from pathlib import Path

from app.core.config import get_settings

_SSL_DIR = Path("data/ssl")


def ip_local() -> str:
    """IP de esta PC en la red local (no envía tráfico real)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]


def _cert_con_cryptography(ruta_cert: Path, ruta_key: Path, ip: str) -> bool:
    """Certificado autofirmado con cryptography (si está instalada)."""
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        return False

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    nombre = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "SIA")])
    ahora = datetime.datetime.now(datetime.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(nombre)
        .issuer_name(nombre)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(ahora - datetime.timedelta(days=1))
        .not_valid_after(ahora + datetime.timedelta(days=3650))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.IPAddress(ipaddress.ip_address(ip)), x509.DNSName("localhost")]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    ruta_key.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    ruta_cert.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return True


def _cert_con_openssl(ruta_cert: Path, ruta_key: Path, ip: str) -> bool:
    """Certificado autofirmado con el binario openssl del sistema."""
    openssl = shutil.which("openssl")
    if not openssl:
        return False
    resultado = subprocess.run(
        [
            openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-keyout", str(ruta_key), "-out", str(ruta_cert),
            "-days", "3650",
            "-subj", "/CN=SIA",
            "-addext", f"subjectAltName=IP:{ip},DNS:localhost",
        ],
        capture_output=True,
        check=False,
    )
    return resultado.returncode == 0


def generar_certificado(ruta_cert: Path, ruta_key: Path, ip: str) -> None:
    """Crea el par cert/llave si aún no existen."""
    if ruta_cert.exists() and ruta_key.exists():
        return
    ruta_cert.parent.mkdir(parents=True, exist_ok=True)
    if _cert_con_cryptography(ruta_cert, ruta_key, ip):
        motor = "cryptography"
    elif _cert_con_openssl(ruta_cert, ruta_key, ip):
        motor = "openssl"
    else:
        raise SystemExit(
            "No pude generar el certificado TLS. Instala una de estas opciones:\n"
            "  pip install cryptography\n"
            "  sudo dnf install openssl   (Fedora) / sudo apt install openssl"
        )
    print(f"Certificado autofirmado creado ({motor}): {ruta_cert}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SIA por HTTPS para tu celular")
    parser.add_argument("--port", type=int, default=None, help="Puerto (por defecto PORT de .env)")
    parser.add_argument(
        "--regenerar-cert", action="store_true",
        help="Fuerza un certificado nuevo (útil si cambió la IP)",
    )
    args = parser.parse_args()

    settings = get_settings()
    puerto = args.port or settings.port
    ip = ip_local()

    ruta_cert = _SSL_DIR / "sia.crt"
    ruta_key = _SSL_DIR / "sia.key"
    if args.regenerar_cert:
        ruta_cert.unlink(missing_ok=True)
        ruta_key.unlink(missing_ok=True)
    generar_certificado(ruta_cert, ruta_key, ip)

    import uvicorn

    print()
    print("═" * 52)
    print("  SIA en tu celular")
    print("═" * 52)
    print("  1. Conecta el celular al MISMO WiFi que esta PC")
    print(f"  2. Abre:  https://{ip}:{puerto}")
    print("  3. Acepta el aviso de seguridad (autofirmado)")
    print("  4. Menú del navegador → Añadir a pantalla de inicio")
    print("═" * 52)
    if shutil.which("firewall-cmd"):
        print("  Si no carga, abre el puerto en el firewall:")
        print("    sudo firewall-cmd --add-port=8000/tcp --permanent")
        print("    sudo firewall-cmd --reload")
    print()

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=puerto,
        ssl_certfile=str(ruta_cert),
        ssl_keyfile=str(ruta_key),
    )


if __name__ == "__main__":
    main()
