"""Sincroniza la interfaz web de SIA dentro de la app móvil (Capacitor).

Toma ``app/web/static/interface.html`` y genera ``mobile/www/index.html``
con dos ajustes mínimos:

  1. La dirección del WebSocket ya no es ``location.host`` sino el servidor
     configurado en el celular (``Servidor.websocket()``, de movil.js).
  2. Se inyecta ``<script src="movil.js">`` en el ``</head>``.

Además pre-carga el ACCESS_TOKEN del ``.env`` del servidor en
``mobile/www/movil.js`` (marcador ``__TOKEN_ACCESO__``) para que la app
conecte sin pedir datos.

Así, cualquier mejora futura de la interfaz llega a la app con un solo
comando:  python scripts/sincronizar_movil.py

Uso:  python scripts/sincronizar_movil.py [--verificar]
"""
import argparse
import re
from pathlib import Path

RAIZ = Path(__file__).parent.parent
ORIGEN = RAIZ / "app" / "web" / "static" / "interface.html"
DESTINO = RAIZ / "mobile" / "www" / "index.html"
MOVIL = RAIZ / "mobile" / "www" / "movil.js"
ENV = RAIZ / ".env"

_MARCADOR_TOKEN = "__TOKEN_ACCESO__"

_WS_WEB = '(location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws/interface"'
_WS_APP = "Servidor.websocket()"
_INYECCION = '<script src="movil.js"></script>\n</head>'
_LINEAS_APP = ('rel="manifest"', 'rel="icon"')  # solo útiles en el navegador


def token_de_env() -> str:
    """ACCESS_TOKEN del .env (cadena vacía si no está definido)."""
    if not ENV.exists():
        return ""
    for linea in ENV.read_text(encoding="utf-8").splitlines():
        coincide = re.match(r'^ACCESS_TOKEN\s*=\s*"?([^"\s#]+)"?', linea)
        if coincide:
            return coincide.group(1)
    return ""


def inyectar_token() -> None:
    """Deja el token real dentro de movil.js (solo en la copia empaquetada)."""
    js = MOVIL.read_text(encoding="utf-8")
    MOVIL.write_text(
        js.replace(_MARCADOR_TOKEN, token_de_env()), encoding="utf-8"
    )


def sincronizar() -> Path:
    html = ORIGEN.read_text(encoding="utf-8")
    if _WS_WEB not in html:
        raise SystemExit(
            "No encontré el punto de conexión del WebSocket en interface.html; "
            "revisa si cambió su formato."
        )
    html = html.replace(_WS_WEB, _WS_APP)
    lineas = [
        linea
        for linea in html.splitlines()
        if not any(marca in linea for marca in _LINEAS_APP)
    ]
    html = "\n".join(lineas) + "\n"
    html = html.replace("</head>", f"{_INYECCION}", 1)
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    DESTINO.write_text(html, encoding="utf-8")
    inyectar_token()
    return DESTINO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verificar",
        action="store_true",
        help="solo comprueba que los puntos de inyección existen",
    )
    args = parser.parse_args()
    html = ORIGEN.read_text(encoding="utf-8")
    ok = _WS_WEB in html and "</head>" in html
    if args.verificar:
        raise SystemExit(0 if ok else 1)
    if not ok:
        raise SystemExit("interface.html cambió de formato; revisa sincronizar_movil.py")
    print(f"Generado: {sincronizar()}")


if __name__ == "__main__":
    main()
