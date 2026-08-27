#!/usr/bin/env bash
# Lanzador de A.R.I.A como app de escritorio (para el menú de aplicaciones).
#
# Resuelve el directorio del proyecto, entra en él para que los imports
# (app.main) y la carpeta data/ queden bien, y arranca sia_app.py.
set -euo pipefail

# Ruta real del proyecto (resuelve enlaces simbólicos).
SCRIPT_SRC="$(readlink -f "${BASH_SOURCE[0]}")"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_SRC")")"

cd "$PROJECT_DIR"

# Preferir el python del venv si existe; si no, el del sistema.
if [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python"
else
    PYTHON="$(command -v python3)"
fi

exec "$PYTHON" "$PROJECT_DIR/scripts/sia_app.py" "$@"
