#!/usr/bin/env bash
# Instala A.R.I.A como aplicación de escritorio buscable en el menú
# (equivalente a Firefox en el lanzador de aplicaciones).
#
# Instala:
#   - ~/.local/share/applications/aria.desktop   -> entrada del menú
#   - ~/.local/share/icons/.../aria.*            -> iconos
#   - ~/.local/bin/aria                          -> comando `aria` en terminal
#
# Uso:
#   scripts/install_desktop.sh [--system]
set -euo pipefail

SCRIPT_SRC="$(readlink -f "${BASH_SOURCE[0]}")"
PROJECT_DIR="$(dirname "$SCRIPT_SRC")"   # scripts/
PROJECT_DIR="$(dirname "$PROJECT_DIR")"  # raíz del proyecto

MODE="${1:-user}"

if [[ "$MODE" == "--system" ]]; then
    APPS_DIR="/usr/share/applications"
    ICONS_DIR="/usr/share/icons/hicolor"
    BIN_DIR="/usr/local/bin"
    LAUNCHER_DEST="/usr/local/bin/aria"
    if [[ $EUID -ne 0 ]]; then
        echo "Para --system necesitas ser root. Usa: sudo $0 --system" >&2
        exit 1
    fi
else
    APPS_DIR="$HOME/.local/share/applications"
    ICONS_DIR="$HOME/.local/share/icons/hicolor"
    BIN_DIR="$HOME/.local/bin"
    LAUNCHER_DEST="$HOME/.local/bin/aria"
fi

mkdir -p "$APPS_DIR" "$BIN_DIR"

# ---- Iconos: escalable + PNG en varios tamaños ----
install_icon() {
    local name="$1" file="$2"
    local dest="$ICONS_DIR/$3/apps"
    mkdir -p "$dest"
    install -m 0644 "$file" "$dest/$name"
}
# El nombre de icono en el .desktop es "aria"; el buscador añade la extensión
# según el directorio (scalable/*.svg, 256x256/*.png…). Conservamos la extensión.
install_icon aria.svg "$PROJECT_DIR/app/desktop/icons/aria.svg" "scalable"
install_icon aria.png "$PROJECT_DIR/app/desktop/icons/aria-512.png" "512x512"
install_icon aria.png "$PROJECT_DIR/app/desktop/icons/aria-256.png" "256x256"

# ---- Entrada del menú ----
cat > "$APPS_DIR/aria.desktop" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=A.R.I.A
GenericName=Asistente personal
GenericName[es]=Asistente personal
Comment=Asistente personal con inteligencia neural y voz
Comment[es]=Asistente personal con inteligencia neural y voz
Exec=$LAUNCHER_DEST
Icon=aria
Terminal=false
Categories=Utility;Accessibility;AudioVideo;
Keywords=asistente;voz;neural;aria;inteligencia;
StartupNotify=true
EOF

# ---- Comando `aria` en la terminal ----
install -m 0755 "$PROJECT_DIR/scripts/aria-launcher.sh" "$LAUNCHER_DEST"

# ---- Refresh del menú ----
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$APPS_DIR" >/dev/null 2>&1 || true
fi
if command -v gtk-update-icon-cache >/dev/null 2>&1; then
    gtk-update-icon-cache -f -t "$ICONS_DIR" >/dev/null 2>&1 || true
fi

echo "✔ A.R.I.A instalada como aplicación de escritorio."
echo "  Entrada: $APPS_DIR/aria.desktop"
echo "  Comando: $LAUNCHER_DEST"
echo "Búscala en el lanzador de aplicaciones con «Aria»."
