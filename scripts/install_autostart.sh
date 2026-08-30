#!/usr/bin/env bash
# Instala A.R.I.A como servicio de usuario de systemd: arranca sola al
# encender el PC, saluda por voz y queda escuchando.
#
# Uso:
#   bash scripts/install_autostart.sh [--no-greeting] [--uninstall]
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$PROJECT_DIR/.venv/bin/python"
SERVICE_NAME="aria-voice.service"
SERVICE_DIR="$HOME/.config/systemd/user"
SERVICE_FILE="$SERVICE_DIR/$SERVICE_NAME"

UNINSTALL=0
GREETING=1
for arg in "$@"; do
  case "$arg" in
    --no-greeting) GREETING=0 ;;
    --uninstall)   UNINSTALL=1 ;;
  esac
done

if [[ "$UNINSTALL" == "1" ]]; then
  systemctl --user disable --now "$SERVICE_NAME" 2>/dev/null || true
  rm -f "$SERVICE_FILE"
  echo "Servicio A.R.I.A desinstalado, habilitado y detenido."
  exit 0
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: no encuentro el venv en $PYTHON (crea .venv e instala deps primero)." >&2
  exit 1
fi

grep -q EdgeTTSProvider "$PROJECT_DIR/app/voice/tts.py" || true

BOOT_FLAGS=""
if [[ "$GREETING" == "1" ]]; then
  BOOT_FLAGS="--boot-greeting"
fi

mkdir -p "$SERVICE_DIR"

cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=A.R.I.A asistente personal por voz (autostart)
After=graphical-session.target sound.target

[Service]
Type=simple
ExecStart=$PYTHON $PROJECT_DIR/scripts/autostart_aria.py $BOOT_FLAGS
WorkingDirectory=$PROJECT_DIR
Environment=DISPLAY=:0
Environment=XDG_RUNTIME_DIR=%t
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable "$SERVICE_NAME"
systemctl --user start "$SERVICE_NAME"

echo "A.R.I.A configurada para arrancar sola al encender el PC."
echo "  Servicio: $SERVICE_FILE"
systemctl --user status "$SERVICE_NAME" --no-pager || true
