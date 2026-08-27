"""Satélite de presencia ESP32: A.R.I.A sabe si el jefe está en casa.

Un ESP32 (solo la placa, sin hardware extra) escanea Bluetooth BLE buscando
el celular o reloj del jefe y reporta por WebSocket en /ws/presencia:

    esp32     → {"type": "presencia", "presente": true, "rssi": -62}
    esp32     → {"type": "presencia", "presente": false}
    servidor  → {"type": "ok", "presente": true}   (confirmación)

El estado global lo consulta la herramienta ``get_presence`` y cada LLEGADA
dispara el saludo de bienvenida por las interfaces de voz que estén
escuchando (nunca interrumpe: ``hablar_a_todas`` lo garantiza).

Histéresis anti-fantasma: los celulares apagan su BLE visible a ratos, así
que la salida solo se confirma tras VARIOS escaneos seguidos sin verlo —
esa lógica vive en el firmware; aquí se confía en lo que llega.
"""
import logging
import random
import time

from fastapi import WebSocket, WebSocketDisconnect

from app.voice.base import TTSProvider
from app.web.interface_ws import hablar_a_todas, hay_alguien_escuchando

logger = logging.getLogger("sia.presencia")

# Frases de bienvenida: primera palabra al llegar. Se sortean para no sonar
# grabadas; el trato es de jefe, como manda la personalidad.
_FRASES_DIA = (
    "Buenos días, jefe. Ya sé que estás en casa.",
    "Buenos días, jefe. Aquí estoy si me necesitas.",
)
_FRASES_TARDE = (
    "Bienvenido a casa, jefe.",
    "Ya te siento, jefe. ¿Cómo te fue?",
    "Qué bueno verte por aquí, jefe.",
)
_FRASES_NOCHE = (
    "Buenas noches, jefe. Ya estás en casa.",
    "Llegaste, jefe. Yo me encargo de lo demás.",
)


def frase_de_bienvenida(reloj=time.localtime) -> str:
    """Frase corta de llegada según la hora del día."""
    hora = reloj().tm_hour
    if 5 <= hora < 12:
        opciones = _FRASES_DIA
    elif 12 <= hora < 20:
        opciones = _FRASES_TARDE
    else:
        opciones = _FRASES_NOCHE
    return random.choice(opciones)


class EstadoPresencia:
    """Último estado conocido del jefe (único por proceso)."""

    def __init__(self) -> None:
        self.presente: bool | None = None  # None = sin datos todavía
        self.mono_cambio: float | None = None  # time.monotonic() del cambio
        self.mono_avistamiento: float | None = None  # última vez visto en casa
        self.rssi: int | None = None  # potencia de la señal BLE (cercanía)

    def marcar(self, presente: bool, *, rssi: int | None = None) -> str | None:
        """Actualiza el estado y devuelve 'llegada' / 'salida' si hubo transición."""
        ahora = time.monotonic()
        antes = self.presente
        self.presente = presente
        self.mono_cambio = ahora
        if rssi is not None:
            self.rssi = rssi
        if presente:
            self.mono_avistamiento = ahora
            return "llegada" if antes is not True else None
        return "salida" if antes is True else None

    def visto_hace_segundos(self) -> int | None:
        if self.mono_avistamiento is None:
            return None
        return round(time.monotonic() - self.mono_avistamiento)

    def resumen(self) -> dict:
        """Datos crudos para logs y para la herramienta get_presence."""
        return {
            "en_casa": self.presente,
            "visto_hace_s": self.visto_hace_segundos(),
            "rssi": self.rssi,
        }

    def reiniciar(self) -> None:
        self.__init__()


_estado = EstadoPresencia()


def get_estado_presencia() -> EstadoPresencia:
    """Estado compartido del satélite de presencia (patrón hub único)."""
    return _estado


class PresenciaConnection:
    """Una conexión de satélite de presencia (típicamente una sola)."""

    def __init__(self, ws: WebSocket, *, tts: TTSProvider | None = None) -> None:
        self._ws = ws
        self._tts = tts
        self.estado = get_estado_presencia()

    async def run(self) -> None:
        await self._ws.accept()
        logger.info("Satélite de presencia conectado")
        try:
            while True:
                datos = await self._ws.receive_json()
                if datos.get("type") != "presencia":
                    continue
                presente = bool(datos.get("presente"))
                rssi = datos.get("rssi")
                transicion = self.estado.marcar(
                    presente,
                    rssi=int(rssi) if isinstance(rssi, (int, float)) else None,
                )
                logger.info(
                    "Presencia: %s (%s)", "EN CASA" if presente else "FUERA", transicion
                )
                if transicion == "llegada":
                    await self._saludar_llegada()
                await self._ws.send_json(
                    {"type": "ok", "presente": self.estado.presente}
                )
        except WebSocketDisconnect:
            pass
        finally:
            logger.info("Satélite de presencia desconectado")

    async def _saludar_llegada(self) -> None:
        """Saluda al jefe cuando llega… solo si alguien está escuchando."""
        if not hay_alguien_escuchando():
            logger.debug("Llegada detectada pero nadie escuchaba; sin saludo")
            return
        try:
            from app.voice.tts import build_tts_provider

            tts = self._tts or build_tts_provider()
            await hablar_a_todas(frase_de_bienvenida(), tts)
        except Exception as exc:  # noqa: BLE001 — saludar nunca rompe el canal
            logger.warning("No pude saludar la llegada: %s", exc)
