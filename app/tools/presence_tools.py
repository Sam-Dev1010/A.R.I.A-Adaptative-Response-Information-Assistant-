"""Herramienta de presencia: A.R.I.A sabe dónde está su jefe.

Consume el estado que publica el satélite ESP32 en /ws/presencia
(ver ``app/web/presencia_ws.py``). Sin satélite conectado responde con
honestidad en vez de inventar.
"""
import json

from app.tools.base import BaseTool, ToolPermission
from app.web.presencia_ws import get_estado_presencia


class GetPresenceTool(BaseTool):
    """¿Está el jefe en casa? Lo dice el satélite de presencia ESP32."""

    name = "get_presence"
    description = (
        "Informa si tu creador/jefe está físicamente en casa según el "
        "satélite de presencia (ESP32 con Bluetooth). Devuelve cuánto hace "
        "que fue visto por última vez y la cercanía del dispositivo."
    )
    permission = ToolPermission.SAFE

    async def _run(self, **kwargs) -> str:
        estado = get_estado_presencia()
        datos = estado.resumen()
        if datos["en_casa"] is None:
            return (
                "No hay satélite de presencia conectado: no sé si el jefe "
                "está en casa."
            )
        if not datos["en_casa"]:
            detalle = "El jefe está FUERA de casa."
            visto = datos["visto_hace_s"]
            if isinstance(visto, int):
                detalle += f" Lo vi por última vez antes de salir (hace {visto} s)."
        else:
            visto = datos["visto_hace_s"]
            hace = "ahora mismo" if visto == 0 else f"hace {visto} s"
            detalle = f"El jefe está EN CASA (detectado {hace})."
        return json.dumps({"detalle": detalle, **datos}, ensure_ascii=False)
