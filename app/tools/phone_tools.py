"""Herramientas de teléfono: A.R.I.A controla el celular del usuario.

Los comandos viajan por el canal ``/ws/device`` (ver ``app/web/device_ws.py``)
hasta la app S.I.A, que los ejecuta de forma nativa: llamadas, WhatsApp,
correos, abrir aplicaciones y buscar contactos.

Si no hay ningún celular enlazado, responden con un mensaje claro en vez
de fallar.
"""
from typing import ClassVar

from app.tools.base import BaseTool, ToolPermission
from app.web.device_ws import DeviceError, DeviceHub


class _PhoneTool(BaseTool):
    """Base común: manda un comando al hub y traduce el resultado a texto."""

    action = ""

    def __init__(self, hub: DeviceHub | None = None) -> None:
        if hub is None:
            from app.web.device_ws import get_hub

            hub = get_hub()
        self._hub = hub

    async def _enviar(self, params: dict) -> str:
        try:
            result = await self._hub.command(self.action, params)
        except DeviceError as exc:
            return str(exc)
        if result.ok:
            return result.message or "Listo."
        return (
            f"El celular no pudo hacerlo: {result.message or 'error desconocido'}."
        )


class PhoneCallTool(_PhoneTool):
    """Llama por teléfono desde el celular del usuario."""

    name = "phone_call"
    action = "call"
    description = (
        "Llama desde el celular del usuario: número directo o nombre de contacto."
    )
    permission = ToolPermission.CONFIRM
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "number": {"type": "string", "description": "Número de teléfono."},
            "contact": {
                "type": "string",
                "description": "Nombre del contacto (se busca su número; ignora 'number').",
            },
        },
        "required": [],
    }

    async def _run(self, number: str = "", contact: str = "", **kwargs) -> str:
        if contact.strip():
            found = await _buscar_contacto(self._hub, contact)
            if not found:
                return f"No encontré a '{contact}' en la agenda del celular."
            number = found[0][1]
        if not str(number).strip():
            return "Dime el número o el contacto al que quieres llamar."
        return await self._enviar({"number": str(number).strip()})


class WhatsAppTool(_PhoneTool):
    """Abre un chat de WhatsApp con destinatario y mensaje listos."""

    name = "whatsapp_message"
    action = "whatsapp"
    description = (
        "Prepara un WhatsApp desde el celular: abre el chat del destinatario "
        "(número o contacto) con el texto escrito; el usuario toca enviar."
    )
    permission = ToolPermission.CONFIRM
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "Texto del mensaje."},
            "phone": {"type": "string", "description": "Número destinatario."},
            "contact": {
                "type": "string",
                "description": "Nombre del contacto (ignora 'phone').",
            },
        },
        "required": ["message"],
    }

    async def _run(
        self, message: str, phone: str = "", contact: str = "", **kwargs
    ) -> str:
        if not message.strip():
            return "Dime qué mensaje quieres enviar."
        if contact.strip():
            found = await _buscar_contacto(self._hub, contact)
            if not found:
                return f"No encontré a '{contact}' en la agenda del celular."
            phone = found[0][1]
        if not str(phone).strip():
            return await self._enviar({"text": message})
        return await self._enviar({"number": str(phone).strip(), "text": message})


class SendEmailTool(_PhoneTool):
    """Envía un correo desde el celular del usuario."""

    name = "send_email"
    action = "email"
    description = (
        "Abre el correo del celular con un email listo (destinatario, asunto "
        "y cuerpo); el usuario toca enviar."
    )
    permission = ToolPermission.CONFIRM
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "Correo del destinatario."},
            "subject": {"type": "string", "description": "Asunto."},
            "body": {"type": "string", "description": "Cuerpo del correo."},
        },
        "required": ["to"],
    }

    async def _run(self, to: str, subject: str = "", body: str = "", **kwargs) -> str:
        if not to.strip() or "@" not in to:
            return "Dame una dirección de correo válida."
        return await self._enviar(
            {"to": to.strip(), "subject": subject, "body": body}
        )


class OpenPhoneAppTool(_PhoneTool):
    """Abre cualquier aplicación instalada en el celular."""

    name = "open_phone_app"
    action = "open_app"
    description = "Abre una app del celular por nombre ('spotify', 'cámara'…)."
    permission = ToolPermission.CONFIRM
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "app": {"type": "string", "description": "Nombre de la app."}
        },
        "required": ["app"],
    }

    async def _run(self, app: str, **kwargs) -> str:
        if not app.strip():
            return "Dime qué aplicación quieres abrir."
        return await self._enviar({"name": app.strip()})


class PhoneContactsTool(BaseTool):
    """Busca contactos en la agenda del celular (solo lectura)."""

    name = "phone_contacts"
    description = (
        "Busca contactos en la agenda del celular por nombre y devuelve sus números."
    )
    permission = ToolPermission.SAFE
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Nombre (o parte). Vacío = los primeros.",
            }
        },
        "required": [],
    }

    def __init__(self, hub: DeviceHub | None = None) -> None:
        if hub is None:
            from app.web.device_ws import get_hub

            hub = get_hub()
        self._hub = hub

    async def _run(self, query: str = "", **kwargs) -> str:
        found = await _buscar_contacto(self._hub, query)
        if not found:
            return f"No encontré contactos que coincidan con '{query}'."
        lines = [f"{nombre}: {numero}" for nombre, numero in found]
        return "Contactos encontrados:\n" + "\n".join(lines)


class PhoneStatusTool(_PhoneTool):
    """Batería y estado de carga del celular."""

    name = "phone_status"
    action = "status"
    description = (
        "Consulta el celular: nivel de batería y si está cargando."
    )
    permission = ToolPermission.SAFE
    parameters: ClassVar[dict] = {"type": "object", "properties": {}}

    async def _run(self, **kwargs) -> str:
        return await self._enviar({})


class PhoneTorchTool(_PhoneTool):
    """Enciende o apaga la linterna del celular."""

    name = "phone_torch"
    action = "torch"
    description = (
        "Enciende ('on') o apaga ('off') la linterna del celular."
    )
    permission = ToolPermission.SAFE
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {"mode": {"type": "string", "enum": ["on", "off"]}},
        "required": [],
    }

    async def _run(self, mode: str = "on", **kwargs) -> str:
        return await self._enviar({"mode": mode})


class PhoneVibrateTool(_PhoneTool):
    """Hace vibrar el celular un instante."""

    name = "phone_vibrate"
    action = "vibrate"
    description = (
        "Hace vibrar el celular; útil para encontrarlo o llamar la atención."
    )
    permission = ToolPermission.SAFE
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {"ms": {"type": "integer", "description": "Duración en ms (50-5000)."}},
        "required": [],
    }

    async def _run(self, ms: int = 400, **kwargs) -> str:
        return await self._enviar({"ms": int(ms)})


class PhoneClipboardTool(_PhoneTool):
    """Copia texto al portapapeles del celular."""

    name = "phone_clipboard"
    action = "clipboard"
    description = "Copia un texto al portapapeles del celular."
    permission = ToolPermission.SAFE
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "Texto a copiar."}},
        "required": ["text"],
    }

    async def _run(self, text: str = "", **kwargs) -> str:
        if not text.strip():
            return "Dime qué texto copiar al portapapeles."
        return await self._enviar({"text": text})


class SendSmsTool(_PhoneTool):
    """Prepara un SMS en el celular (el usuario toca enviar)."""

    name = "send_sms"
    action = "sms"
    description = (
        "Prepara un SMS desde el celular: abre la app de mensajes con "
        "destinatario y texto listos; el usuario toca enviar."
    )
    permission = ToolPermission.CONFIRM
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Texto del mensaje."},
            "phone": {"type": "string", "description": "Número destinatario."},
            "contact": {
                "type": "string",
                "description": "Nombre del contacto (ignora 'phone').",
            },
        },
        "required": ["text"],
    }

    async def _run(self, text: str, phone: str = "", contact: str = "", **kwargs) -> str:
        if not text.strip():
            return "Dime qué mensaje quieres enviar por SMS."
        if contact.strip():
            found = await _buscar_contacto(self._hub, contact)
            if not found:
                return f"No encontré a '{contact}' en la agenda del celular."
            phone = found[0][1]
        if not str(phone).strip():
            return "Dime el número o contacto para el SMS."
        return await self._enviar({"number": str(phone).strip(), "text": text})


class SetAlarmTool(_PhoneTool):
    """Programa una alarma o temporizador en el reloj del celular."""

    name = "set_alarm"
    action = "alarm"
    description = (
        "Pone una alarma en el reloj del celular (hora y minuto), o un "
        "temporizador con 'minutes' si no indicas hora. El usuario confirma."
    )
    permission = ToolPermission.CONFIRM
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "hour": {"type": "integer", "description": "Hora 0-23 (alarma)."},
            "minute": {"type": "integer", "description": "Minuto 0-59 (o duración en minutos si es temporizador)."},
            "label": {"type": "string", "description": "Etiqueta de la alarma."},
            "timer": {"type": "boolean", "description": "True = temporizador en vez de alarma."},
        },
        "required": [],
    }

    async def _run(
        self,
        hour: int | None = None,
        minute: int | None = None,
        label: str = "",
        timer: bool = False,
        **kwargs,
    ) -> str:
        if not timer and (hour is None or hour < 0):
            return "Dime la hora de la alarma, o pide un temporizador con 'timer'."
        params: dict = {"timer": timer}
        if hour is not None:
            params["hour"] = int(hour)
        if minute is not None:
            params["minute"] = int(minute)
        if label.strip():
            params["label"] = label.strip()
        return await self._enviar(params)


class NavigateTool(_PhoneTool):
    """Abre la navegación de Maps hacia un destino en el celular."""

    name = "navigate"
    action = "navigate"
    description = (
        "Inicia la navegación en el celular hacia una dirección o lugar "
        "(abre Maps con la ruta)."
    )
    permission = ToolPermission.CONFIRM
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "destination": {"type": "string", "description": "Dirección o lugar destino."}
        },
        "required": ["destination"],
    }

    async def _run(self, destination: str = "", **kwargs) -> str:
        if not destination.strip():
            return "Dime a dónde quieres navegar."
        return await self._enviar({"destination": destination.strip()})


class PhoneVolumeTool(_PhoneTool):
    """Sube, baja o consulta el volumen multimedia del celular."""

    name = "phone_volume"
    action = "volume"
    description = (
        "Volumen multimedia del celular: número 0-100 para fijarlo, "
        "sin argumento para consultarlo."
    )
    permission = ToolPermission.SAFE
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {"level": {"type": "integer", "description": "0-100."}},
        "required": [],
    }

    async def _run(self, level: int | None = None, **kwargs) -> str:
        params: dict = {}
        if level is not None:
            params["level"] = max(0, min(100, int(level)))
        return await self._enviar(params)


class NotifyPhoneTool(_PhoneTool):
    """Manda una notificación local al celular del usuario."""

    name = "notify_phone"
    action = "notify"
    description = (
        "Envía una notificación al celular del usuario (título y mensaje); "
        "útil para avisarle cosas importantes cuando está lejos de la PC."
    )
    permission = ToolPermission.SAFE
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Título corto."},
            "message": {"type": "string", "description": "Texto del aviso."},
        },
        "required": ["message"],
    }

    async def _run(self, message: str = "", title: str = "A.R.I.A", **kwargs) -> str:
        if not message.strip():
            return "Dime qué avisar en la notificación."
        return await self._enviar({"title": title or "A.R.I.A", "message": message})


async def _buscar_contacto(hub: DeviceHub, query: str) -> list[tuple[str, str]]:
    """Pide la agenda al celular; devuelve [(nombre, número), …]."""
    try:
        result = await hub.command("contacts", {"query": query})
    except DeviceError:
        return []
    if not result.ok:
        return []
    items = result.data.get("contacts") or []
    return [
        (str(item.get("name", "")), str(item.get("number", "")))
        for item in items
        if item.get("number")
    ]
