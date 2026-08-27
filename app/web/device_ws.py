"""Canal de control del celular: SIA le habla, el celular obedece.

La app S.I.A abre un segundo WebSocket contra ``/ws/device`` y queda
registrada como ejecutora de comandos. Las herramientas de teléfono
(``app/tools/phone_tools.py``) piden acciones al hub y este se las envía
al celular conectado:

    celular   → {"type": "hello", "model": "Pixel 7", "battery": 87}
    servidor  → {"type": "welcome", "id": 1}
    servidor  → {"type": "cmd", "id": 10, "action": "call",
                 "params": {"number": "+52..."}}
    celular   → {"type": "result", "id": 10, "ok": true,
                 "message": "Llamando a +52...", "data": {}}

El celular también manda ``{"type": "ping"}`` cada tanto para mantener
viva la conexión; el servidor responde ``{"type": "pong"}``.
"""
import asyncio
import itertools
import logging

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("sia.device")

_TIMEOUT_COMANDO = 25.0


class DeviceError(RuntimeError):
    """No hay celular disponible o no respondió a tiempo."""


class CommandResult:
    """Respuesta del celular a un comando."""

    def __init__(self, ok: bool, message: str = "", data: dict | None = None):
        self.ok = ok
        self.message = message
        self.data = data or {}


class Device:
    """Un celular conectado."""

    _ids = itertools.count(1)

    def __init__(self, ws: WebSocket, model: str = ""):
        self.id = next(self._ids)
        self.ws = ws
        self.model = model or "celular"
        self.pending: dict[int, asyncio.Future] = {}
        self._cmd_ids = itertools.count(1)

    def description(self) -> str:
        return f"{self.model} (#{self.id})"


class DeviceHub:
    """Registro compartido de celulares conectados y despacho de comandos."""

    def __init__(self) -> None:
        self.devices: dict[int, Device] = {}

    # ---- gestión de conexiones -------------------------------------------
    @property
    def count(self) -> int:
        return len(self.devices)

    def descriptions(self) -> list[str]:
        return [d.description() for d in self.devices.values()]

    async def register(self, ws: WebSocket, model: str = "") -> Device:
        device = Device(ws, model)
        self.devices[device.id] = device
        logger.info("Celular conectado: %s (%d en total)", device.model, self.count)
        await ws.send_json({"type": "welcome", "id": device.id})
        return device

    def unregister(self, device: Device) -> None:
        self.devices.pop(device.id, None)
        for future in device.pending.values():
            if not future.done():
                future.set_result(CommandResult(False, "El celular se desconectó."))
        device.pending.clear()
        logger.info("Celular desconectado: %s (%d en total)", device.model, self.count)

    # ---- despacho de comandos --------------------------------------------
    def _pick(self, device_id: int | None = None) -> Device:
        if device_id is not None:
            device = self.devices.get(device_id)
            if device is None:
                raise DeviceError("Ese celular ya no está conectado.")
            return device
        if not self.devices:
            raise DeviceError(
                "No tengo ningún celular conectado. "
                "Abre la app S.I.A en tu teléfono para enlazarlo."
            )
        return next(iter(self.devices.values()))

    async def command(
        self,
        action: str,
        params: dict | None = None,
        *,
        timeout: float = _TIMEOUT_COMANDO,
        device_id: int | None = None,
    ) -> CommandResult:
        """Envía una acción al celular y espera su resultado."""
        device = self._pick(device_id)
        cmd_id = next(device._cmd_ids)
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        device.pending[cmd_id] = future
        try:
            await device.ws.send_json(
                {"type": "cmd", "id": cmd_id, "action": action, "params": params or {}}
            )
            return await asyncio.wait_for(future, timeout)
        except TimeoutError as exc:
            raise DeviceError(
                f"El celular no respondió al comando '{action}' ({timeout:.0f}s)."
            ) from exc
        finally:
            device.pending.pop(cmd_id, None)

    def resolve(self, cmd_id: int, result: CommandResult, device_id: int) -> None:
        """Entrega al futuro el resultado llegado desde el celular."""
        device = self.devices.get(device_id)
        if device is None:
            return
        future = device.pending.pop(cmd_id, None)
        if future is not None and not future.done():
            future.set_result(result)


_hub: DeviceHub | None = None


def get_hub() -> DeviceHub:
    """Hub único por proceso (las herramientas lo comparten)."""
    global _hub
    if _hub is None:
        _hub = DeviceHub()
    return _hub


def reset_hub() -> None:
    """Solo para tests."""
    global _hub
    _hub = None


class DeviceHubConnection:
    """Ciclo de vida de un celular dentro del canal /ws/device."""

    def __init__(self, hub: DeviceHub, ws: WebSocket) -> None:
        self._hub = hub
        self._ws = ws

    async def run(self) -> None:
        await self._ws.accept()
        device: Device | None = None
        try:
            while True:
                raw = await self._ws.receive_json()
                tipo = raw.get("type")
                if tipo == "hello":
                    device = await self._hub.register(
                        self._ws, str(raw.get("model") or "")
                    )
                    continue
                if device is None:
                    await self._ws.send_json(
                        {"type": "error", "message": "Falta hello."}
                    )
                    continue
                if tipo == "result":
                    self._hub.resolve(
                        int(raw.get("id", -1)),
                        CommandResult(
                            ok=bool(raw.get("ok")),
                            message=str(raw.get("message") or ""),
                            data=raw.get("data") or {},
                        ),
                        device.id,
                    )
                elif tipo == "ping":
                    await self._ws.send_json({"type": "pong"})
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            if device is not None:
                self._hub.unregister(device)
