"""Tests del canal de dispositivo (/ws/device) y de las herramientas de teléfono."""
import asyncio

import pytest
from fastapi.testclient import TestClient

from app.ai.orchestrator import AssistantOrchestrator
from app.ai.providers.base import LLMProvider
from app.ai.schemas import ChatResponse, TokenUsage, ToolCall
from app.main import create_app
from app.tools.base import ToolPermission
from app.tools.phone_tools import (
    NavigateTool,
    NotifyPhoneTool,
    OpenPhoneAppTool,
    PhoneCallTool,
    PhoneClipboardTool,
    PhoneContactsTool,
    PhoneStatusTool,
    PhoneTorchTool,
    PhoneVibrateTool,
    PhoneVolumeTool,
    SendEmailTool,
    SendSmsTool,
    SetAlarmTool,
    WhatsAppTool,
)
from app.tools.registry import ToolRegistry
from app.web.device_ws import CommandResult, DeviceError, DeviceHub, reset_hub


@pytest.fixture(autouse=True)
def _hub_limpio():
    reset_hub()
    yield
    reset_hub()


class FakeWS:
    """WebSocket falso que graba lo enviado."""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_json(self, data) -> None:
        self.sent.append(data)


class CapturingProvider(LLMProvider):
    """Devuelve un guion fijo y recuerda los mensajes recibidos."""

    name = "capturing"
    model = "capturing-model"

    def __init__(self, script) -> None:
        self._script = list(script)
        self.calls: list[list] = []

    async def chat(self, messages, tools=None) -> ChatResponse:
        self.calls.append(list(messages))
        content, tool_calls = self._script.pop(0)
        return ChatResponse(
            content=content,
            model=self.model,
            provider=self.name,
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1),
            tool_calls=tool_calls,
        )


# ---------------------------------------------------------------------------
# DeviceHub (unidad)
# ---------------------------------------------------------------------------


async def test_hub_comando_ida_y_vuelta():
    hub = DeviceHub()
    ws = FakeWS()
    device = await hub.register(ws, "Pixel 7")
    assert hub.count == 1
    assert device.description() == "Pixel 7 (#1)"

    async def responder():
        await asyncio.sleep(0)
        cmd = ws.sent[-1]
        assert cmd["type"] == "cmd"
        assert cmd["action"] == "call"
        hub.resolve(cmd["id"], CommandResult(True, "Llamando a 123."), device.id)

    tarea = asyncio.create_task(responder())
    resultado = await hub.command("call", {"number": "123"})
    await tarea
    assert resultado.ok
    assert resultado.message == "Llamando a 123."


async def test_hub_sin_celular_da_mensaje_claro():
    hub = DeviceHub()
    with pytest.raises(DeviceError) as excinfo:
        await hub.command("call", {"number": "123"})
    assert "ningún celular" in str(excinfo.value)


async def test_hub_timeout_devuelve_error():
    hub = DeviceHub()
    await hub.register(FakeWS(), "Pixel")
    with pytest.raises(DeviceError) as excinfo:
        await hub.command("call", {"number": "123"}, timeout=0.01)
    assert "no respondió" in str(excinfo.value)


async def test_unregister_resuelve_pendientes():
    hub = DeviceHub()
    device = await hub.register(FakeWS(), "Pixel")
    tarea = asyncio.create_task(hub.command("open_app", {"name": "spotify"}))
    await asyncio.sleep(0)
    hub.unregister(device)
    resultado = await tarea
    assert not resultado.ok


# ---------------------------------------------------------------------------
# Herramientas de teléfono (con hub real y fake)
# ---------------------------------------------------------------------------


class HubFalso:
    def __init__(self, resultado: CommandResult | Exception) -> None:
        self._resultado = resultado
        self.comandos: list[tuple[str, dict]] = []

    async def command(self, action, params=None, **kwargs) -> CommandResult:
        self.comandos.append((action, params or {}))
        if isinstance(self._resultado, Exception):
            raise self._resultado
        return self._resultado


def test_phone_call_con_numero():
    hub = HubFalso(CommandResult(True, "Llamando a 5512345678."))
    herramienta = PhoneCallTool(hub)
    texto = asyncio.run(herramienta.execute(number="5512345678"))
    assert texto == "Llamando a 5512345678."
    assert hub.comandos == [("call", {"number": "5512345678"})]


def test_whatsapp_con_texto_y_numero():
    hub = HubFalso(CommandResult(True, "WhatsApp abierto."))
    herramienta = WhatsAppTool(hub)
    texto = asyncio.run(
        herramienta.execute(message="ya voy", phone="+525512345678")
    )
    assert texto == "WhatsApp abierto."
    assert hub.comandos[0][0] == "whatsapp"
    assert hub.comandos[0][1]["text"] == "ya voy"


def test_email_valida_destinatario():
    hub = HubFalso(CommandResult(True, "Correo listo."))
    herramienta = SendEmailTool(hub)
    texto = asyncio.run(herramienta.execute(to="no-valido"))
    assert "correo" in texto.lower()
    texto_ok = asyncio.run(herramienta.execute(to="a@b.com", subject="hola"))
    assert texto_ok == "Correo listo."
    assert hub.comandos[-1] == ("email", {"to": "a@b.com", "subject": "hola", "body": ""})


def test_open_app_sin_nombre():
    hub = HubFalso(CommandResult(True, "ok"))
    herramienta = OpenPhoneAppTool(hub)
    texto = asyncio.run(herramienta.execute(app=" "))
    assert "aplicación" in texto.lower()


def test_herramientas_avisan_si_no_hay_celular():
    hub = HubFalso(DeviceError("No tengo ningún celular conectado."))
    texto = asyncio.run(PhoneCallTool(hub).execute(number="123"))
    assert "celular" in texto


def test_permisos_de_herramientas_de_telefono():
    assert PhoneCallTool.permission == ToolPermission.CONFIRM
    assert WhatsAppTool.permission == ToolPermission.CONFIRM
    assert SendEmailTool.permission == ToolPermission.CONFIRM
    assert OpenPhoneAppTool.permission == ToolPermission.CONFIRM
    assert PhoneContactsTool.permission == ToolPermission.SAFE
    assert SendSmsTool.permission == ToolPermission.CONFIRM
    assert SetAlarmTool.permission == ToolPermission.CONFIRM
    assert NavigateTool.permission == ToolPermission.CONFIRM
    assert PhoneStatusTool.permission == ToolPermission.SAFE
    assert PhoneTorchTool.permission == ToolPermission.SAFE
    assert PhoneVibrateTool.permission == ToolPermission.SAFE
    assert PhoneClipboardTool.permission == ToolPermission.SAFE
    assert PhoneVolumeTool.permission == ToolPermission.SAFE
    assert NotifyPhoneTool.permission == ToolPermission.SAFE


def test_phone_status_consulta_bateria():
    hub = HubFalso(CommandResult(True, "Batería al 87%, cargando."))
    texto = asyncio.run(PhoneStatusTool(hub).execute())
    assert "87" in texto
    assert hub.comandos == [("status", {})]


def test_torch_y_vibrate_mandan_parametros():
    hub = HubFalso(CommandResult(True, "Linterna encendida."))
    asyncio.run(PhoneTorchTool(hub).execute(mode="off"))
    assert hub.comandos[-1] == ("torch", {"mode": "off"})
    hub2 = HubFalso(CommandResult(True, "Vibré 700 ms."))
    asyncio.run(PhoneVibrateTool(hub2).execute(ms=700))
    assert hub2.comandos[-1] == ("vibrate", {"ms": 700})


def test_clipboard_exige_texto():
    hub = HubFalso(CommandResult(True, "Copiado."))
    texto = asyncio.run(PhoneClipboardTool(hub).execute(text="  "))
    assert "texto" in texto.lower()
    asyncio.run(PhoneClipboardTool(hub).execute(text="clave secreta"))
    assert hub.comandos[-1] == ("clipboard", {"text": "clave secreta"})


def test_sms_con_contacto_resuelve_numero():
    class HubAgenda(HubFalso):
        async def command(self, action, params=None, **kwargs):
            self.comandos.append((action, params or {}))
            if action == "contacts":
                return CommandResult(
                    True, "", {"contacts": [{"name": "Mamá", "number": "551"}]}
                )
            return self._resultado

    hub = HubAgenda(CommandResult(True, "SMS listo para 551."))
    texto = asyncio.run(SendSmsTool(hub).execute(text="ya salgo", contact="Mamá"))
    assert "SMS listo" in texto
    assert hub.comandos[-1] == ("sms", {"number": "551", "text": "ya salgo"})
    # sin destinatario no manda nada
    texto_vacio = asyncio.run(
        SendSmsTool(HubFalso(CommandResult(False, ""))).execute(text="hola")
    )
    assert "número" in texto_vacio.lower()


def test_alarma_requiere_hora_o_temporizador():
    hub = HubFalso(CommandResult(True, "Alarma lista: confirma."))
    texto = asyncio.run(SetAlarmTool(hub).execute())
    assert "hora" in texto.lower() and "timer" in texto.lower()
    asyncio.run(SetAlarmTool(hub).execute(hour=7, minute=30, label="trabajo"))
    assert hub.comandos[-1] == (
        "alarm", {"timer": False, "hour": 7, "minute": 30, "label": "trabajo"},
    )
    asyncio.run(SetAlarmTool(hub).execute(timer=True, minute=10))
    assert hub.comandos[-1][1]["timer"] is True


def test_navigate_valida_destino():
    hub = HubFalso(CommandResult(True, "Navegación abierta."))
    texto = asyncio.run(NavigateTool(hub).execute(destination=" "))
    assert "navegar" in texto.lower()
    asyncio.run(NavigateTool(hub).execute(destination="Aeropuerto"))
    assert hub.comandos[-1] == ("navigate", {"destination": "Aeropuerto"})


def test_volume_acota_nivel_y_consulta():
    hub = HubFalso(CommandResult(True, "Volumen multimedia al 40%."))
    asyncio.run(PhoneVolumeTool(hub).execute(level=250))
    assert hub.comandos[-1] == ("volume", {"level": 100})
    asyncio.run(PhoneVolumeTool(hub).execute())
    assert hub.comandos[-1] == ("volume", {})


def test_notify_phone_exige_mensaje():
    hub = HubFalso(CommandResult(True, "Notificación enviada a tu celular."))
    texto = asyncio.run(NotifyPhoneTool(hub).execute(message=""))
    assert "avisar" in texto.lower()
    asyncio.run(NotifyPhoneTool(hub).execute(message="Ya está listo tu build"))
    assert hub.comandos[-1] == (
        "notify", {"title": "A.R.I.A", "message": "Ya está listo tu build"},
    )


# ---------------------------------------------------------------------------
# Integración: el chat dispara una llamada en un celular conectado por WS
# ---------------------------------------------------------------------------


def _telefono_conectado(client: TestClient):
    ws = client.websocket_connect("/ws/device")
    ws.__enter__()
    ws.send_json({"type": "hello", "model": "TestPhone"})
    bienvenida = ws.receive_json()
    assert bienvenida["type"] == "welcome"
    return ws, bienvenida["id"]


def test_chat_ejecuta_llamada_en_el_celular(monkeypatch):
    llamada = ToolCall(id="c1", name="phone_call", arguments={"number": "5512345678"})
    proveedor = CapturingProvider([("", [llamada]), ("Ya la llamo.", None)])

    registry = ToolRegistry()
    registry.register(PhoneCallTool())
    client = TestClient(
        create_app(orchestrator=AssistantOrchestrator(proveedor, registry=registry))
    )

    telefono, _device_id = _telefono_conectado(client)
    try:
        with client.websocket_connect("/ws/chat") as chat:
            chat.send_json({"type": "message", "text": "llama a casa"})
            confirmacion = chat.receive_json()
            assert confirmacion["type"] == "confirm"
            assert confirmacion["tool"] == "phone_call"

            chat.send_json(
                {
                    "type": "confirm_response",
                    "request_id": confirmacion["request_id"],
                    "approved": True,
                }
            )
            # El celular recibe el comando y responde.
            comando = telefono.receive_json()
            assert comando["type"] == "cmd"
            assert comando["action"] == "call"
            assert comando["params"]["number"] == "5512345678"
            telefono.send_json(
                {
                    "type": "result",
                    "id": comando["id"],
                    "ok": True,
                    "message": "Llamando a 5512345678.",
                }
            )

            respuesta = chat.receive_json()
        assert respuesta["type"] == "response"
        assert respuesta["content"] == "Ya la llamo."
        # El resultado del celular llegó al LLM como texto de la herramienta.
        ultimo = proveedor.calls[-1]
        assert any("Llamando a 5512345678." in m.content for m in ultimo)
    finally:
        telefono.__exit__(None, None, None)


def test_device_rechaza_token_invalido(monkeypatch):
    from fastapi import WebSocketDisconnect

    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "access_token", "secreto")
    client = TestClient(create_app(orchestrator=AssistantOrchestrator(CapturingProvider([]))))

    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws/device") as ws:
        aviso = ws.receive_json()
        assert "token" in aviso["message"].lower()
        # Tras el aviso el servidor cierra la conexión.
        ws.receive_json()
