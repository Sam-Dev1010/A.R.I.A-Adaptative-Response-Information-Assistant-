"""GUI web de SIA (FASE 8): chat por WebSocket con confirmaciones de tools."""
import asyncio
import logging
import uuid

from fastapi import WebSocket, WebSocketDisconnect

from app.ai.orchestrator import AssistantOrchestrator
from app.ai.providers.base import LLMError
from app.ai.schemas import ToolCall

logger = logging.getLogger("sia.web")

_CONFIRM_TIMEOUT_SECONDS = 120


class _PendingConfirmations:
    """Espera la decisión del usuario para una herramienta CONFIRM."""

    def __init__(self) -> None:
        self._pending: dict[str, asyncio.Future[bool]] = {}

    def create(self) -> tuple[str, asyncio.Future[bool]]:
        request_id = uuid.uuid4().hex
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        return request_id, future

    def resolve(self, request_id: str, approved: bool) -> bool:
        future = self._pending.pop(request_id, None)
        if future is None or future.done():
            return False
        future.set_result(approved)
        return True

    async def wait(self, future: asyncio.Future[bool]) -> bool:
        try:
            return await asyncio.wait_for(future, _CONFIRM_TIMEOUT_SECONDS)
        except TimeoutError:
            return False


class ChatConnection:
    """Atiende una conexión WebSocket de chat contra un orquestador."""

    def __init__(self, ws: WebSocket, orchestrator: AssistantOrchestrator) -> None:
        self._ws = ws
        self._orchestrator = orchestrator
        self._pending = _PendingConfirmations()
        self._queue: asyncio.Queue = asyncio.Queue()

    async def run(self) -> None:
        await self._ws.accept()

        # Un orquestador por conexión para no mezclar historiales entre clientes.
        session = AssistantOrchestrator(
            self._orchestrator.provider,
            system_prompt=self._orchestrator.system_prompt,
            registry=self._orchestrator.registry,
            policy=self._orchestrator.policy,
            confirm=self._confirm,
            memory=self._orchestrator.memory,
            auto_learner=self._orchestrator.auto_learner,
        )

        reader = asyncio.create_task(self._receive_loop())
        try:
            while True:
                text = (await self._queue.get()).strip()
                if not text:
                    continue
                try:
                    response = await session.ask(text)
                except LLMError as exc:
                    await self._ws.send_json({"type": "error", "message": str(exc)})
                    continue
                await self._ws.send_json(
                    {
                        "type": "response",
                        "content": response.content,
                        "usage": response.usage.model_dump() if response.usage else None,
                    }
                )
        except WebSocketDisconnect:
            pass
        finally:
            reader.cancel()

    async def _confirm(self, call: ToolCall) -> bool:
        request_id, future = self._pending.create()
        await self._ws.send_json(
            {
                "type": "confirm",
                "request_id": request_id,
                "tool": call.name,
                "arguments": call.arguments,
            }
        )
        return await self._pending.wait(future)

    async def _receive_loop(self) -> None:
        """Lee los mensajes del cliente: chats al queue, confirmaciones resueltas."""
        while True:
            raw = await self._ws.receive_json()
            kind = raw.get("type")
            if kind == "message":
                await self._queue.put(raw.get("text", ""))
            elif kind == "confirm_response":
                self._pending.resolve(raw.get("request_id", ""), bool(raw.get("approved")))