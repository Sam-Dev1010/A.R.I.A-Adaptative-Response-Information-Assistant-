"""Orquestador del asistente: conversación con el LLM, historial y
ciclo de herramientas con permisos (FASES 2-3).

Flujo: LLM → tool_calls → permisos → ejecución → resultados → LLM → respuesta.
"""
import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime

from app.ai.personality import build_personality_prompt
from app.ai.providers.base import LLMProvider
from app.ai.schemas import ChatMessage, ChatResponse, ChatRole, ToolCall
from app.core.logging import get_logger
from app.memory.manager import MemoryManager
from app.tools.policy import PermissionDenied, ToolPolicy
from app.tools.registry import ToolRegistry
from app.tools.schemas import ToolResult, ToolSpec

logger = get_logger("sia.ai")

DEFAULT_SYSTEM_PROMPT = build_personality_prompt()

_DIAS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
_MESES = (
    "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
    "agosto", "septiembre", "octubre", "noviembre", "diciembre",
)


def _contexto_temporal() -> str:
    """Fecha y hora actuales en español (evita una llamada a get_time)."""
    now = datetime.now().astimezone()
    fecha = f"{_DIAS[now.weekday()]}, {now.day} de {_MESES[now.month - 1]} de {now.year}"
    zona = now.strftime("%Z") or "hora local"
    return f"{fecha} — {now:%H:%M} ({zona})"

_DEFAULT_MAX_HISTORY = 12
_MAX_TOOL_ROUNDS = 5

ConfirmFn = Callable[[ToolCall], Awaitable[bool]]
LearnFn = Callable[[str, str], Awaitable[object]]


class ToolLoopError(RuntimeError):
    """El LLM encadenó demasiadas rondas de herramientas."""


class AssistantOrchestrator:
    """Mantiene el estado de la conversación y delega las llamadas al LLM.

    Si se entrega un ``registry``, el orquestador expone las herramientas al
    LLM y resuelve sus llamadas: aplica la política de permisos, ejecuta la
    herramienta y devuelve el resultado al modelo.
    """

    def __init__(
        self,
        provider: LLMProvider | None = None,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_history_messages: int = _DEFAULT_MAX_HISTORY,
        registry: ToolRegistry | None = None,
        policy: ToolPolicy | None = None,
        confirm: ConfirmFn | None = None,
        memory: MemoryManager | None = None,
        auto_learner: LearnFn | None = None,
        neural_brain=None,
    ) -> None:
        self._provider = provider
        self._system_prompt = system_prompt
        self._auto_learner = auto_learner
        self._learning = False
        self._neural_brain = neural_brain
        # El historial debe conservar pares user/assistant completos.
        self._max_history = max_history_messages - (max_history_messages % 2)
        self._memory = memory
        if memory is not None:
            # Retoma la última conversación guardada (memoria persistente).
            self._history = memory.recent_messages(self._max_history)
        else:
            self._history = []
        self._registry = registry
        self._policy = policy or ToolPolicy()
        self._confirm = confirm
        # Descubrimientos autónomos aún no contados: se reservan al abrir la
        # sesión (y se marcan como contados) para mencionarlos si encaja.
        if memory is not None:
            self._novedades = memory.pending_discoveries()
            memory.mark_shared(self._novedades)
        else:
            self._novedades = []

    @property
    def provider(self) -> LLMProvider | None:
        return self._provider

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def history(self) -> list[ChatMessage]:
        """Copia del historial actual (solo user/assistant)."""
        return list(self._history)

    @property
    def tool_count(self) -> int:
        return len(self._registry) if self._registry else 0

    @property
    def registry(self) -> ToolRegistry | None:
        return self._registry

    @property
    def policy(self) -> ToolPolicy:
        return self._policy

    @property
    def memory(self) -> MemoryManager | None:
        return self._memory

    @property
    def neural_brain(self):
        """Cerebro neural de A.R.I.A (si está habilitado)."""
        return self._neural_brain

    @property
    def auto_learner(self) -> LearnFn | None:
        return self._auto_learner

    @property
    def novedades(self) -> list[str]:
        """Descubrimientos recientes reservados para contar en esta sesión."""
        return list(self._novedades)

    def reset(self) -> None:
        """Borra el historial de la conversación."""
        self._history.clear()

    async def ask(self, user_message: str) -> ChatResponse:
        """Procesa un mensaje del usuario: construye el contexto, llama al LLM,
        resuelve las herramientas que pida y registra el intercambio final."""
        user_message = user_message.strip()
        if not user_message:
            raise ValueError("El mensaje del usuario no puede estar vacío")

        # Intentar usar el neural brain para TODO
        if self._neural_brain is not None:
            neural_response = await self._try_neural_with_tools(user_message)
            if neural_response is not None:
                # Registrar el intercambio y devolver respuesta del neural
                response = ChatResponse(
                    content=neural_response,
                    model="neural",
                    provider="local",
                )
                self._record_exchange(user_message, response)
                if self._memory is not None:
                    self._memory.add_exchange(user_message, neural_response)
                self._trigger_learning(user_message, neural_response)
                return response

        # Si el neural no pudo responder, usar el LLM (si hay proveedor)
        if self._provider is None:
            response = ChatResponse(
                content="No puedo procesar eso ahora mismo.",
                model="none",
                provider="none",
            )
            self._record_exchange(user_message, response)
            return response

        working = self._build_messages(user_message)
        tools = self._tools_payload()

        started = time.monotonic()
        for round_number in range(1, _MAX_TOOL_ROUNDS + 1):
            response = await self._provider.chat(working, tools=tools)
            self._log_provider_call(working, response, started)

            if not response.tool_calls:
                break

            if round_number == _MAX_TOOL_ROUNDS:
                raise ToolLoopError(
                    f"El LLM pidió herramientas {_MAX_TOOL_ROUNDS} veces seguidas"
                )

            working.append(
                ChatMessage(
                    role=ChatRole.ASSISTANT,
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )
            for call in response.tool_calls:
                result = await self._execute_tool_call(call)
                working.append(
                    ChatMessage(
                        role=ChatRole.TOOL,
                        content=result.output,
                        tool_call_id=result.tool_call_id,
                    )
                )
        else:
            raise ToolLoopError("El bucle de herramientas terminó sin respuesta final")

        self._record_exchange(user_message, response)
        if self._memory is not None:
            self._memory.add_exchange(user_message, response.content)
        self._trigger_learning(user_message, response.content)

        logger.info(
            "Respuesta del LLM recibida",
            extra={
                "chars": len(response.content),
                "usage": response.usage.model_dump() if response.usage else None,
            },
        )
        return response

    async def _try_neural_with_tools(self, user_message: str) -> str | None:
        """Intenta usar el neural brain con capacidad de ejecutar herramientas."""
        try:
            # think() es async ahora — ejecutar directamente
            response = await self._neural_brain.think(user_message)
            return response

        except Exception as e:  # noqa: BLE001 - fallback definido: ante cualquier fallo local, usar el LLM
            logger.debug("Neural brain falló: %s", e)
            return None

    def _extract_tool_args(self, tool_name: str, message: str) -> dict | None:
        """Extrae argumentos para una herramienta desde el mensaje."""
        msg_lower = message.lower()

        if tool_name == "run_command":
            # Extraer comando del mensaje
            for prefix in ["ejecuta", "corre", "run", "instala", "abre"]:
                if prefix in msg_lower:
                    cmd = msg_lower.split(prefix, 1)[1].strip()
                    if cmd:
                        return {"command": cmd, "timeout": 30}

        elif tool_name == "create_file":
            # Detectar si quiere crear un archivo
            if "crea" in msg_lower or "guarda" in msg_lower:
                return {"path": "output.txt", "content": message}

        elif tool_name == "list_files":
            return {"path": "."}

        return None

    async def ask_stream(self, user_message: str) -> AsyncIterator[str]:
        """Como ``ask`` pero emite la respuesta por fragmentos según llega.

        Permite que la voz empiece a sonar mientras el LLM todavía escribe.
        El texto completo (todos los fragmentos emitidos) queda registrado en
        historial y memoria al terminar.
        """
        user_message = user_message.strip()
        if not user_message:
            raise ValueError("El mensaje del usuario no puede estar vacío")

        working = self._build_messages(user_message)
        tools = self._tools_payload()
        started = time.monotonic()
        fragmentos: list[str] = []
        response: ChatResponse | None = None

        for round_number in range(1, _MAX_TOOL_ROUNDS + 1):
            response = None
            async for kind, payload in self._provider.stream_chat(working, tools=tools):
                if kind == "delta":
                    fragmentos.append(payload)
                    yield payload
                elif kind == "final":
                    response = payload
            if response is None:
                raise ToolLoopError("El proveedor no devolvió respuesta final")
            self._log_provider_call(working, response, started)

            if not response.tool_calls:
                break
            if round_number == _MAX_TOOL_ROUNDS:
                raise ToolLoopError(
                    f"El LLM pidió herramientas {_MAX_TOOL_ROUNDS} veces seguidas"
                )

            working.append(
                ChatMessage(
                    role=ChatRole.ASSISTANT,
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )
            for call in response.tool_calls:
                result = await self._execute_tool_call(call)
                working.append(
                    ChatMessage(
                        role=ChatRole.TOOL,
                        content=result.output,
                        tool_call_id=result.tool_call_id,
                    )
                )
        else:
            raise ToolLoopError("El bucle de herramientas terminó sin respuesta final")

        full_text = "".join(fragmentos)
        self._record_exchange(
            user_message,
            ChatResponse(
                content=full_text,
                model=response.model,
                provider=response.provider,
                usage=response.usage,
            ),
        )
        if self._memory is not None:
            self._memory.add_exchange(user_message, full_text)
        self._trigger_learning(user_message, full_text)
        logger.info(
            "Respuesta del LLM en streaming",
            extra={
                "chars": len(full_text),
                "usage": response.usage.model_dump() if response.usage else None,
            },
        )

    def _trigger_learning(self, user_message: str, assistant_message: str) -> None:
        """Lanza el autoaprendizaje en segundo plano (no bloquea ni falla).

        Aprende de intercambios con contenido real; si ya hay una extracción
        en curso se salta esta para no acumular llamadas al LLM.
        """
        if self._auto_learner is None or self._learning:
            return
        if len(user_message) < 20 or len(assistant_message) < 10:
            return

        async def _aprender() -> None:
            try:
                await self._auto_learner(user_message, assistant_message)
            except Exception as exc:  # noqa: BLE001 — aprender jamás rompe la charla
                logger.debug("Autoaprendizaje falló: %s", exc)
            finally:
                self._learning = False

        self._learning = True
        asyncio.create_task(_aprender())

    def _log_provider_call(
        self,
        messages: list[ChatMessage],
        response: ChatResponse,
        started: float,
    ) -> None:
        logger.info(
            "Llamada al LLM",
            extra={
                "model": self._provider.model,
                "provider": self._provider.name,
                "messages": len(messages),
                "tool_calls": len(response.tool_calls) if response.tool_calls else 0,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
            },
        )

    def _tools_payload(self) -> list[ToolSpec] | None:
        if not self._registry:
            return None
        specs = self._registry.specs()
        return specs or None

    async def _execute_tool_call(self, call: ToolCall) -> ToolResult:
        tool = self._registry.get(call.name) if self._registry else None
        if tool is None:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                ok=False,
                output=f"Herramienta desconocida: {call.name}",
            )

        try:
            self._policy.check(tool)
        except PermissionDenied as exc:
            if self._confirm is None:
                raise
            if not await self._confirm(call):
                return ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    ok=False,
                    output=f"El usuario denegó el permiso: {exc.reason}",
                )

        try:
            output = await tool.execute(**call.arguments)
        except Exception as exc:  # noqa: BLE001 — el error va al LLM, no al proceso
            logger.warning(
                "Error ejecutando herramienta",
                extra={"tool": call.name, "error": str(exc)},
            )
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                ok=False,
                output=f"Error ejecutando {call.name}: {exc}",
            )

        logger.info("Herramienta ejecutada", extra={"tool": call.name, "ok": True})
        return ToolResult(tool_call_id=call.id, name=call.name, ok=True, output=output)

    def _build_messages(self, user_message: str) -> list[ChatMessage]:
        history = self._history[-self._max_history :]
        return [
            ChatMessage(role=ChatRole.SYSTEM, content=self._system_content()),
            *history,
            ChatMessage(role=ChatRole.USER, content=user_message),
        ]

    def _system_content(self) -> str:
        """Prompt de sistema + contexto temporal + hechos recordados."""
        partes = [
            self._system_prompt,
            (
                "Contexto temporal: "
                f"{_contexto_temporal()}. Puedes responder preguntas de fecha "
                "y hora directamente con estos datos, sin usar herramientas."
            ),
        ]
        if self._novedades:
            bloque = "\n".join(f"- {d}" for d in self._novedades)
            partes.append(
                "Cosas que aprendiste o investigaste recientemente por tu cuenta "
                "y todavía no le has contado al usuario. Si saluda, pregunta qué "
                "hay de nuevo, o el tema encaja naturalmente, cuéntale UNA de "
                "estas como curiosidad ('¿sabes qué? estuve investigando...'); "
                "nunca fuerces el tema:\n" + bloque
            )
        if self._memory is None:
            return "\n\n".join(partes)
        facts = self._memory.facts()
        if not facts:
            return "\n\n".join(partes)
        facts_block = "\n".join(f"- {fact}" for fact in facts)
        return "\n\n".join(
            [*partes, f"Recuerdos de conversaciones pasadas:\n{facts_block}"]
        )

    def _record_exchange(self, user_message: str, response: ChatResponse) -> None:
        self._history.append(ChatMessage(role=ChatRole.USER, content=user_message))
        self._history.append(ChatMessage(role=ChatRole.ASSISTANT, content=response.content))
        if len(self._history) > self._max_history:
            del self._history[: len(self._history) - self._max_history]