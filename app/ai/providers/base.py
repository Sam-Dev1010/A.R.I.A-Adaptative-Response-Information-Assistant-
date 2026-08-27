"""Contrato de los proveedores LLM (FASE 2)."""
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.ai.schemas import ChatMessage, ChatResponse


class LLMError(RuntimeError):
    """Error de la capa de IA (red, API, formato de respuesta, etc.)."""


class LLMProvider(ABC):
    """Interfaz común de cualquier proveedor de LLM.

    FASE 3: las herramientas se integrarán aquí mediante el parámetro ``tools``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Identificador del proveedor (ej: ``openai_compatible``)."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Nombre del modelo en uso (ej: ``gpt-4o-mini``)."""

    @abstractmethod
    async def chat(self, messages: list[ChatMessage]) -> ChatResponse:
        """Envía una conversación completa y devuelve la respuesta del modelo.

        Levanta :class:`LLMError` ante cualquier fallo de red, API o formato.
        """

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        tools=None,
    ) -> AsyncIterator[tuple[str, object]]:
        """Versión en streaming de ``chat`` (opcional).

        Emite eventos ``("delta", texto)`` con cada fragmento conforme llega y
        termina con exactamente un evento ``("final", ChatResponse)``, cuyo
        ``content`` es la unión de los deltas. El contrato lo respeta cualquier
        implementación; por defecto delega en ``chat()`` sin streaming real.
        """
        response = await self.chat(messages, **({"tools": tools} if tools else {}))
        if response.content:
            yield ("delta", response.content)
        yield ("final", response)

    async def aclose(self) -> None:
        """Libera los recursos del proveedor (idempotente)."""
