"""Failover entre proveedores LLM: principal + respaldo (FASE 10).

Prueba el proveedor principal y, si lanza un error, usa el respaldo
automáticamente. Ideal para combinar Gemini (principal) con Groq (respaldo).
"""
import logging

from app.ai.providers.base import LLMError, LLMProvider
from app.ai.schemas import ChatMessage, ChatResponse
from app.tools.schemas import ToolSpec

logger = logging.getLogger("sia.ai")


class FallbackProvider(LLMProvider):
    """Envuelve un proveedor principal y uno de respaldo con failover."""

    name = "fallback"

    def __init__(self, primary: LLMProvider, fallback: LLMProvider) -> None:
        self._primary = primary
        self._fallback = fallback
        self._active = primary
        self._switched = False

    @property
    def model(self) -> str:
        return self._active.model

    @property
    def primary(self) -> LLMProvider:
        return self._primary

    @property
    def fallback(self) -> LLMProvider:
        return self._fallback

    @property
    def active(self) -> LLMProvider:
        return self._active

    @property
    def switched(self) -> bool:
        return self._switched

    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
    ) -> ChatResponse:
        if self._switched:
            return await self._fallback.chat(messages, tools=tools)

        try:
            return await self._primary.chat(messages, tools=tools)
        except LLMError as exc:
            logger.warning(
                "Conmutando al proveedor de respaldo: %s",
                exc,
                extra={
                    "primary": self._primary.name,
                    "fallback": self._fallback.name,
                },
            )
            self._switched = True
            self._active = self._fallback
            return await self._fallback.chat(messages, tools=tools)

    async def aclose(self) -> None:
        await self._primary.aclose()
        await self._fallback.aclose()