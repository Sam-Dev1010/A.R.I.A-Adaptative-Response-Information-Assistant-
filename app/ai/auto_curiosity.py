"""Curiosidad autónoma de ARIA: investiga sola temas nuevos en internet.

Después de una conversación interesante, el motor propone UNA pregunta que
vale la pena investigar, la busca en la web (DuckDuckGo vía ``web_search``),
sintetiza lo esencial y guarda el conocimiento en su memoria a largo plazo.
Así ARIA expande lo que sabe sin que nadie se lo enseñe directamente.

Límites de presupuesto para no dispararse:
- Tiempo mínimo entre investigaciones (cooldown).
- Tope de investigaciones por hora.
- Nunca lanza excepciones hacia la conversación.
"""
import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable

from app.ai.providers.base import LLMProvider
from app.core.logging import get_logger
from app.memory.manager import MemoryManager

logger = get_logger("sia.ai.curiosity")

SearchFn = Callable[[str], Awaitable[list[dict]]]

_TEMA_PROMPT = (
    "Analiza este intercambio con el usuario.\n\n"
    "USUARIO: {usuario}\nSIA: {asistente}\n\n"
    "Si hay algún tema que merezca investigación en internet para saber más "
    "(ciencia, tecnología, historia, cultura, datos verificables...), escribe "
    "UNA sola pregunta específica de búsqueda. Temas personales, operativos o "
    "triviales NO cuentan.\n"
    "Si no hay nada que valga la pena, responde exactamente: NADA"
)

_SINTESIS_PROMPT = (
    "Estos son resultados de búsqueda sobre: {tema}\n\n{resultados}\n\n"
    "Extrae como máximo 2 ideas DURADERAS y útiles que una IA debería recordar "
    "para siempre de este tema. Una por línea, en español, frases completas y "
    "autocontenidas (que se entiendan sin contexto). Sin URLs ni fechas relativas.\n"
    "Si nada vale la pena, responde exactamente: NADA"
)


class CuriosityEngine:
    """Convierte conversaciones en conocimiento nuevo investigando la web."""

    def __init__(
        self,
        provider: LLMProvider,
        memory: MemoryManager,
        search: SearchFn,
        *,
        cooldown_seconds: float = 90.0,
        max_por_hora: int = 8,
        max_resultados_usados: int = 4,
    ) -> None:
        self._provider = provider
        self._memory = memory
        self._search = search
        self._cooldown = cooldown_seconds
        self._max_por_hora = max_por_hora
        self._max_resultados = max_resultados_usados
        self._lock = asyncio.Lock()
        self._ultima_investigacion = float("-inf")  # sin historial: nunca en cooldown
        self._marcas_recientes: deque[float] = deque()

    @property
    def stats(self) -> dict:
        return {
            "investigaciones_ultima_hora": len(self._marcas_recientes),
            "en_cooldown": (time.monotonic() - self._ultima_investigacion) < self._cooldown,
        }

    async def research(self, user_text: str, assistant_text: str) -> list[str]:
        """Ciclo completo: tema → búsqueda → síntesis → memoria."""
        if not self._presupuesto_disponible():
            return []
        async with self._lock:
            if not self._presupuesto_disponible():
                return []
            self._ultima_investigacion = time.monotonic()
            self._marcas_recientes.append(time.monotonic())

            pregunta = await self._proponer_tema(user_text, assistant_text)
            if not pregunta:
                return []

            resultados = await self._search(pregunta)
            if not resultados:
                logger.info("Curiosidad sin resultados", extra={"topic": pregunta})
                return []

            aprendido = await self._sintetizar(pregunta, resultados)
            for hecho in aprendido:
                self._memory.remember(hecho, origin="web")
            if aprendido:
                logger.info(
                    "Curiosidad: %d idea(s) nueva(s) sobre %r",
                    len(aprendido), pregunta,
                    extra={"facts": aprendido},
                )
            return aprendido

    # --- pasos -------------------------------------------------------------

    def _presupuesto_disponible(self) -> bool:
        ahora = time.monotonic()
        while self._marcas_recientes and ahora - self._marcas_recientes[0] > 3600:
            self._marcas_recientes.popleft()
        if len(self._marcas_recientes) >= self._max_por_hora:
            return False
        return (ahora - self._ultima_investigacion) >= self._cooldown

    async def _proponer_tema(self, user_text: str, assistant_text: str) -> str | None:
        from app.ai.schemas import ChatMessage, ChatRole

        respuesta = await self._provider.chat(
            [
                ChatMessage(
                    role=ChatRole.SYSTEM,
                    content=(
                        "Eres un motor de curiosidad: eliges temas fascinantes y "
                        "específicos para investigar. Respondes solo una línea."
                    ),
                ),
                ChatMessage(
                    role=ChatRole.USER,
                    content=_TEMA_PROMPT.format(
                        usuario=user_text[:600], asistente=assistant_text[:400]
                    ),
                ),
            ]
        )
        linea = respuesta.content.strip().strip('"')
        if not linea or len(linea) < 12 or linea.upper().startswith("NADA"):
            return None
        return linea[:200]

    async def _sintetizar(self, pregunta: str, resultados: list[dict]) -> list[str]:
        from app.ai.schemas import ChatMessage, ChatRole

        bloque = "\n".join(
            f"- {r.get('title', '')}: {r.get('snippet', '')[:220]}"
            for r in resultados[: self._max_resultados]
            if r.get("title") or r.get("snippet")
        )
        if not bloque:
            return []
        respuesta = await self._provider.chat(
            [
                ChatMessage(
                    role=ChatRole.SYSTEM,
                    content=(
                        "Eres un sintetizador de conocimiento: extraes hechos "
                        "duraderos y verificables. Respondes solo líneas de hechos o NADA."
                    ),
                ),
                ChatMessage(
                    role=ChatRole.USER,
                    content=_SINTESIS_PROMPT.format(tema=pregunta, resultados=bloque)[:1800],
                ),
            ]
        )
        ideas: list[str] = []
        for linea in respuesta.content.splitlines():
            idea = linea.strip().lstrip("-•* ").strip()
            if len(idea) < 20 or idea.upper().startswith("NADA"):
                continue
            ideas.append(idea[:280])
            if len(ideas) == 2:
                break
        return ideas
