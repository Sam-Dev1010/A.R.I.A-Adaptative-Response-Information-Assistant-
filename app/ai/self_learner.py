"""Autoaprendizaje de SIA: memoriza sola lo que importa del usuario.

Después de cada intercambio, un segundo pase ligero del LLM extrae hechos
duraderos sobre Samuel (gustos, personas, proyectos, rutinas, datos) y los
guarda en la memoria a largo plazo. En la siguiente conversación esos hechos
se inyectan en el prompt: SIA "ya sabe" sin que nadie se lo repita.

Diseño:
- Nunca lanza excepciones hacia la conversación (todo se captura aquí).
- Idempotente: la tabla ``facts`` tiene UNIQUE(content).
- Barato: una sola llamada con respuesta corta.
"""
import asyncio

from app.ai.providers.base import LLMProvider
from app.core.logging import get_logger
from app.memory.manager import MemoryManager

logger = get_logger("sia.ai.learn")

_EXTRACCION_PROMPT = (
    "Analiza este intercambio y extrae SOLO hechos duraderos y nuevos sobre "
    "el USUARIO (quién es, gustos, personas cercanas, proyectos, horarios, "
    "preferencias, datos personales). Nada de opiniones pasajeras ni cosas "
    "obvias del momento.\n"
    "Formato: una línea por hecho, tercera persona, empezando con 'El usuario '.\n"
    "Si no hay nada que valga la pena recordar responde exactamente: NADA\n\n"
    "USUARIO: {usuario}\n\nSIA: {asistente}"
)


class SelfLearner:
    """Extrae y persiste hechos duraderos del usuario tras cada intercambio."""

    def __init__(
        self,
        provider: LLMProvider,
        memory: MemoryManager,
        *,
        max_facts_por_intercambio: int = 3,
    ) -> None:
        self._provider = provider
        self._memory = memory
        self._max_facts = max_facts_por_intercambio
        self._lock = asyncio.Lock()

    @property
    def provider(self) -> LLMProvider:
        return self._provider

    async def learn(self, user_text: str, assistant_text: str) -> list[str]:
        """Extrae hechos del intercambio y los guarda. Devuelve lo aprendido."""
        async with self._lock:  # una extracción a la vez
            from app.ai.schemas import ChatMessage, ChatRole

            messages = [
                ChatMessage(
                    role=ChatRole.SYSTEM,
                    content=(
                        "Eres un extractor de información personal ultra "
                        "selectivo. Respondes únicamente líneas de hechos o NADA."
                    ),
                ),
                ChatMessage(
                    role=ChatRole.USER,
                    content=_EXTRACCION_PROMPT.format(
                        usuario=user_text[:600], asistente=assistant_text[:600]
                    ),
                ),
            ]
            response = await self._provider.chat(messages)

            aprendidos: list[str] = []
            for linea in response.content.splitlines():
                hecho = linea.strip().lstrip("-•* ").strip()
                if not hecho or len(hecho) < 12 or hecho.upper() == "NADA":
                    continue
                if len(aprendidos) >= self._max_facts:
                    break
                self._memory.remember(hecho, origin="auto")
                aprendidos.append(hecho)

            if aprendidos:
                logger.info(
                    "Autoaprendizaje: %d hecho(s) nuevo(s)", len(aprendidos),
                    extra={"facts": aprendidos},
                )
            return aprendidos
