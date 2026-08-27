"""Estudio profundo de SIA: investigación a fondo de un tema en internet.

Cuando el usuario pide estudiar o investigar algo a fondo ("estudia la
fusión nuclear"), SIA:
1. Genera sub-preguntas diversas sobre el tema.
2. Busca cada una en la web (DuckDuckGo).
3. Redacta un resumen con los hallazgos.
4. Extrae los hechos más duraderos y los guarda en su memoria permanente
   (origin='web'), así el conocimiento queda para futuras conversaciones.
"""
from typing import ClassVar

from app.ai.providers.base import LLMProvider
from app.core.logging import get_logger
from app.memory.manager import MemoryManager
from app.tools.base import BaseTool, ToolPermission

logger = get_logger("sia.tools.study")

SearchFn = object  # Callable[[str], Awaitable[list[dict]]] (inyectado)

_SUBPREGUNTAS_PROMPT = (
    "Quiero entender bien este tema: {tema}\n"
    "Genera exactamente {n} preguntas de búsqueda cortas, específicas y "
    "diversas entre sí (qué es, cómo funciona, datos actuales, curiosidades). "
    "Una por línea, sin numeración ni guiones."
)

_RESUMEN_PROMPT = (
    "Tema: {tema}\n\nHallazgos de búsqueda:\n{hallazgos}\n\n"
    "Redacta un resumen claro de máximo 150 palabras que explique el tema "
    "usando esos hallazgos. Datos concretos por encima de generalidades."
)

_HECHOS_PROMPT = (
    "Del siguiente resumen extrae máximo 3 hechos DURADEROS y autocontenidos "
    "que valga la pena recordar para siempre. Una línea por hecho, español, "
    "frases completas:\n\n{resumen}"
)


class DeepStudyTool(BaseTool):
    """Investigación multi-búsqueda con memoria permanente."""

    name = "deep_study"
    description = (
        "Estudia un tema a fondo en internet y guarda lo aprendido en su "
        "memoria permanente. Para 'estudia/investiga a fondo X'."
    )
    permission = ToolPermission.SAFE
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "Tema a estudiar.",
            }
        },
        "required": ["topic"],
    }

    def __init__(
        self,
        provider: LLMProvider,
        memory: MemoryManager,
        search,
        *,
        n_subpreguntas: int = 3,
        max_resultados_por_busqueda: int = 2,
    ) -> None:
        self._provider = provider
        self._memory = memory
        self._search = search
        self._n_subpreguntas = n_subpreguntas
        self._max_por_busqueda = max_resultados_por_busqueda

    async def _run(self, topic: str, **kwargs) -> str:
        from app.ai.schemas import ChatMessage, ChatRole

        async def _chat(contenido: str, sistema: str) -> str:
            respuesta = await self._provider.chat(
                [
                    ChatMessage(role=ChatRole.SYSTEM, content=sistema),
                    ChatMessage(role=ChatRole.USER, content=contenido[:1800]),
                ]
            )
            return respuesta.content.strip()

        # 1) Sub-preguntas de investigación
        lineas = await _chat(
            _SUBPREGUNTAS_PROMPT.format(tema=topic, n=self._n_subpreguntas),
            "Eres un planificador de investigación: solo listas de preguntas.",
        )
        preguntas = [
            l.strip().lstrip("-•*0123456789. ").strip()
            for l in lineas.splitlines()
            if len(l.strip()) > 8
        ][: self._n_subpreguntas]
        if not preguntas:
            return f"No pude planificar la investigación de: {topic}"

        # 2) Búsquedas (el buscador ya traga errores de red → lista vacía)
        hallazgos: list[dict] = []
        for pregunta in preguntas:
            resultados = await self._search(pregunta)
            hallazgos.extend(resultados[: self._max_por_busqueda])
        if not hallazgos:
            return f"No encontré información web para estudiar: {topic}"

        bloque = "\n".join(
            f"- [{r.get('title', '')}] {r.get('snippet', '')} ({r.get('url', '')})"
            for r in hallazgos
            if r.get("title") or r.get("snippet")
        )

        # 3) Resumen legible para el usuario
        resumen = await _chat(
            _RESUMEN_PROMPT.format(tema=topic, hallazgos=bloque),
            "Eres una investigadora rigurosa: resúmenes claros y concretos.",
        )

        # 4) Hechos duraderos → memoria permanente
        hechos_texto = await _chat(
            _HECHOS_PROMPT.format(resumen=resumen),
            "Extraes hechos duraderos: solo líneas de hechos, sin preámbulos.",
        )
        hechos = [
            h.strip().lstrip("-•* ").strip()
            for h in hechos_texto.splitlines()
            if len(h.strip()) > 20 and not h.strip().upper().startswith("NADA")
        ][:3]
        for hecho in hechos:
            self._memory.remember(hecho, origin="web")

        logger.info(
            "Estudio profundo completado",
            extra={"topic": topic, "busquedas": len(preguntas), "hechos": len(hechos)},
        )
        salida = f"Estudio de '{topic}' ({len(preguntas)} búsquedas):\n\n{resumen}"
        if hechos:
            salida += "\n\nGuardado en mi memoria permanente:\n" + "\n".join(
                f"- {h}" for h in hechos
            )
        return salida
