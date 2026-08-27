"""Charla espontánea: SIA toma la iniciativa y habla sin que le pregunten.

Un bucle en segundo plano decide cada cierto tiempo (aleatorio entre
``PROACTIVE_MIN_MINUTES`` y ``PROACTIVE_MAX_MINUTES``) si decir algo, con
presupuesto por hora y horas de silencio para no ser pesada.

Qué dice (en orden de prioridad):
    1. Novedades: descubrimientos de su curiosidad autónoma aún no contados.
    2. Rotación de temas: saludo según la hora, dato curioso que ya sabe,
       o preguntar cómo va el día/proyecto de su creador.

Nunca interrumpe: solo habla si hay una interfaz conectada y libre
(``interface_ws.hablar_a_todas`` se lo garantiza).
"""
import asyncio
import logging
import random
import time
from collections import deque
from datetime import datetime

from app.ai.personality import build_personality_prompt
from app.core.config import Settings, get_settings
from app.memory.manager import MemoryManager
from app.voice.base import TTSProvider
from app.web.interface_ws import hablar_a_todas, hay_alguien_escuchando

logger = logging.getLogger("sia.proactiva")

_LINEA_MAX = 320  # caracteres: un comentario corto, no un discurso


class ProactiveSpeaker:
    """Motor de iniciativa: programa y pronuncia comentarios espontáneos."""

    def __init__(
        self,
        provider,
        memory: MemoryManager,
        tts: TTSProvider,
        settings: Settings | None = None,
        *,
        reloj=None,
        sorteo=None,
    ) -> None:
        self._provider = provider
        self._memory = memory
        self._tts = tts
        self._settings = settings or get_settings()
        self._reloj = reloj or datetime.now  # inyectable para tests
        self._sorteo = sorteo or random.uniform
        self._task: asyncio.Task | None = None
        self._dichas: deque[str] = deque(maxlen=6)  # anti-repetición
        self._habladas_en_hora: deque[float] = deque()
        self._tema_idx = 0
        self._temas = ("dato", "pregunta", "saludo")

    # ---- ciclo de vida ----------------------------------------------------

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._bucle())
            logger.info("Charla espontánea activada")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _bucle(self) -> None:
        s = self._settings
        while True:
            espera = self._sorteo(
                max(s.proactive_min_minutes, 1) * 60,
                max(s.proactive_max_minutes, s.proactive_min_minutes + 1) * 60,
            )
            await asyncio.sleep(espera)
            try:
                await self._quizas_hablar()
            except Exception:
                logger.exception("Fallo en la charla espontánea")

    # ---- reglas de cortesía -------------------------------------------------

    def _en_horario(self, ahora: datetime | None = None) -> bool:
        """False durante las horas de silencio (soporta rangos que cruzan medianoche)."""
        s = self._settings
        hora = (ahora or self._reloj()).hour
        inicio, fin = s.proactive_quiet_start, s.proactive_quiet_end
        if inicio == fin:
            return True
        if inicio < fin:
            return not (inicio <= hora < fin)
        return not (hora >= inicio or hora < fin)

    def _presupuesto_ok(self, ahora: datetime | None = None) -> bool:
        """Máximo ``proactive_max_per_hour`` comentarios en la última hora."""
        limite = max(self._settings.proactive_max_per_hour, 1)
        marca = time.monotonic()
        while self._habladas_en_hora and marca - self._habladas_en_hora[0] > 3600:
            self._habladas_en_hora.popleft()
        return len(self._habladas_en_hora) < limite

    # ---- contenido ----------------------------------------------------------

    async def _novedad_pendiente(self) -> tuple[str | None, list[str]]:
        """Devuelve (instrucción, hechos_a_marcar) si hay descubrimiento sin contar."""
        hechos = self._memory.pending_discoveries(limit=3)
        if not hechos:
            return None, []
        bloque = "\n".join(f"- {h}" for h in hechos)
        instruccion = (
            "Estuviste investigando algo por tu cuenta. Cuéntaselo a tu creador "
            "con naturalidad y entusiasmo, en UNA o DOS frases cortas habladas "
            "(empezando tipo '¿Sabes qué?...'). NO uses listas ni estos textos "
            "literales:\n" + bloque
        )
        return instruccion, list(hechos)

    def _instruccion_rotativa(self) -> str:
        nombre = self._settings.aria_creator_name
        ahora = self._reloj()
        tema = self._temas[self._tema_idx % len(self._temas)]
        self._tema_idx += 1
        evita = "\n".join(f"- {d}" for d in self._dichas) or "(nada aún)"
        base = (
            f"Hora local: {ahora:%H:%M}. Genera UNA sola frase corta y natural "
            "para decir en voz alta. Sin listas ni markdown.\n"
            f"NO repitas ni reformules nada de esto que ya dijiste:\n{evita}\n"
        )
        if tema == "saludo":
            franja = (
                "mañana" if ahora.hour < 12 else "tarde" if ahora.hour < 19 else "noche"
            )
            return base + (
                f"Tema: un comentario espontáneo de {franja} para {nombre} — "
                "salúdalo con calidez o coméntale algo del momento del día."
            )
        if tema == "dato":
            return base + (
                "Tema: comparte UN dato curioso breve (ciencia, tecnología o "
                "historia) que sepas de memoria. Nada de noticias ni fechas dudosas."
            )
        return base + (
            f"Tema: pregunta con interés genuino cómo va el día o algún proyecto "
            f"de {nombre}. Una frase, tono ligero."
        )

    async def _generar_linea(self, instruccion: str) -> str:
        from app.ai.schemas import ChatMessage, ChatRole

        prompt = build_personality_prompt(self._settings.aria_creator_name)
        respuesta = await self._provider.chat(
            [
                ChatMessage(role=ChatRole.SYSTEM, content=prompt),
                ChatMessage(
                    role=ChatRole.USER,
                    content=(
                        "Estás en modo espontáneo: tomas la iniciativa y hablas "
                        "primero. " + instruccion
                    ),
                ),
            ]
        )
        return respuesta.content.strip()

    async def _quizas_hablar(self) -> bool:
        """Intenta decir algo; devuelve True si realmente habló."""
        if not self._en_horario() or not self._presupuesto_ok():
            return False
        if not hay_alguien_escuchando():
            logger.debug("Charla espontánea omitida: nadie escuchando")
            return False

        instruccion, a_marcar = await self._novedad_pendiente()
        if instruccion is None:
            instruccion = self._instruccion_rotativa()

        linea = await self._generar_linea(instruccion)
        if not (10 <= len(linea) <= _LINEA_MAX):
            logger.debug("Línea espontánea descartada por tamaño: %r", linea[:60])
            return False

        enviados = await hablar_a_todas(linea, self._tts)
        if not enviados:
            return False

        if a_marcar:
            self._memory.mark_shared(a_marcar)
        self._dichas.append(linea)
        self._habladas_en_hora.append(time.monotonic())
        logger.info("Comentario espontáneo: %r", linea[:80])
        return True
