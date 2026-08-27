"""Generador de respuestas para A.R.I.A: combina conocimiento + personalidad.

Genera respuestas coherentes sin depender de LLMs externos.
"""
import random
from typing import Any

from app.ai.neural.knowledge_base import KnowledgeBase


class ResponseGenerator:
    """Genera respuestas basadas en conocimiento y contexto."""

    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self.kb = knowledge_base
        self._conversation_context: list[str] = []

    def generate(
        self,
        user_message: str,
        intent: str,
        confidence: float,
        facts: list[str] | None = None,
    ) -> str:
        """Genera una respuesta basada en la intención y el conocimiento."""
        # Añadir al contexto
        self._conversation_context.append(user_message)
        if len(self._conversation_context) > 5:
            self._conversation_context.pop(0)

        # Buscar en la base de conocimiento
        kb_facts = self.kb.search_facts(user_message, limit=3)
        kb_entities = self.kb.search_entities(user_message, limit=3)

        # Generar respuesta según la intención
        if intent == "SALUDO":
            return self._generate_greeting()
        elif intent == "DESPEDIDA":
            return self._generate_farewell()
        elif intent == "AGRADECIMIENTO":
            return self._generate_thanks()
        elif intent == "PREGUNTA":
            return self._generate_answer(user_message, kb_facts, kb_entities, facts)
        elif intent == "COMANDO":
            return self._generate_command_response(user_message, kb_facts)
        elif intent == "QUEJA":
            return self._generate_empathy()
        elif intent == "CURIOSIDAD":
            return self._generate_curiosity(user_message, kb_facts)
        else:  # CHAT
            return self._generate_chat(user_message, kb_facts)

    def _generate_greeting(self) -> str:
        """Genera un saludo según la hora."""
        from datetime import datetime
        hour = datetime.now().hour

        if 5 <= hour < 12:
            saludos = [
                "Buenos días, jefe. ¿En qué puedo ayudarle?",
                "Hola, buen día. ¿Qué necesita?",
                "Buenos días. Estoy lista para lo que necesite.",
            ]
        elif 12 <= hour < 20:
            saludos = [
                "Buenas tardes, jefe. ¿Qué hay de nuevo?",
                "Hola, buenas tardes. ¿Cómo puedo asistirle?",
                "Buenas tardes. ¿En qué puedo ayudarle hoy?",
            ]
        else:
            saludos = [
                "Buenas noches, jefe. ¿Trabajando tarde?",
                "Hola, buenas noches. ¿Necesita algo?",
                "Buenas noches. Estoy aquí para lo que necesite.",
            ]

        return random.choice(saludos)

    def _generate_farewell(self) -> str:
        """Genera una despedida."""
        despedidas = [
            "Hasta luego, jefe. Estaré aquí cuando me necesite.",
            "Nos vemos. Llámeme si necesita algo.",
            "Hasta pronto. Que le vaya bien.",
        ]
        return random.choice(despedidas)

    def _generate_thanks(self) -> str:
        """Genera una respuesta a agradecimiento."""
        respuestas = [
            "De nada, jefe. Para eso estoy.",
            "A su servicio. ¿Necesita algo más?",
            "No hay de qué. Cualquier cosa, avíseme.",
        ]
        return random.choice(respuestas)

    def _generate_answer(
        self,
        question: str,
        kb_facts: list[dict[str, Any]],
        kb_entities: list[dict[str, Any]],
        extra_facts: list[str] | None = None,
    ) -> str:
        """Genera una respuesta a una pregunta."""
        # Si hay hechos en la base de conocimiento
        if kb_facts:
            fact = kb_facts[0]
            return f"Según mi conocimiento: {fact['content']}"

        # Si hay entidades relacionadas
        if kb_entities:
            entity = kb_entities[0]
            props = entity.get("properties", {})
            if props:
                props_str = ", ".join(f"{k}: {v}" for k, v in props.items())
                return f"De {entity['name']} sé que: {props_str}"
            return f"Conozco a {entity['name']} ({entity['type']}), pero no tengo más detalles aún."

        # Si hay hechos externos
        if extra_facts:
            return f"Lo que sé es: {extra_facts[0]}"

        # Respuesta por defecto
        respuestas = [
            "No tengo información sobre eso aún. ¿Puede contarme más?",
            "Esa es buena pregunta. Aún no lo sé, pero puedo investigar.",
            "No tengo datos suficientes. ¿Quiere que busque en internet?",
            "Hmm, no estoy segura. ¿Puede darme más contexto?",
        ]
        return random.choice(respuestas)

    def _generate_command_response(
        self,
        command: str,
        kb_facts: list[dict[str, Any]],
    ) -> str:
        """Genera una respuesta a un comando."""
        return f"Entendido, jefe. Procesando: {command}"

    def _generate_empathy(self) -> str:
        """Genera una respuesta empática."""
        respuestas = [
            "Entiendo su frustración, jefe. ¿Cómo puedo ayudarle a solucionarlo?",
            "Lamento que esté teniendo problemas. ¿Qué puedo hacer para mejorar la situación?",
            "Comprendo. ¿Quiere que intente algo diferente?",
        ]
        return random.choice(respuestas)

    def _generate_curiosity(
        self,
        question: str,
        kb_facts: list[dict[str, Any]],
    ) -> str:
        """Genera una respuesta a curiosidad."""
        if kb_facts:
            return f"¡Buena pregunta! Lo que sé es: {kb_facts[0]['content']}"

        respuestas = [
            "Eso es muy interesante. ¿Quiere que investigue más sobre ello?",
            "Me intriga. Aún no lo sé, pero puedo buscar información.",
            "Buena pregunta. ¿Puedo investigar para darle una mejor respuesta?",
        ]
        return random.choice(respuestas)

    def _generate_chat(
        self,
        message: str,
        kb_facts: list[dict[str, Any]],
    ) -> str:
        """Genera una respuesta de conversación general."""
        if kb_facts:
            return f"Interesante. Algo relacionado que sé: {kb_facts[0]['content']}"

        respuestas = [
            "Entiendo. ¿Hay algo específico en lo que pueda ayudarle?",
            "Veo. ¿Qué más tiene en mente?",
            "Ya. ¿Necesita algo o solo quería comentar?",
        ]
        return random.choice(respuestas)

    def add_to_context(self, message: str) -> None:
        """Añade un mensaje al contexto de conversación."""
        self._conversation_context.append(message)
        if len(self._conversation_context) > 10:
            self._conversation_context.pop(0)

    def clear_context(self) -> None:
        """Limpia el contexto de conversación."""
        self._conversation_context.clear()
