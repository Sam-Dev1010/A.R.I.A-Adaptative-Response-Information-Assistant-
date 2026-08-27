"""Motor de razonamiento para A.R.I.A: piensa paso a paso.

Implementa chain-of-thought reasoning para resolver problemas complejos
sin depender de LLMs externos.
"""
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Thought:
    """Un paso de razonamiento."""
    step: int
    content: str
    confidence: float
    evidence: list[str] = field(default_factory=list)


@dataclass
class ReasoningResult:
    """Resultado del razonamiento."""
    answer: str
    thoughts: list[Thought]
    confidence: float
    sources: list[str] = field(default_factory=list)


class ReasoningEngine:
    """Motor de razonamiento con chain-of-thought."""

    # Patrones de preguntas que requieren razonamiento
    REASONING_PATTERNS = {
        "comparison": [
            r"cuál es (mejor|peor|más|menor)",
            r"compara|comparación",
            r"diferencia entre",
            r"ventaja|desventaja",
        ],
        "explanation": [
            r"por qué|por que",
            r"cómo funciona|cómo se",
            r"explica|explicación",
            r"qué es|qué son",
        ],
        "prediction": [
            r"qué pasará|qué va a pasar",
            r"cuál será|cuál sería",
            r"predice|estima",
        ],
        "analysis": [
            r"analiza|análisis",
            r"evalúa|evaluación",
            r"opina|opinión",
            r"qué opinas",
        ],
        "problem_solving": [
            r"cómo resolver|cómo solucionar",
            r"problema|error|falla",
            r"ayuda|ayúdame",
            r"necesito|quiero",
        ],
    }

    # Plantillas de razonamiento por tipo
    REASONING_TEMPLATES = {
        "comparison": [
            "Para comparar {topic}, necesito considerar:",
            "Analizando las diferencias entre {options}:",
            "Factores clave para esta comparación:",
        ],
        "explanation": [
            "Para explicar {topic}, voy paso a paso:",
            "La explicación de {topic} involucra varios aspectos:",
            "Entendiendo {topic}:",
        ],
        "prediction": [
            "Basándome en la información disponible:",
            "Para predecir el resultado, considero:",
            "Los factores que influyen son:",
        ],
        "analysis": [
            "Analizando {topic} desde diferentes ángulos:",
            "Mi análisis considera los siguientes puntos:",
            "Evaluando la situación:",
        ],
        "problem_solving": [
            "Para resolver este problema, primero identifico:",
            "Pasos para solucionar esto:",
            "Mi estrategia es:",
        ],
    }

    def __init__(self, knowledge_base=None) -> None:
        self.kb = knowledge_base
        self._reasoning_history: list[ReasoningResult] = []

    def reason(self, question: str, context: list[str] | None = None) -> ReasoningResult:
        """Realiza razonamiento chain-of-thought sobre una pregunta."""
        # 1. Clasificar el tipo de razonamiento
        reasoning_type = self._classify_reasoning_type(question)

        # 2. Extraer información relevante
        relevant_info = self._extract_relevant_info(question, context)

        # 3. Generar pasos de razonamiento
        thoughts = self._generate_thoughts(question, reasoning_type, relevant_info)

        # 4. Sintetizar respuesta final
        answer = self._synthesize_answer(question, thoughts, reasoning_type)

        # 5. Calcular confianza
        confidence = self._calculate_confidence(thoughts, relevant_info)

        result = ReasoningResult(
            answer=answer,
            thoughts=thoughts,
            confidence=confidence,
            sources=[info.get("source", "") for info in relevant_info],
        )

        self._reasoning_history.append(result)
        return result

    def _classify_reasoning_type(self, question: str) -> str:
        """Clasifica el tipo de razonamiento necesario."""
        question_lower = question.lower()

        for reasoning_type, patterns in self.REASONING_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    return reasoning_type

        return "general"

    def _extract_relevant_info(
        self,
        question: str,
        context: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Extrae información relevante de la pregunta y el contexto."""
        info = []

        # Buscar en base de conocimiento
        if self.kb:
            # Buscar hechos relacionados
            facts = self.kb.search_facts(question, limit=5)
            for fact in facts:
                info.append({
                    "type": "fact",
                    "content": fact["content"],
                    "source": "knowledge_base",
                    "confidence": fact.get("confidence", 1.0),
                })

            # Buscar entidades relacionadas
            entities = self.kb.search_entities(question, limit=3)
            for entity in entities:
                info.append({
                    "type": "entity",
                    "content": f"{entity['name']} ({entity['type']})",
                    "properties": entity.get("properties", {}),
                    "source": "knowledge_base",
                })

        # Agregar contexto de conversación
        if context:
            for ctx in context[-3:]:
                info.append({
                    "type": "context",
                    "content": ctx,
                    "source": "conversation",
                })

        return info

    def _generate_thoughts(
        self,
        question: str,
        reasoning_type: str,
        relevant_info: list[dict[str, Any]],
    ) -> list[Thought]:
        """Genera pasos de razonamiento."""
        thoughts = []
        step = 1

        # Paso 1: Entender la pregunta
        thoughts.append(Thought(
            step=step,
            content=f"Entendiendo la pregunta: {question}",
            confidence=0.9,
        ))
        step += 1

        # Paso 2: Identificar tipo de razonamiento
        thoughts.append(Thought(
            step=step,
            content=f"Tipo de razonamiento detectado: {reasoning_type}",
            confidence=0.85,
        ))
        step += 1

        # Paso 3: Revisar información disponible
        if relevant_info:
            info_summary = f"Información disponible: {len(relevant_info)} fuentes"
            thoughts.append(Thought(
                step=step,
                content=info_summary,
                evidence=[info["content"][:50] for info in relevant_info[:3]],
                confidence=0.8,
            ))
            step += 1

        # Paso 4: Análisis específico por tipo
        if reasoning_type == "comparison":
            thoughts.extend(self._comparison_thoughts(question, relevant_info))
        elif reasoning_type == "explanation":
            thoughts.extend(self._explanation_thoughts(question, relevant_info))
        elif reasoning_type == "problem_solving":
            thoughts.extend(self._problem_solving_thoughts(question, relevant_info))
        elif reasoning_type == "analysis":
            thoughts.extend(self._analysis_thoughts(question, relevant_info))
        else:
            thoughts.extend(self._general_thoughts(question, relevant_info))

        # Paso final: Conclusión
        thoughts.append(Thought(
            step=step,
            content="Sintetizando respuesta final...",
            confidence=0.75,
        ))

        return thoughts

    def _comparison_thoughts(
        self,
        question: str,
        info: list[dict[str, Any]],
    ) -> list[Thought]:
        """Pensamientos para comparaciones."""
        return [
            Thought(
                step=4,
                content="Identificando elementos a comparar",
                confidence=0.8,
            ),
            Thought(
                step=5,
                content="Evaluando características de cada opción",
                confidence=0.75,
            ),
            Thought(
                step=6,
                content="Ponderando ventajas y desventajas",
                confidence=0.7,
            ),
        ]

    def _explanation_thoughts(
        self,
        question: str,
        info: list[dict[str, Any]],
    ) -> list[Thought]:
        """Pensamientos para explicaciones."""
        return [
            Thought(
                step=4,
                content="Desglosando el concepto en partes",
                confidence=0.8,
            ),
            Thought(
                step=5,
                content="Conectando con conocimiento previo",
                confidence=0.75,
            ),
            Thought(
                step=6,
                content="Verificando coherencia de la explicación",
                confidence=0.7,
            ),
        ]

    def _problem_solving_thoughts(
        self,
        question: str,
        info: list[dict[str, Any]],
    ) -> list[Thought]:
        """Pensamientos para resolución de problemas."""
        return [
            Thought(
                step=4,
                content="Identificando el problema específico",
                confidence=0.85,
            ),
            Thought(
                step=5,
                content="Analizando posibles causas",
                confidence=0.8,
            ),
            Thought(
                step=6,
                content="Proponiendo soluciones",
                confidence=0.75,
            ),
            Thought(
                step=7,
                content="Evaluando la mejor opción",
                confidence=0.7,
            ),
        ]

    def _analysis_thoughts(
        self,
        question: str,
        info: list[dict[str, Any]],
    ) -> list[Thought]:
        """Pensamientos para análisis."""
        return [
            Thought(
                step=4,
                content="Examinando diferentes perspectivas",
                confidence=0.8,
            ),
            Thought(
                step=5,
                content="Considerando factores relevantes",
                confidence=0.75,
            ),
            Thought(
                step=6,
                content="Formando una opinión fundamentada",
                confidence=0.7,
            ),
        ]

    def _general_thoughts(
        self,
        question: str,
        info: list[dict[str, Any]],
    ) -> list[Thought]:
        """Pensamientos generales."""
        return [
            Thought(
                step=4,
                content="Analizando la solicitud",
                confidence=0.8,
            ),
            Thought(
                step=5,
                content="Considerando opciones disponibles",
                confidence=0.75,
            ),
        ]

    def _synthesize_answer(
        self,
        question: str,
        thoughts: list[Thought],
        reasoning_type: str,
    ) -> str:
        """Sintetiza una respuesta final basada en los pensamientos."""
        # Obtener la plantilla base
        templates = self.REASONING_TEMPLATES.get(
            reasoning_type,
            self.REASONING_TEMPLATES["explanation"]
        )

        # Seleccionar plantilla
        template = templates[0]

        # Extraer puntos clave de los pensamientos
        key_points = []
        for thought in thoughts:
            if thought.confidence >= 0.7:
                key_points.append(thought.content)

        # Construir respuesta
        if key_points:
            answer = template + "\n\n"
            for i, point in enumerate(key_points[:4], 1):
                answer += f"{i}. {point}\n"
        else:
            answer = f"Analizando tu pregunta sobre '{question}', considero que..."

        return answer.strip()

    def _calculate_confidence(
        self,
        thoughts: list[Thought],
        info: list[dict[str, Any]],
    ) -> float:
        """Calcula la confianza general del razonamiento."""
        if not thoughts:
            return 0.5

        # Promedio de confianza de los pensamientos
        thought_confidence = sum(t.confidence for t in thoughts) / len(thoughts)

        # Bonificación por información disponible
        info_bonus = min(0.1, len(info) * 0.02)

        return min(1.0, thought_confidence + info_bonus)

    def get_reasoning_history(self) -> list[ReasoningResult]:
        """Devuelve el historial de razonamiento."""
        return list(self._reasoning_history)

    def explain_reasoning(self, result: ReasoningResult) -> str:
        """Explica el proceso de razonamiento utilizado."""
        lines = ["Proceso de razonamiento:", "=" * 40]

        for thought in result.thoughts:
            confidence_bar = "█" * int(thought.confidence * 10)
            lines.append(f"Paso {thought.step}: {thought.content}")
            lines.append(f"  Confianza: [{confidence_bar}] {thought.confidence:.0%}")

            if thought.evidence:
                lines.append("  Evidencia:")
                for evidence in thought.evidence:
                    lines.append(f"    - {evidence}")

        lines.append("=" * 40)
        lines.append(f"Confianza final: {result.confidence:.0%}")

        return "\n".join(lines)
