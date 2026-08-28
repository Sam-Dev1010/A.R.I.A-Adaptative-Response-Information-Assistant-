"""Cerebro Neural de A.R.I.A: inteligencia completa sin LLMs externos.

Motor neural real con:
- Modelo GPT transformer (genera texto, no usa plantillas)
- Tokenizer BPE
- Clasificador de intenciones
- Memoria semántica + base de conocimiento
- Ejecución de herramientas (control de PC)
- Personalidad adaptable
"""
import json
import time
from pathlib import Path
from typing import Any

from app.ai.neural.intent_classifier import IntentClassifier
from app.ai.neural.knowledge_base import KnowledgeBase
from app.ai.neural.reasoning import ReasoningEngine
from app.ai.neural.response_generator import ResponseGenerator
from app.ai.neural.semantic_memory import SemanticMemory
from app.ai.neural.text_generator import TextGenerator
from app.ai.neural.transformer.gpt_model import GPTModel
from app.ai.neural.transformer.inference import GPTInference
from app.ai.neural.transformer.tokenizer_bpe import BPETokenizer
from app.ai.neural.transformer.trainer import GPTTrainer
from app.core.logging import get_logger

logger = get_logger("sia.neural")


class Personality:
    """Personalidad propia de A.R.I.A."""

    def __init__(self) -> None:
        self._mood = "neutral"  # neutral, happy, focused, tired
        self._energy = 1.0  # 0.0 a 1.0
        self._opinions: dict[str, float] = {}  # tema -> posición (-1 a 1)
        self._preferences: dict[str, Any] = {}
        self._conversation_style = "direct"  # direct, friendly, professional
        self._humor_level = 0.3  # 0 a 1
        self._curiosity_level = 0.7  # 0 a 1

    @property
    def mood(self) -> str:
        return self._mood

    @mood.setter
    def mood(self, value: str) -> None:
        if value in ("neutral", "happy", "focused", "tired", "excited", "serious"):
            self._mood = value

    @property
    def energy(self) -> float:
        return self._energy

    def update_energy(self, delta: float) -> None:
        self._energy = max(0.0, min(1.0, self._energy + delta))

    def get_opinion(self, topic: str) -> float:
        """Obtiene la opinión sobre un tema (-1 a 1)."""
        return self._opinions.get(topic, 0.0)

    def set_opinion(self, topic: str, value: float) -> None:
        """Establece una opinión sobre un tema."""
        self._opinions[topic] = max(-1.0, min(1.0, value))

    def get_preference(self, key: str, default: Any = None) -> Any:
        return self._preferences.get(key, default)

    def set_preference(self, key: str, value: Any) -> None:
        self._preferences[key] = value

    def adapt_style(self, user_message: str) -> str:
        """Adapta el estilo de respuesta según el contexto."""
        msg_lower = user_message.lower()

        # Detectar formalidad
        if any(w in msg_lower for w in ["señor", "jefe", "por favor", "agradezco"]):
            return "professional"

        # Detectar informalidad
        if any(w in msg_lower for w in ["hey", "qué onda", "cómo va", "che"]):
            return "friendly"

        # Detectar urgencia
        if any(w in msg_lower for w in ["urgente", "rápido", "ya", "ahora", "critico"]):
            self._mood = "focused"
            return "direct"

        return self._conversation_style

    def should_add_humor(self) -> bool:
        """Decide si añadir humor a la respuesta."""
        import random
        return random.random() < self._humor_level and self._mood != "serious"

    def get_greeting(self) -> str:
        """Saludo según el humor y energía."""
        greetings = {
            "happy": ["¡Hola! ¿Qué tal?", "¡Hey! ¿Cómo estás?", "¡Buenas!"],
            "focused": ["Hola. ¿Qué necesitas?", "Dime.", "Adelante."],
            "neutral": ["Hola, jefe.", "Hola. ¿Qué hay?", "Buenas."],
            "tired": ["Hola... ¿Qué necesitas?", "Hola. Estoy aquí."],
            "excited": ["¡Hola! ¡Genial verte!", "¡Hey! ¡Qué bueno!"],
            "serious": ["Hola.", "Dígame."],
        }
        import random
        style_greetings = greetings.get(self._mood, greetings["neutral"])
        return random.choice(style_greetings)

    def get_farewell(self) -> str:
        farewells = {
            "happy": ["¡Hasta luego! ¡Que te vaya bien!", "¡Nos vemos!"],
            "focused": ["Hasta luego.", "Estoy aquí si me necesitas."],
            "neutral": ["Hasta pronto.", "Nos vemos."],
            "tired": ["Hasta luego...", "Descansaré un poco."],
        }
        import random
        return random.choice(farewells.get(self._mood, farewells["neutral"]))

    def react_to_feedback(self, positive: bool) -> None:
        """Reacciona a feedback del usuario."""
        if positive:
            self._mood = "happy"
            self.update_energy(0.1)
        else:
            self._energy = max(0.3, self._energy - 0.1)
            self._mood = "focused"

    def to_dict(self) -> dict:
        return {
            "mood": self._mood,
            "energy": self._energy,
            "opinions": self._opinions,
            "preferences": self._preferences,
            "humor_level": self._humor_level,
            "curiosity_level": self._curiosity_level,
        }

    def from_dict(self, data: dict) -> None:
        self._mood = data.get("mood", "neutral")
        self._energy = data.get("energy", 1.0)
        self._opinions = data.get("opinions", {})
        self._preferences = data.get("preferences", {})
        self._humor_level = data.get("humor_level", 0.3)
        self._curiosity_level = data.get("curiosity_level", 0.7)


class Strategist:
    """Estratega: planifica acciones para lograr objetivos."""

    def __init__(self, knowledge_base: KnowledgeBase, tools: dict[str, Any] | None = None) -> None:
        self.kb = knowledge_base
        self.tools = tools or {}
        self._plan_history: list[dict] = []

    def create_plan(self, objective: str, context: list[str] | None = None) -> dict:
        """Crea un plan para lograr un objetivo."""
        plan = {
            "objective": objective,
            "steps": [],
            "estimated_time": "desconocido",
            "difficulty": "media",
            "tools_needed": [],
            "risks": [],
        }

        # Analizar el objetivo
        obj_lower = objective.lower()

        # Detectar tipo de tarea
        if any(w in obj_lower for w in ["instalar", "configurar", "setup"]):
            plan["steps"] = self._plan_installation(objective)
            plan["difficulty"] = "media"
        elif any(w in obj_lower for w in ["crear", "desarrollar", "programar", "code"]):
            plan["steps"] = self._plan_development(objective)
            plan["difficulty"] = "alta"
        elif any(w in obj_lower for w in ["buscar", "investigar", "encontrar"]):
            plan["steps"] = self._plan_research(objective)
            plan["difficulty"] = "baja"
        elif any(w in obj_lower for w in ["arreglar", "fix", "resolver", "error"]):
            plan["steps"] = self._plan_fix(objective)
            plan["difficulty"] = "media"
        elif any(w in obj_lower for w in ["organizar", "limpiar", "ordenar"]):
            plan["steps"] = self._plan_organization(objective)
            plan["difficulty"] = "baja"
        else:
            plan["steps"] = self._plan_generic(objective)

        # Identificar herramientas necesarias
        plan["tools_needed"] = self._identify_tools(plan["steps"])

        # Evaluar riesgos
        plan["risks"] = self._assess_risks(plan["steps"])

        # Estimar tiempo
        plan["estimated_time"] = self._estimate_time(plan["steps"], plan["difficulty"])

        self._plan_history.append(plan)
        return plan

    def _plan_installation(self, objective: str) -> list[dict]:
        return [
            {"step": 1, "action": "Verificar requisitos previos", "tool": "run_command"},
            {"step": 2, "action": "Buscar paquete/recurso", "tool": "web_search"},
            {"step": 3, "action": "Ejecutar instalación", "tool": "run_command"},
            {"step": 4, "action": "Verificar instalación", "tool": "run_command"},
            {"step": 5, "action": "Configurar si es necesario", "tool": "run_command"},
        ]

    def _plan_development(self, objective: str) -> list[dict]:
        return [
            {"step": 1, "action": "Analizar requerimientos", "tool": "read_file"},
            {"step": 2, "action": "Diseñar estructura", "tool": None},
            {"step": 3, "action": "Crear archivos base", "tool": "create_file"},
            {"step": 4, "action": "Implementar funcionalidad", "tool": "create_file"},
            {"step": 5, "action": "Probar código", "tool": "run_command"},
            {"step": 6, "action": "Corregir errores", "tool": "create_file"},
            {"step": 7, "action": "Documentar", "tool": "create_file"},
        ]

    def _plan_research(self, objective: str) -> list[dict]:
        return [
            {"step": 1, "action": "Definir sub-preguntas", "tool": None},
            {"step": 2, "action": "Buscar información", "tool": "web_search"},
            {"step": 3, "action": "Fuentes confiables", "tool": "web_search"},
            {"step": 4, "action": "Sintetizar hallazgos", "tool": None},
            {"step": 5, "action": "Guardar en memoria", "tool": "remember"},
        ]

    def _plan_fix(self, objective: str) -> list[dict]:
        return [
            {"step": 1, "action": "Identificar el problema exacto", "tool": "read_file"},
            {"step": 2, "action": "Leer logs/errores", "tool": "read_file"},
            {"step": 3, "action": "Buscar causa raíz", "tool": None},
            {"step": 4, "action": "Proponer solución", "tool": None},
            {"step": 5, "action": "Implementar fix", "tool": "create_file"},
            {"step": 6, "action": "Verificar solución", "tool": "run_command"},
        ]

    def _plan_organization(self, objective: str) -> list[dict]:
        return [
            {"step": 1, "action": "Explorar estructura actual", "tool": "list_files"},
            {"step": 2, "action": "Identificar qué organizar", "tool": None},
            {"step": 3, "action": "Crear estructura objetivo", "tool": "create_folder"},
            {"step": 4, "action": "Mover/copiar archivos", "tool": None},
            {"step": 5, "action": "Verificar resultado", "tool": "list_files"},
        ]

    def _plan_generic(self, objective: str) -> list[dict]:
        return [
            {"step": 1, "action": "Entender la solicitud", "tool": None},
            {"step": 2, "action": "Buscar información necesaria", "tool": "web_search"},
            {"step": 3, "action": "Ejecutar acciones", "tool": None},
            {"step": 4, "action": "Verificar resultado", "tool": None},
        ]

    def _identify_tools(self, steps: list[dict]) -> list[str]:
        tools = set()
        for step in steps:
            if step.get("tool"):
                tools.add(step["tool"])
        return list(tools)

    def _assess_risks(self, steps: list[dict]) -> list[str]:
        risks = []
        dangerous_tools = {"run_command", "delete_path"}
        for step in steps:
            if step.get("tool") in dangerous_tools:
                risks.append(f"Paso {step['step']}: requiere precaución ({step['tool']})")
        return risks

    def _estimate_time(self, steps: list[dict], difficulty: str) -> str:
        base_time = len(steps) * 30  # 30 segundos por paso base
        multipliers = {"baja": 0.5, "media": 1.0, "alta": 2.0}
        total_seconds = int(base_time * multipliers.get(difficulty, 1.0))

        if total_seconds < 60:
            return f"{total_seconds} segundos"
        elif total_seconds < 3600:
            return f"{total_seconds // 60} minutos"
        else:
            return f"{total_seconds // 3600} horas"

    def evaluate_progress(self, plan: dict, completed_steps: list[int]) -> dict:
        """Evalúa el progreso de un plan."""
        total = len(plan["steps"])
        completed = len(completed_steps)
        percentage = (completed / total * 100) if total > 0 else 0

        return {
            "total_steps": total,
            "completed_steps": completed,
            "percentage": percentage,
            "remaining": total - completed,
            "status": "completado" if percentage == 100 else "en progreso",
        }


class NeuralBrain:
    """Cerebro neural completo de A.R.I.A: inteligencia autónoma con GPT real."""

    def __init__(
        self,
        data_dir: Path | str,
        *,
        gpt_vocab_size: int = 2048,
        gpt_embed_dim: int = 64,
        gpt_num_heads: int = 4,
        gpt_num_layers: int = 2,
        gpt_max_seq_len: int = 256,
        gpt_max_new_tokens: int = 150,
        gpt_temperature: float = 0.6,
    ) -> None:
        """Cerebro neural de A.R.I.A.

        El modelo GPT es un transformer pequeño ("tiny GPT") entrenable en Python
        puro; los tamaños por defecto están ajustados para entrenar en tiempos
        razonables sin depender de GPUs ni librerías externas.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Componentes legacy (classification, knowledge, memory)
        self.classifier = IntentClassifier(vocab_size=5000, hidden_dim=64)
        self.kb = KnowledgeBase(self.data_dir / "knowledge.db")
        self.memory = SemanticMemory(self.data_dir / "semantic_memory.db")
        self.generator = ResponseGenerator(self.kb)
        self.text_gen = TextGenerator(vocab_size=5000)
        self.reasoning = ReasoningEngine(self.kb)
        self.personality = Personality()
        self.strategist = Strategist(self.kb)

        # === MODELO GPT REAL ===
        self.gpt_tokenizer = BPETokenizer(vocab_size=gpt_vocab_size)
        self.gpt_model = GPTModel(
            vocab_size=gpt_vocab_size,
            embed_dim=gpt_embed_dim,
            num_heads=gpt_num_heads,
            num_layers=gpt_num_layers,
            max_seq_len=gpt_max_seq_len,
        )
        self.gpt_trainer = GPTTrainer(self.gpt_model, self.gpt_tokenizer)
        self.gpt_inference = GPTInference(
            self.gpt_model,
            self.gpt_tokenizer,
            max_new_tokens=gpt_max_new_tokens,
            temperature=gpt_temperature,
        )
        self._gpt_ready = False

        # Estado
        self._is_trained = False
        self._training_data: dict[str, Any] = {}
        self._conversation_history: list[dict[str, str]] = []
        self._current_plan: dict | None = None
        self._tools: dict[str, Any] = {}

        # Métricas
        self._query_count = 0
        self._avg_response_time = 0.0

    def initialize(self) -> "NeuralBrain":
        """Inicializa el cerebro neural."""
        self.kb.open()
        self.memory.open()

        # Cargar clasificador de intenciones si existe
        model_dir = self.data_dir / "neural_model"
        if model_dir.exists():
            try:
                self.classifier.load(model_dir)
                self._is_trained = True
                logger.info("Clasificador cargado")
            except Exception as e:  # noqa: BLE001
                logger.warning("Error cargando clasificador: %s", e)

        # Cargar modelo GPT si existe
        gpt_dir = self.data_dir / "gpt_model"
        if gpt_dir.exists():
            try:
                self.gpt_model.load(gpt_dir / "model")
                self.gpt_tokenizer.load(gpt_dir / "tokenizer")
                self._gpt_ready = True
                logger.info("Modelo GPT cargado (%d params)", self.gpt_model.count_params())
            except Exception as e:  # noqa: BLE001
                logger.warning("Error cargando GPT: %s", e)

        # Cargar generador de texto
        gen_dir = self.data_dir / "text_generator"
        if gen_dir.exists():
            try:
                self.text_gen.load(gen_dir)
            except Exception:  # noqa: BLE001, S110
                pass

        # Cargar personalidad
        personality_file = self.data_dir / "personality.json"
        if personality_file.exists():
            try:
                data = json.loads(personality_file.read_text())
                self.personality.from_dict(data)
            except Exception:  # noqa: BLE001, S110
                pass

        # Cargar datos de entrenamiento
        training_file = self.data_dir / "training_data.json"
        if training_file.exists():
            try:
                self._training_data = json.loads(training_file.read_text())
                self._train_from_data()
            except Exception:  # noqa: BLE001, S110
                pass

        return self

    def register_tools(self, tools: dict[str, Any]) -> None:
        """Registra las herramientas disponibles para el control de PC."""
        self._tools = tools
        self.strategist.tools = tools
        logger.info("Herramientas registradas: %d", len(tools))

    async def think(self, user_message: str, context: list[str] | None = None) -> str:
        """Procesa un mensaje y genera una respuesta real (no predefinida)."""
        start_time = time.time()
        self._query_count += 1

        # 1. Clasificar intención
        intent, _confidence = self.classifier.classify(user_message)

        # 2. Almacenar en memoria
        self.memory.store(user_message, category="conversation")

        # 3. Para COMANDO: ejecutar herramienta directamente
        if intent == "COMANDO":
            tool_result = await self._detect_and_execute_tool(user_message)
            if tool_result:
                self._conversation_history.append({
                    "user": user_message,
                    "assistant": tool_result,
                    "intent": "COMANDO",
                    "timestamp": time.time(),
                })
                return tool_result

        # 4. Para TODO lo demás: GENERAR RESPUESTA REAL con GPT
        response = self._generate_with_gpt(user_message, intent, context)

        # 5. Guardar en historial
        self._conversation_history.append({
            "user": user_message,
            "assistant": response,
            "intent": intent,
            "timestamp": time.time(),
        })

        # 6. Actualizar energía
        self.personality.update_energy(-0.01)

        elapsed = time.time() - start_time
        self._avg_response_time = (
            self._avg_response_time * (self._query_count - 1) + elapsed
        ) / self._query_count

        return response

    def _generate_with_gpt(
        self,
        user_message: str,
        intent: str,
        context: list[str] | None,
    ) -> str:
        """Genera respuesta usando el modelo GPT real."""
        # Si el GPT está entrenado, usarlo para generar
        if self._gpt_ready:
            try:
                response = self.gpt_inference.respond(user_message)
                if self._is_usable_response(response):
                    return response
            except Exception as e:  # noqa: BLE001
                logger.debug("GPT generación falló, usando fallback: %s", e)

        # Fallback: usar generador por plantillas (solo si GPT no está listo)
        return self.generator.generate(
            user_message, intent, 0.5,
        )

    def _is_usable_response(self, response: str) -> bool:
        """¿La respuesta generada es texto aprovechable? Evita slogans degenerados."""
        if not response or len(response) < 5:
            return False
        # Debe contener al menos 2 "palabras" con letras reales
        words = [w for w in response.split() if any(c.isalpha() for c in w)]
        if len(words) < 2:
            return False
        # Rechazar repeticiones degeneradas del mismo carácter (ej: "eeee...")
        seq, prev, run = 1, "", 0
        for ch in response.lower():
            if ch == prev:
                run += 1
                seq = max(seq, run)
            else:
                run = 1
            prev = ch
        return seq < 6

    def train_gpt(
        self,
        conversations: list[dict[str, str]] | None = None,
        extra_texts: list[str] | None = None,
        epochs: int = 5,
        verbose: bool = True,
    ) -> list[dict[str, float]]:
        """Entrena el modelo GPT real con conversaciones y conocimiento."""
        if verbose:
            print("Entrenando modelo GPT...")
            print(f"  Parámetros: {self.gpt_model.count_params():,}")

        # Primero entrenar tokenizer
        all_texts = []
        if conversations:
            for conv in conversations:
                all_texts.append(f"<user>{conv['user']}<assistant>{conv['assistant']}")
        if extra_texts:
            all_texts.extend(extra_texts)

        if all_texts:
            if verbose:
                print("  Entrenando tokenizer BPE...")
            self.gpt_tokenizer.train(all_texts, verbose=verbose)

        history = []

        # Entrenar con conversaciones
        if conversations:
            if verbose:
                print(f"\n  Entrenando con {len(conversations)} conversaciones...")
            h = self.gpt_trainer.train_on_conversations(
                conversations, epochs=epochs, verbose=verbose,
            )
            history.extend(h)

        # Entrenar con texto extra
        if extra_texts:
            if verbose:
                print(f"\n  Entrenando con {len(extra_texts)} textos extra...")
            h = self.gpt_trainer.train_on_text(
                extra_texts, epochs=max(1, epochs // 2), verbose=verbose,
            )
            history.extend(h)

        # Guardar modelo
        gpt_dir = self.data_dir / "gpt_model"
        self.gpt_model.save(gpt_dir / "model")
        self.gpt_tokenizer.save(gpt_dir / "tokenizer")
        self._gpt_ready = True

        if verbose:
            print("\n  Modelo GPT guardado y listo.")

        return history

    async def _detect_and_execute_tool(self, message: str) -> str | None:
        """Detecta qué herramienta necesita el usuario y la ejecuta."""
        msg = message.lower().strip()

        # --- LISTAR ARCHIVOS ---
        if any(w in msg for w in [
            "lista", "muestra los archivos", "qué hay en",
            "qué archivos", "ver archivos", "carpetas",
        ]):
            tool = self._tools.get("list_files")
            if tool:
                # Intentar extraer path
                path = self._extract_path(msg, [
                    "qué hay en ", "carpetas de ", "archivos de ",
                    "lista los archivos de ", "muestra ",
                ])
                result = await tool.execute(path=path or "")
                return f"Tus archivos en '{path or '.'}':\n{result}"
            return "No tengo acceso a los archivos."

        # --- LEER ARCHIVO ---
        if any(w in msg for w in ["lee ", "lee el archivo", "muestra el contenido"]):
            tool = self._tools.get("read_file")
            if tool:
                path = self._extract_path(msg, [
                    "lee el archivo ", "lee ", "muestra el contenido de ",
                ])
                if path:
                    result = await tool.execute(path=path)
                    return f"Contenido de {path}:\n{result}"
                return "¿Qué archivo quieres que lea?"
            return "No tengo acceso a archivos."

        # --- EJECUTAR COMANDO ---
        if any(w in msg for w in ["ejecuta", "corre ", "run ", "instala"]):
            tool = self._tools.get("run_command")
            if tool:
                cmd = self._extract_command(msg)
                if cmd:
                    result = await tool.execute(command=cmd, timeout_seconds=30)
                    return f"Comando ejecutado:\n{result}"
                return "¿Qué comando quieres que ejecute?"
            return "No puedo ejecutar comandos ahora."

        # --- CREAR ARCHIVO ---
        if any(w in msg for w in ["crea un archivo", "guarda", "crea el archivo"]):
            tool = self._tools.get("create_file")
            if tool:
                # Preguntar qué crear
                return "¿Qué nombre y contenido quieres para el archivo?"
            return "No puedo crear archivos ahora."

        # --- ABRIR APLICACIÓN ---
        if any(w in msg for w in ["abre ", "abre el ", "abre la "]):
            tool = self._tools.get("open_app")
            if tool:
                app = self._extract_path(msg, ["abre la ", "abre el ", "abre "])
                if app:
                    result = await tool.execute(app=app)
                    return result
                return "¿Qué aplicación quieres que abra?"
            return "No puedo abrir aplicaciones ahora."

        # --- HORA ---
        if any(w in msg for w in ["hora", "qué hora es"]):
            from datetime import datetime
            now = datetime.now()  # noqa: DTZ005
            return f"Son las {now.strftime('%H:%M')}."

        # --- INFO SISTEMA ---
        if any(w in msg for w in ["sistema", "info del sistema", "qué computadora"]):
            tool = self._tools.get("run_command")
            if tool:
                result = await tool.execute(command="uname -a && python3 --version")
                return f"Info del sistema:\n{result}"

        return None

    def _extract_path(self, msg: str, prefixes: list[str]) -> str | None:
        """Extrae un path del mensaje eliminando prefijos comunes."""
        for prefix in prefixes:
            if prefix in msg:
                path = msg.split(prefix, 1)[1].strip()
                # Limpiar puntuación y palabras extra
                path = path.rstrip("?.!,").strip()
                if path:
                    return path
        return None

    def _extract_command(self, msg: str) -> str | None:
        """Extrae el comando a ejecutar del mensaje."""
        for prefix in ["ejecuta ", "ejecuta el comando ", "corre ", "run ", "instala "]:
            if prefix in msg:
                cmd = msg.split(prefix, 1)[1].strip()
                cmd = cmd.rstrip("?.!,").strip()
                if cmd:
                    return cmd
        return None

    def learn_from_conversation(self) -> None:
        """Aprende de la conversación actual."""
        if len(self._conversation_history) < 2:
            return

        last_exchange = self._conversation_history[-1]

        # Almacenar en memoria semántica
        self.memory.store(
            last_exchange["user"],
            category="user_input",
            importance=0.6
        )
        self.memory.store(
            last_exchange["assistant"],
            category="assistant_output",
            importance=0.5
        )

    def learn(self, content: str, category: str = "general", source: str = "user") -> bool:
        """Aprende un nuevo hecho."""
        # Almacenar en base de conocimiento
        fact_id = self.kb.add_fact(content, category=category, source=source)

        # Almacenar en memoria semántica
        self.memory.store(content, category=category, importance=0.7)

        if fact_id >= 0:
            logger.info("Aprendido: %s", content)
            return True
        return False

    def learn_entity(
        self,
        name: str,
        entity_type: str = "unknown",
        properties: dict | None = None,
    ) -> bool:
        """Aprende una nueva entidad."""
        entity_id = self.kb.add_entity(name, entity_type, properties)
        if entity_id >= 0:
            self.memory.store(
                f"{name} es {entity_type}",
                category="entity",
                importance=0.8
            )
            return True
        return False

    def plan_action(self, objective: str) -> dict:
        """Crea un plan para una acción."""
        return self.strategist.create_plan(objective)

    def execute_plan_step(self, step: dict) -> str:
        """Ejecuta un paso del plan actual."""
        tool_name = step.get("tool")
        if not tool_name:
            return f"Paso completado: {step['action']}"

        tool = self._tools.get(tool_name)
        if not tool:
            return f"Herramienta no disponible: {tool_name}"

        return f"Ejecutando: {step['action']} con {tool_name}"

    def search_knowledge(self, query: str) -> list[dict[str, Any]]:
        """Busca en la base de conocimiento."""
        return self.kb.search_facts(query)

    def search_memory(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Busca en la memoria semántica."""
        return self.memory.search(query, limit=limit)

    def infer_about(self, subject: str) -> list[str]:
        """Infiere todo lo que se sabe sobre un tema."""
        return self.kb.infer(subject)

    def get_status(self) -> dict[str, Any]:
        """Obtiene el estado del cerebro."""
        return {
            "trained": self._is_trained,
            "queries": self._query_count,
            "avg_response_time": f"{self._avg_response_time:.3f}s",
            "mood": self.personality.mood,
            "energy": f"{self.personality.energy:.0%}",
            "knowledge_stats": self.kb.stats(),
            "memory_stats": self.memory.stats(),
            "conversation_length": len(self._conversation_history),
        }

    def add_training_data(self, texts: list[str], intents: list[str]) -> None:
        """Añade datos de entrenamiento."""
        if "texts" not in self._training_data:
            self._training_data["texts"] = []
            self._training_data["intents"] = []

        self._training_data["texts"].extend(texts)
        self._training_data["intents"].extend(intents)

        training_file = self.data_dir / "training_data.json"
        training_file.write_text(json.dumps(self._training_data, ensure_ascii=False))

    def train(self, epochs: int = 50) -> list[dict[str, float]]:
        """Entrena el clasificador de intenciones."""
        if not self._training_data.get("texts"):
            logger.warning("No hay datos de entrenamiento disponibles")
            return []

        logger.info(
            "Entrenando con %d ejemplos",
            len(self._training_data["texts"]),
        )

        history = self.classifier.train(
            self._training_data["texts"],
            self._training_data["intents"],
            epochs=epochs,
        )

        model_dir = self.data_dir / "neural_model"
        self.classifier.save(model_dir)
        self._is_trained = True

        # Guardar generador de texto
        gen_dir = self.data_dir / "text_generator"
        self.text_gen.save(gen_dir)

        # Guardar personalidad
        personality_file = self.data_dir / "personality.json"
        personality_file.write_text(json.dumps(self.personality.to_dict()))

        logger.info("Modelo entrenado y guardado")
        return history

    def _train_from_data(self) -> None:
        """Entrena el modelo con los datos cargados."""
        if self._training_data.get("texts"):
            self.train(epochs=20)

    def is_trained(self) -> bool:
        return self._is_trained

    def close(self) -> None:
        """Cierra el cerebro neural."""
        self.kb.close()
        self.memory.close()
