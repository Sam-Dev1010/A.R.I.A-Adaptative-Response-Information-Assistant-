"""Clasificador de intenciones para A.R.I.A: entiende qué quiere el usuario.

Usa una red neuronal para clasificar mensajes en categorías como:
- PREGUNTA: El usuario hace una pregunta
- COMANDO: El usuario pide ejecutar algo
- CHAT: Conversación general
- SALUDO: Saludo
- DESPEDIDA: Despedida
"""
import json
from pathlib import Path
from typing import ClassVar

from app.ai.neural.intent_rules import intent_label
from app.ai.neural.layers import Activation, Dense
from app.ai.neural.network import SequentialNetwork
from app.ai.neural.tokenizer import Tokenizer
from app.ai.neural.trainer import Trainer


class IntentClassifier:
    """Clasificador de intenciones basado en red neuronal."""

    # Intenciones conocidas
    INTENTS: ClassVar[list[str]] = [
        "PREGUNTA",      # El usuario hace una pregunta
        "COMANDO",       # El usuario pide ejecutar algo
        "CHAT",          # Conversación general
        "SALUDO",        # Saludo
        "DESPEDIDA",     # Despedida
        "AGRADECIMIENTO",# Agradecimiento
        "QUEJA",         # Queja o frustración
        "CURIOSIDAD",    # Curiosidad sobre algo
    ]

    def __init__(
        self,
        vocab_size: int = 5000,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        max_seq_len: int = 50,
    ) -> None:
        self.tokenizer = Tokenizer(vocab_size=vocab_size)
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.intent_to_id = {intent: i for i, intent in enumerate(self.INTENTS)}
        self.id_to_intent = {i: intent for intent, i in self.intent_to_id.items()}

        # Tamaño de entrada: 8 features manuales + vocab_size (limitado a 200)
        input_size = 8 + min(vocab_size, 200)

        # Red neuronal: dense -> relu -> dense -> softmax
        self.network = SequentialNetwork()
        self.network.add(Dense(input_size, hidden_dim))
        self.network.add(Activation("relu"))
        self.network.add(Dense(hidden_dim, len(self.INTENTS)))
        self.network.add(Activation("softmax"))

        self.trainer = Trainer(self.network, loss_fn="cross_entropy", learning_rate=0.01)

    def _text_to_features(self, text: str) -> list[float]:
        """Convierte texto a vector de características mejorado.

        Combina bag of words con features manuales para mejor clasificación.
        """
        tokens = self.tokenizer.tokenize(text)
        text_lower = text.lower()

        # Features manuales (8 features)
        manual_features = [
            1.0 if "?" in text else 0.0,          # Es pregunta
            1.0 if "!" in text else 0.0,          # Tiene exclamación
            len(tokens) / 20.0,                    # Longitud normalizada
            1.0 if any(w in text_lower for w in ["abre", "ejecuta", "corre", "instala", "pon", "enciende", "apaga"]) else 0.0,  # Palabras de comando
            1.0 if any(w in text_lower for w in ["hola", "buenos", "buenas", "hey", "saludos"]) else 0.0,  # Saludo
            1.0 if any(w in text_lower for w in ["adiós", "hasta", "bye", "chao", "nos vemos"]) else 0.0,   # Despedida
            1.0 if any(w in text_lower for w in ["gracias", "thanks", "agradezco"]) else 0.0,  # Agradecimiento
            1.0 if any(w in text_lower for w in ["no funciona", "error", "roto", "problema", "molesta"]) else 0.0,  # Queja
        ]

        # Bag of words (vocab_size features)
        bow = [0.0] * min(self.tokenizer.vocab_size, 200)  # Limitar para eficiencia
        for token in tokens:
            if token in self.tokenizer.token_to_id:
                idx = self.tokenizer.token_to_id[token]
                if idx < len(bow):
                    bow[idx] += 1.0
        # Normalizar BOW
        total = sum(bow)
        if total > 0:
            bow = [f / total for f in bow]

        # Combinar features
        return manual_features + bow

    def classify(self, text: str) -> tuple[str, float]:
        """Clasifica un mensaje y devuelve (intención, confianza).

        La red neuronal puede fallar con clases poco representadas (p. ej.
        PREGUNTA con pocos ejemplos), así que las reglas deterministas actúan
        como respaldo cuando la red no está suficientemente convencida.
        """
        features = self._text_to_features(text)
        probabilities = self.network.predict(features)

        max_idx = 0
        max_prob = probabilities[0]
        for i, p in enumerate(probabilities):
            if p > max_prob:
                max_prob = p
                max_idx = i

        neural_intent = self.id_to_intent[max_idx]

        # Las reglas son la fuente de las etiquetas de entrenamiento, así que
        # si coinciden con una clase explícita tienen prioridad; la red solo
        # decide cuando no hay ninguna señal determinista.
        rule_intent = intent_label(text)
        if rule_intent != "CHAT" and rule_intent != neural_intent:
            return rule_intent, 0.85

        return neural_intent, max_prob

    def classify_detailed(self, text: str) -> dict[str, float]:
        """Clasifica y devuelve probabilidades de todas las intenciones."""
        features = self._text_to_features(text)
        probabilities = self.network.predict(features)

        return {
            self.id_to_intent[i]: prob
            for i, prob in enumerate(probabilities)
        }

    def train(
        self,
        texts: list[str],
        intents: list[str],
        epochs: int = 50,
    ) -> list[dict[str, float]]:
        """Entrena el clasificador con datos etiquetados."""
        # Entrenar tokenizer
        self.tokenizer.train(texts)

        # Convertir a features y labels
        X = [self._text_to_features(text) for text in texts]
        y = [self.intent_to_id[intent] for intent in intents]

        # Entrenar red
        return self.trainer.train(X, y, epochs=epochs, validation_split=0.15)

    def evaluate(self, texts: list[str], intents: list[str]) -> dict[str, float]:
        """Evalúa el clasificador."""
        X = [self._text_to_features(text) for text in texts]
        y = [self.intent_to_id[intent] for intent in intents]
        return self.trainer.evaluate(X, y)

    def save(self, directory: Path | str) -> None:
        """Guarda el modelo completo."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        self.network.save(directory / "network.json")
        self.tokenizer.save(directory / "tokenizer.json")

        metadata = {
            "intents": self.INTENTS,
            "vocab_size": self.tokenizer.vocab_size,
            "max_seq_len": self.max_seq_len,
            "hidden_dim": self.hidden_dim,
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2)
        )

    def load(self, directory: Path | str) -> "IntentClassifier":
        """Carga un modelo guardado."""
        directory = Path(directory)

        metadata = json.loads((directory / "metadata.json").read_text())
        self.INTENTS = metadata["intents"]
        self.intent_to_id = {intent: i for i, intent in enumerate(self.INTENTS)}
        self.id_to_intent = {i: intent for intent, i in self.intent_to_id.items()}

        saved_vocab_size = metadata.get("vocab_size", 5000)
        saved_hidden_dim = metadata.get("hidden_dim", 64)
        input_size = 8 + min(saved_vocab_size, 200)
        output_size = len(self.INTENTS)
        self.network = SequentialNetwork()
        self.network.add(Dense(input_size, saved_hidden_dim))
        self.network.add(Activation("relu"))
        self.network.add(Dense(saved_hidden_dim, output_size))
        self.network.add(Activation("softmax"))

        self.tokenizer.load(directory / "tokenizer.json")
        self.network.load(directory / "network.json")

        return self
