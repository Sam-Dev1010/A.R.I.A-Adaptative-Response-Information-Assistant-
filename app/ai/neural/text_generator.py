"""Generador de texto neuronal para A.R.I.A: genera respuestas aprendidas.

Usa una red neuronal secuencial para predecir la siguiente palabra
y generar texto coherente basado en el contexto.
"""
import json
import random
from pathlib import Path
from collections import defaultdict

from app.ai.neural.tokenizer import Tokenizer


class TextGenerator:
    """Generador de texto basado en modelos de lenguaje simple (n-grams + neural)."""

    def __init__(self, vocab_size: int = 5000, context_window: int = 3) -> None:
        self.tokenizer = Tokenizer(vocab_size=vocab_size)
        self.context_window = context_window

        # Modelo de lenguaje: frecuencias de n-gramas
        self._bigrams: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._trigrams: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._responses: dict[str, list[str]] = defaultdict(list)  # patrón -> respuestas
        self._vocab_trained = False

    def train(self, conversations: list[dict[str, str]]) -> None:
        """Entrena el generador con conversaciones.

        conversations: [{"input": "hola", "response": "hola jefe"}, ...]
        """
        all_texts = []
        for conv in conversations:
            all_texts.append(conv["input"])
            all_texts.append(conv["response"])

        self.tokenizer.train(all_texts)
        self._vocab_trained = True

        # Entrenar n-gramas
        for conv in conversations:
            input_tokens = self.tokenizer.tokenize(conv["input"])
            response_tokens = self.tokenizer.tokenize(conv["response"])

            # Guardar pares input -> response
            input_key = " ".join(input_tokens[:5])
            self._responses[input_key].append(conv["response"])

            # Entrenar bigramas del response
            for i in range(len(response_tokens) - 1):
                ctx = response_tokens[i]
                next_word = response_tokens[i + 1]
                self._bigrams[ctx][next_word] += 1

            # Entrenar trigramas
            for i in range(len(response_tokens) - 2):
                ctx = f"{response_tokens[i]} {response_tokens[i+1]}"
                next_word = response_tokens[i + 2]
                self._trigrams[ctx][next_word] += 1

    def generate(
        self,
        prompt: str,
        max_length: int = 50,
        temperature: float = 0.7,
        context: list[str] | None = None,
    ) -> str:
        """Genera una respuesta basada en el prompt."""
        prompt_tokens = self.tokenizer.tokenize(prompt)

        # 1. Buscar respuesta directa (matching de patrones)
        direct_response = self._find_direct_response(prompt_tokens)
        if direct_response:
            return direct_response

        # 2. Buscar en contexto de conversación previa
        if context:
            context_response = self._use_context(prompt_tokens, context)
            if context_response:
                return context_response

        # 3. Generar con modelo de lenguaje
        return self._generate_with_model(prompt_tokens, max_length, temperature)

    def _find_direct_response(self, tokens: list[str]) -> str | None:
        """Busca una respuesta directa basada en patrones aprendidos."""
        # Buscar con diferentes longitudes de contexto
        for length in range(min(5, len(tokens)), 0, -1):
            key = " ".join(tokens[:length])
            if key in self._responses:
                return random.choice(self._responses[key])
        return None

    def _use_context(self, tokens: list[str], context: list[str]) -> str | None:
        """Usa el contexto de la conversación para generar respuesta."""
        # Buscar tokens del contexto en las respuestas conocidas
        context_tokens = []
        for ctx in context[-3:]:  # Últimos 3 mensajes
            context_tokens.extend(self.tokenizer.tokenize(ctx))

        # Buscar coincidencias
        for ctx_token in context_tokens[-5:]:
            if ctx_token in self._bigrams:
                # Generar continuación desde este token
                return self._continue_from(ctx_token, max_length=20)
        return None

    def _generate_with_model(
        self,
        seed_tokens: list[str],
        max_length: int,
        temperature: float,
    ) -> str:
        """Genera texto usando el modelo de lenguaje."""
        if not seed_tokens:
            # Empezar con un token aleatorio
            if self.tokenizer._token_freq:
                seed_tokens = [max(self.tokenizer._token_freq, key=self.tokenizer._token_freq.get)]
            else:
                return ""

        # Empezar desde el último token del prompt
        current = seed_tokens[-1]
        generated = list(seed_tokens)

        for _ in range(max_length):
            next_word = self._predict_next(current, temperature)
            if next_word is None or next_word == Tokenizer.EOS_TOKEN:
                break
            generated.append(next_word)
            current = next_word

        # Limpiar y devolver
        return self.tokenizer.decode(generated, skip_special=True)

    def _predict_next(self, current: str, temperature: float) -> str | None:
        """Predice la siguiente palabra usando bigramas y trigramas."""
        candidates: dict[str, float] = defaultdict(float)

        # Buscar en bigramas
        if current in self._bigrams:
            for word, count in self._bigrams[current].items():
                candidates[word] += count

        # Buscar en trigramas (con contexto anterior)
        candidates_list = list(candidates.items())
        if candidates_list:
            # Aplicar temperatura
            total = sum(c ** (1 / temperature) for _, c in candidates_list)
            if total > 0:
                r = random.random() * total
                cumulative = 0
                for word, count in candidates_list:
                    cumulative += count ** (1 / temperature)
                    if r <= cumulative:
                        return word

        return None

    def _continue_from(self, start_token: str, max_length: int = 20) -> str:
        """Genera texto continuando desde un token dado."""
        generated = [start_token]
        current = start_token

        for _ in range(max_length):
            next_word = self._predict_next(current, temperature=0.8)
            if next_word is None or next_word == Tokenizer.EOS_TOKEN:
                break
            generated.append(next_word)
            current = next_word

        return self.tokenizer.decode(generated, skip_special=True)

    def add_response_pattern(self, pattern: str, response: str) -> None:
        """Añade un patrón de respuesta manualmente."""
        tokens = self.tokenizer.tokenize(pattern)
        key = " ".join(tokens[:5])
        self._responses[key].append(response)

        # También entrenar n-gramas
        response_tokens = self.tokenizer.tokenize(response)
        for i in range(len(response_tokens) - 1):
            ctx = response_tokens[i]
            next_word = response_tokens[i + 1]
            self._bigrams[ctx][next_word] += 1

    def save(self, directory: Path | str) -> None:
        """Guarda el modelo de generación."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        self.tokenizer.save(directory / "tokenizer.json")

        data = {
            "bigrams": dict(self._bigrams),
            "trigrams": dict(self._trigrams),
            "responses": dict(self._responses),
        }
        (directory / "generator.json").write_text(
            json.dumps(data, ensure_ascii=False, default=dict)
        )

    def load(self, directory: Path | str) -> "TextGenerator":
        """Carga el modelo de generación."""
        directory = Path(directory)

        self.tokenizer.load(directory / "tokenizer.json")

        data = json.loads((directory / "generator.json").read_text())
        self._bigrams = defaultdict(lambda: defaultdict(int), data.get("bigrams", {}))
        self._trigrams = defaultdict(lambda: defaultdict(int), data.get("trigrams", {}))
        self._responses = defaultdict(list, data.get("responses", {}))

        return self
