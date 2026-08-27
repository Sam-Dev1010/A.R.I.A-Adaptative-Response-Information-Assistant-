"""Inference Engine: genera respuestas reales usando el modelo GPT.

No usa respuestas predefinidas — genera texto token por token.
"""
import time
from pathlib import Path

from app.ai.neural.transformer.gpt_model import GPTModel
from app.ai.neural.transformer.tokenizer_bpe import BPETokenizer
from app.core.logging import get_logger

logger = get_logger("sia.transformer")


class GPTInference:
    """Motor de inferencia: genera respuestas reales."""

    def __init__(
        self,
        model: GPTModel,
        tokenizer: BPETokenizer,
        max_new_tokens: int = 150,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        # Estado de conversación
        self._context: list[int] = []
        self._max_context = 384  # Dejar espacio para generar

    def respond(self, user_message: str) -> str:
        """Genera una respuesta al mensaje del usuario."""
        start = time.time()

        # Tokenizar input con formato especial
        prompt = f"<user>{user_message}<assistant>"
        prompt_ids = self.tokenizer.encode(prompt)

        # Agregar al contexto
        self._context.extend(prompt_ids)

        # Truncar contexto si es muy largo
        if len(self._context) > self._max_context:
            self._context = self._context[-self._max_context:]

        # Generar
        generated_ids = self.model.generate(
            self._context.copy(),
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )

        # Extraer solo los tokens nuevos
        new_tokens = generated_ids[len(self._context):]
        self._context.extend(new_tokens)

        # Decodificar
        response = self.tokenizer.decode(new_tokens)

        # Limpiar
        response = response.strip()
        # Cortar en <eos> si aparece
        for stop in ["<eos>", "<user>", "<pad>"]:
            if stop in response:
                response = response[:response.index(stop)].strip()

        elapsed = time.time() - start
        logger.debug(
            "Respuesta generada: %d tokens en %.2fs",
            len(new_tokens),
            elapsed,
        )

        return response if response else "No pude generar una respuesta."

    def reset_context(self) -> None:
        """Reinicia el contexto de conversación."""
        self._context.clear()

    def get_context_tokens(self) -> int:
        return len(self._context)
