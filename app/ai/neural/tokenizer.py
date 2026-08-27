"""Tokenizador propio para A.R.I.A: convierte texto a IDs numéricos.

No depende de librerías externas. Aprende vocabulario nuevo durante el
entrenamiento y maneja tokens desconocidos.
"""
import json
import re
from pathlib import Path


class Tokenizer:
    """Tokenizador basado en vocabulario con soporte para subpalabras simples."""

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self, vocab_size: int = 10000) -> None:
        self.vocab_size = vocab_size
        self.token_to_id: dict[str, int] = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.SOS_TOKEN: 2,
            self.EOS_TOKEN: 3,
        }
        self.id_to_token: dict[int, str] = {v: k for k, v in self.token_to_id.items()}
        self._token_freq: dict[str, int] = {}
        self._next_id = len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    @property
    def sos_id(self) -> int:
        return 2

    @property
    def eos_id(self) -> int:
        return 3

    def tokenize(self, text: str) -> list[str]:
        """Tokeniza texto en una lista de tokens.

        Reglas:
        - Minúsculas
        - Separa puntuación
        - Mantén números como tokens únicos
        - Separa por espacios
        """
        text = text.lower().strip()
        # Separar puntuación
        text = re.sub(r'([.,!?;:¿¡()"\'])', r' \1 ', text)
        # Separar números de letras
        text = re.sub(r'(\d+)', r' \1 ', text)
        # Múltiples espacios a uno
        text = re.sub(r'\s+', ' ', text)
        return text.split()

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        """Convierte texto a secuencia de IDs."""
        tokens = self.tokenize(text)
        ids = []
        if add_special:
            ids.append(self.sos_id)
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                ids.append(self.unk_id)
        if add_special:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Convierte secuencia de IDs a texto."""
        tokens = []
        for id_ in ids:
            token = self.id_to_token.get(id_, self.UNK_TOKEN)
            if skip_special and token in (
                self.PAD_TOKEN,
                self.UNK_TOKEN,
                self.SOS_TOKEN,
                self.EOS_TOKEN,
            ):
                continue
            tokens.append(token)
        return " ".join(tokens)

    def train(self, texts: list[str]) -> None:
        """Aprende vocabulario de una lista de textos.

        Los tokens más frecuentes se añaden primero hasta cubrir vocab_size.
        """
        # Contar frecuencias
        for text in texts:
            for token in self.tokenize(text):
                self._token_freq[token] = self._token_freq.get(token, 0) + 1

        # Ordenar por frecuencia (mayor primero)
        sorted_tokens = sorted(self._token_freq.items(), key=lambda x: -x[1])

        # Añadir hasta vocab_size
        for token, _ in sorted_tokens:
            if self._next_id >= self.vocab_size:
                break
            if token not in self.token_to_id:
                self.token_to_id[token] = self._next_id
                self.id_to_token[self._next_id] = token
                self._next_id += 1

    def save(self, path: Path | str) -> None:
        """Guarda el vocabulario a un archivo JSON."""
        data = {
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def load(self, path: Path | str) -> "Tokenizer":
        """Carga el vocabulario desde un archivo JSON."""
        data = json.loads(Path(path).read_text())
        self.vocab_size = data["vocab_size"]
        self.token_to_id = data["token_to_id"]
        self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}
        self._next_id = max(self.id_to_token.keys()) + 1
        return self

    def __len__(self) -> int:
        return self._next_id

    def __contains__(self, token: str) -> bool:
        return token in self.token_to_id
