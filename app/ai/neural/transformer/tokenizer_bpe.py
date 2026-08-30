"""Byte-Pair Encoding tokenizer — tokenize texto como GPT.

Entrena pares de bytes frecuentes para crear un vocabulario efectivo.
"""
import json
from collections import Counter
from pathlib import Path
from typing import ClassVar


class BPETokenizer:
    """Tokenizador BPE (Byte Pair Encoding) para español."""

    SPECIAL_TOKENS: ClassVar[dict[str, int]] = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "<sep>": 4,
        "<user>": 5,
        "<assistant>": 6,
        "<think>": 7,
        "<action>": 8,
        "<result>": 9,
    }

    def __init__(self, vocab_size: int = 8000) -> None:
        self.vocab_size = vocab_size
        self.merges: list[tuple[str, str]] = []
        self.vocab: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self._is_trained = False

    def _get_stats(self, ids: list[int]) -> Counter:
        """Cuenta pares de tokens adyacentes (sin fusionar tokens especiales)."""
        counts: Counter = Counter()
        ns = len(self.SPECIAL_TOKENS)
        for i in range(len(ids) - 1):
            if ids[i] >= ns and ids[i + 1] >= ns:
                counts[(ids[i], ids[i + 1])] += 1
        return counts

    def _merge(self, ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Fusiona un par de tokens en uno nuevo."""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def _tokenize_ids(self, text: str) -> list[int]:
        """Convierte texto a IDs, tratando los tokens especiales como atómicos."""
        ids = []
        i = 0
        specials = sorted(self.SPECIAL_TOKENS, key=len, reverse=True)
        while i < len(text):
            matched = False
            for tok in specials:
                if text.startswith(tok, i):
                    ids.append(self.SPECIAL_TOKENS[tok])
                    i += len(tok)
                    matched = True
                    break
            if matched:
                continue
            ch = text[i]
            i += 1
            if ch in self.vocab:
                ids.append(self.vocab[ch])
            else:
                # Carácter UTF-8 compuesto — buscar sus bytes
                for b in ch.encode("utf-8"):
                    if chr(b) in self.vocab:
                        ids.append(self.vocab[chr(b)])
                    else:
                        ids.append(self.vocab.get("<unk>", 1))
        return ids

    def train(self, texts: list[str], verbose: bool = False) -> None:
        """Entrena el tokenizer en una lista de textos."""
        # Paso 1: tokenizar en bytes
        all_bytes = b""
        for text in texts:
            all_bytes += text.encode("utf-8")

        # Contar frecuencia de cada byte
        byte_freq = Counter(all_bytes)

        # Vocabulario base: cada byte individual
        self.vocab = {}
        for special, token_id in self.SPECIAL_TOKENS.items():
            self.vocab[special] = token_id

        # Índice inverso id -> token, mantenido en O(1) durante el entrenamiento
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        next_id = len(self.SPECIAL_TOKENS)

        # Agregar bytes más frecuentes primero
        for byte_val, _ in byte_freq.most_common():
            char = chr(byte_val) if byte_val < 128 else bytes([byte_val]).decode("utf-8", errors="replace")
            if char not in self.vocab:
                self.vocab[char] = next_id
                self.id_to_token[next_id] = char
                next_id += 1

        # Paso 2: BPE merges
        self.merges = []
        # Tokenizar todos los textos en IDs (especiales atómicos + bytes/merges)
        all_ids = [self._tokenize_ids(text) for text in texts]

        # Realizar merges hasta alcanzar vocab_size
        num_merges = self.vocab_size - len(self.vocab)
        for merge_idx in range(num_merges):
            stats: Counter = Counter()
            for ids in all_ids:
                stats.update(self._get_stats(ids))

            if not stats:
                break

            best_pair = stats.most_common(1)[0]
            if best_pair[1] < 2:
                break

            pair = best_pair[0]
            new_id = next_id
            next_id += 1

            pair_str = (
                self._id_to_token_str(pair[0]),
                self._id_to_token_str(pair[1]),
            )
            self.merges.append(pair_str)
            merged_token = pair_str[0] + pair_str[1]
            self.vocab[merged_token] = new_id
            self.id_to_token[new_id] = merged_token

            # Aplicar merge a todos los textos
            all_ids = [self._merge(ids, pair, new_id) for ids in all_ids]

            if verbose and (merge_idx + 1) % 100 == 0:
                print(f"  Merge {merge_idx + 1}/{num_merges}: "
                      f"'{pair_str[0]}' + '{pair_str[1]}'")

        self._is_trained = True

        if verbose:
            print(f"  Vocabulario: {len(self.vocab)} tokens")

    def _id_to_token_str(self, token_id: int) -> str:
        """Convierte un ID a string en O(1) usando el índice inverso."""
        return self.id_to_token.get(token_id, f"<{token_id}>")

    def encode(self, text: str) -> list[int]:
        """Codifica texto a secuencia de IDs."""
        if not self._is_trained:
            raise RuntimeError("Tokenizer no entrenado")

        ids = self._tokenize_ids(text)

        # Aplicar merges en orden sobre tokens no especiales
        for pair_str in self.merges:
            pair = (
                self.vocab.get(pair_str[0], -1),
                self.vocab.get(pair_str[1], -1),
            )
            if pair[0] == -1 or pair[1] == -1:
                continue
            new_id = self.vocab.get(pair_str[0] + pair_str[1], -1)
            if new_id == -1:
                continue
            ids = self._merge(ids, pair, new_id)

        return ids

    def decode(self, ids: list[int]) -> str:
        """Decodifica IDs a texto."""
        tokens = []
        for token_id in ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if token in self.SPECIAL_TOKENS:
                    continue
                tokens.append(token)
            else:
                tokens.append("<unk>")
        return "".join(tokens)

    def save(self, path: Path | str) -> None:
        """Guarda el tokenizer."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        data = {
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "merges": self.merges,
        }
        (path / "bpe.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def load(self, path: Path | str) -> None:
        """Carga el tokenizer."""
        path = Path(path)
        data = json.loads((path / "bpe.json").read_text())
        self.vocab_size = data["vocab_size"]
        self.vocab = data["vocab"]
        self.merges = [tuple(m) for m in data["merges"]]
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self._is_trained = True

    @property
    def vocab_len(self) -> int:
        return len(self.vocab)
