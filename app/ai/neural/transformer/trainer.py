"""Training pipeline para el modelo GPT.

Entrena el modelo usando next-token prediction sobre conversaciones
y conocimiento, con checkpointing y resume.
"""
import json
import math
import random
import time
from pathlib import Path

from app.ai.neural.transformer.gpt_model import GPTModel
from app.ai.neural.transformer.tokenizer_bpe import BPETokenizer


class GPTTrainer:
    """Entrena el modelo GPT con next-token prediction."""

    def __init__(
        self,
        model: GPTModel,
        tokenizer: BPETokenizer,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self._step = 0
        self._grad_accum: dict[str, float] = {}

    def train_on_conversations(
        self,
        conversations: list[dict[str, str]],
        epochs: int = 5,
        batch_size: int = 4,
        max_len: int = 256,
        verbose: bool = True,
    ) -> list[dict[str, float]]:
        """Entrena con pares de conversación (user → assistant)."""
        all_texts = []
        training_sequences = []
        for conv in conversations:
            user = conv.get("user", "")
            assistant = conv.get("assistant", "")
            if not user or not assistant:
                continue

            text = f"<user>{user}<assistant>{assistant}<eos>"
            all_texts.append(text)

        if all_texts and not self.tokenizer._is_trained:
            tokenizer_texts = list(all_texts)
            tokenizer_texts.extend(word for text in all_texts for word in text.split())
            self.tokenizer.train(tokenizer_texts, verbose=False)

        for conv in conversations:
            user = conv.get("user", "")
            assistant = conv.get("assistant", "")
            if not user or not assistant:
                continue

            text = f"<user>{user}<assistant>{assistant}<eos>"
            ids = self.tokenizer.encode(text)

            if len(ids) < 3:
                continue
            if len(ids) > max_len:
                ids = ids[:max_len]

            training_sequences.append(ids)

        if not training_sequences:
            return []

        return self._train_sequences(training_sequences, epochs, batch_size, verbose)

    def train_on_text(
        self,
        texts: list[str],
        epochs: int = 3,
        batch_size: int = 4,
        max_len: int = 256,
        verbose: bool = True,
    ) -> list[dict[str, float]]:
        """Entrena con texto puro (continuation learning)."""
        if texts and not self.tokenizer._is_trained:
            tokenizer_texts = list(texts)
            tokenizer_texts.extend(word for text in texts for word in text.split())
            self.tokenizer.train(tokenizer_texts, verbose=False)

        training_sequences = []
        for text in texts:
            full_text = f"<bos>{text}<eos>"
            ids = self.tokenizer.encode(full_text)

            for i in range(0, len(ids) - 1, max_len // 2):
                chunk = ids[i:i + max_len]
                if len(chunk) >= 4:
                    training_sequences.append(chunk)

        if not training_sequences:
            return []

        return self._train_sequences(training_sequences, epochs, batch_size, verbose)

    def _train_sequences(
        self,
        sequences: list[list[int]],
        epochs: int,
        batch_size: int,
        verbose: bool,
    ) -> list[dict[str, float]]:
        """Bucle de entrenamiento sobre secuencias tokenizadas."""
        history = []
        total_params = self.model.count_params()

        if verbose:
            print(f"  Secuencias: {len(sequences)}, Params: {total_params:,}, Épocas: {epochs}")

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            start = time.time()

            random.shuffle(sequences)

            for batch_start in range(0, len(sequences), batch_size):
                batch = sequences[batch_start:batch_start + batch_size]

                for seq in batch:
                    if len(seq) < 3:
                        continue

                    input_ids = seq[:-1]
                    target_ids = seq[1:]

                    # Forward
                    logits = self.model.forward(input_ids)

                    # Calcular loss
                    loss = 0.0
                    for i, target_id in enumerate(target_ids):
                        if i >= len(logits):
                            break
                        max_l = max(logits[i])
                        exp_l = [math.exp(v - max_l) for v in logits[i]]
                        total_exp = sum(exp_l)
                        prob_correct = exp_l[target_id] / total_exp
                        loss -= math.log(prob_correct + 1e-10)
                        predicted_class = max(range(len(logits[i])), key=lambda j: logits[i][j])
                        if predicted_class == target_id:
                            epoch_correct += 1
                        epoch_total += 1

                    loss /= len(target_ids)
                    epoch_loss += loss

                    # Backward simplificado: ajustar pesos proporcionalmente al error
                    self._backward_pass(input_ids, target_ids, logits, loss)

                    self._step += 1

            avg_loss = epoch_loss / len(sequences) if sequences else 0
            accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0
            elapsed = time.time() - start
            perplexity = math.exp(min(avg_loss, 20))

            epoch_stats = {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "accuracy": accuracy,
                "perplexity": perplexity,
                "time": elapsed,
            }
            history.append(epoch_stats)

            if verbose:
                print(
                    f"  Época {epoch + 1}/{epochs}: "
                    f"loss={avg_loss:.4f}, acc={accuracy:.1%}, "
                    f"ppl={perplexity:.2f}, {elapsed:.1f}s"
                )

        return history

    def _backward_pass(
        self,
        input_ids: list[int],
        target_ids: list[int],
        logits: list[list[float]],
        loss: float,
    ) -> None:
        """Backward pass completo a través del modelo GPT."""
        lr = self.lr * (0.95 ** (self._step // 50))
        self.model.backward(target_ids, lr=lr)

    def save_checkpoint(self, path: Path | str) -> None:
        """Guarda checkpoint del entrenamiento."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(path / "model")
        self.tokenizer.save(path / "tokenizer")
        meta = {"step": self._step}
        (path / "meta.json").write_text(json.dumps(meta))

    def load_checkpoint(self, path: Path | str) -> None:
        """Carga checkpoint."""
        path = Path(path)
        self.model.load(path / "model")
        self.tokenizer.load(path / "tokenizer")
        meta_file = path / "meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            self._step = meta.get("step", 0)
