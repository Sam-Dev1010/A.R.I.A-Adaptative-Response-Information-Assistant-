"""Training pipeline para el modelo GPT.

Entrena el modelo usando next-token prediction sobre conversaciones
y conocimiento, con checkpointing y resume.
"""
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from app.ai.neural.transformer.gpt_model import GPTModel
from app.ai.neural.transformer.tokenizer_bpe import BPETokenizer
from app.ai.neural.transformer.trainer_batch import (
    backward_batched,
    forward_batched,
)


class GPTTrainer:
    """Entrena el modelo GPT con next-token prediction."""

    def __init__(
        self,
        model: GPTModel,
        tokenizer: BPETokenizer,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        lr_decay: float = 0.95,
        lr_decay_every: int = 50,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.lr_decay = lr_decay
        self.lr_decay_every = lr_decay_every
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
        stride: int = 0,
        verbose: bool = True,
    ) -> list[dict[str, float]]:
        """Entrena con texto puro (continuation learning).

        `stride` controla el solapamiento de las ventanas: por defecto es
        `max_len // 2` (conserva continuidad); para corpus grandes conviene
        `stride == max_len` (sin solapamiento, la mitad de secuencias).
        """
        if texts and not self.tokenizer._is_trained:
            tokenizer_texts = list(texts)
            tokenizer_texts.extend(word for text in texts for word in text.split())
            self.tokenizer.train(tokenizer_texts, verbose=False)

        stride = stride or (max_len // 2)
        training_sequences = []
        for text in texts:
            full_text = f"<bos>{text}<eos>"
            ids = self.tokenizer.encode(full_text)

            for i in range(0, len(ids) - 1, stride):
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
        use_batched: bool = True,
    ) -> list[dict[str, float]]:
        """Bucle de entrenamiento sobre secuencias tokenizadas.

        Con `use_batched` y `batch_size > 1` procesa lotes enteros a la vez
        (mini-batch vectorizado con NumPy), mucho más rápido que secuencia a
        secuencia. Las secuencias se agrupan por longitud para mín. padding.
        """
        history = []
        total_params = self.model.count_params()

        if verbose:
            print(f"  Secuencias: {len(sequences)}, Params: {total_params:,}, Épocas: {epochs}")

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            start = time.time()

            seqs = sorted(sequences, key=len)  # agrupar longitudes parecidas
            batches: list[list[list[int]]] = [
                seqs[i:i + max(batch_size, 1)] for i in range(0, len(seqs), max(batch_size, 1))
            ]
            random.shuffle(batches)

            for batch in batches:
                batch = [s for s in batch if len(s) >= 3]
                if not batch:
                    continue
                # Ordenar dentro del lote para reducir el padding
                batch.sort(key=len, reverse=True)

                if use_batched and len(batch) > 1:
                    losses, correct, total = self._train_batch(batch)
                    epoch_loss += float(losses) * len(batch)
                    epoch_correct += correct
                    epoch_total += total
                else:
                    for seq in batch:
                        input_ids = seq[:-1]
                        target_ids = seq[1:]
                        logits = self.model.forward(input_ids)
                        loss, correct, total = self._loss_stats(logits, target_ids)
                        epoch_loss += loss
                        epoch_correct += correct
                        epoch_total += total
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

    def _loss_stats(
        self,
        logits: list[list[float]],
        target_ids: list[int],
    ) -> tuple[float, int, int]:
        """Loss media + aciertos sobre una secuencia (camino por secuencia)."""
        loss = 0.0
        correct = 0
        total = 0
        n = min(len(logits), len(target_ids))
        for i in range(n):
            target_id = target_ids[i]
            max_l = max(logits[i])
            exp_l = [math.exp(v - max_l) for v in logits[i]]
            total_exp = sum(exp_l)
            loss -= math.log(exp_l[target_id] / total_exp + 1e-10)
            if max(range(len(logits[i])), key=lambda j: logits[i][j]) == target_id:
                correct += 1
            total += 1
        loss /= n
        return loss, correct, total

    def _get_lr(self) -> float:
        """lr actual según schedule (ambas ramas usan la misma fórmula)."""
        if self.lr_decay_every <= 0 or self.lr_decay >= 1.0:
            return self.lr
        return self.lr * (self.lr_decay ** (self._step // self.lr_decay_every))

    def _train_batch(self, batch: list[list[int]]) -> tuple[float, int, int]:
        """Entrena un lote completo (pad + forward + backward vectorizados).

        Devuelve (loss_media_del_lote, aciertos, total_posiciones_válidas).
        """
        seq_len = max(len(s) for s in batch)
        M = min(seq_len - 1, self.model.max_seq_len)
        tokens = np.zeros((len(batch), M), dtype=np.int64)
        targets = np.zeros((len(batch), M), dtype=np.int64)
        valid = np.zeros((len(batch), M), dtype=bool)
        for i, seq in enumerate(batch):
            n = len(seq)
            if n - 1 > M:
                # igual que el camino por secuencia: forward usa los últimos
                # max_seq_len inputs y empareja con los PRIMEROS max_seq_len targets
                row_in = seq[:-1][-M:]
                row_tg = seq[1:][:M]
                tokens[i, :] = row_in
                targets[i, :] = row_tg
                valid[i, :] = True
            else:
                tokens[i, :n - 1] = seq[:-1]
                targets[i, :n - 1] = seq[1:]
                valid[i, :n - 1] = True

        logits, cache = forward_batched(self.model, tokens)
        lr = self._get_lr()
        loss = backward_batched(self.model, targets, valid, cache, lr)
        if self.weight_decay > 0:
            self.model.apply_weight_decay(lr, self.weight_decay)
        self._step += len(batch)

        # Aciertos sobre posiciones válidas
        pred = np.argmax(logits, axis=-1)
        correct = int(np.sum((pred == targets) & valid))
        total = int(valid.sum())
        return loss, correct, total

    def _backward_pass(
        self,
        input_ids: list[int],
        target_ids: list[int],
        logits: list[list[float]],
        loss: float,
    ) -> None:
        """Backward pass completo a través del modelo GPT."""
        lr = self._get_lr()
        self.model.backward(target_ids, lr=lr)
        self.model.apply_weight_decay(lr, self.weight_decay)

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
