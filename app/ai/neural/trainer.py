"""Entrenador de redes neuronales para A.R.I.A: backpropagation desde cero.

Implementa gradient descent, cross-entropy loss y training loop completo.
"""
import math
import random
from collections.abc import Callable
from typing import Any

from app.ai.neural.network import SequentialNetwork


def cross_entropy_loss(predicted: list[float], target: int) -> float:
    """Cross-entropy loss para clasificación."""
    # Evitar log(0)
    eps = 1e-15
    predicted_clipped = [max(eps, min(1 - eps, p)) for p in predicted]
    return -math.log(predicted_clipped[target])


def cross_entropy_gradient(predicted: list[float], target: int) -> list[float]:
    """Gradiente de cross-entropy + softmax."""
    grad = predicted.copy()
    grad[target] -= 1.0
    return grad


def mse_loss(predicted: list[float], target: list[float]) -> float:
    """Mean Squared Error para regresión."""
    total = 0.0
    for p, t in zip(predicted, target):
        total += (p - t) ** 2
    return total / len(predicted)


def mse_gradient(predicted: list[float], target: list[float]) -> list[float]:
    """Gradiente de MSE."""
    n = len(predicted)
    return [2 * (p - t) / n for p, t in zip(predicted, target)]


class Trainer:
    """Entrenador con mini-batch gradient descent."""

    def __init__(
        self,
        network: SequentialNetwork,
        loss_fn: str = "cross_entropy",
        learning_rate: float = 0.01,
    ) -> None:
        self.network = network
        self.learning_rate = learning_rate
        if loss_fn == "cross_entropy":
            self._loss_fn = cross_entropy_loss
            self._grad_fn = cross_entropy_gradient
        elif loss_fn == "mse":
            self._loss_fn = mse_loss
            self._grad_fn = mse_gradient
        else:
            raise ValueError(f"Loss desconocida: {loss_fn}")

    def train_step(
        self,
        x: list[float],
        y: int | list[float],
    ) -> float:
        """Un paso de entrenamiento: forward -> loss -> backward -> update.

        Para clasificación, y es un entero (índice de clase).
        Para regresión, y es una lista de floats.
        """
        # Forward
        predicted = self.network.forward(x)

        # Calcular loss y gradiente
        if isinstance(y, int):
            loss = self._loss_fn(predicted, y)
            grad = self._grad_fn(predicted, y)
        else:
            loss = self._loss_fn(predicted, y)
            grad = self._grad_fn(predicted, y)

        # Backward (actualiza pesos)
        self.network.backward(grad, self.learning_rate)

        return loss

    def train(
        self,
        X: list[list[float]],
        y: list[int] | list[list[float]],
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.1,
        callback: Callable[[int, float, float], Any] | None = None,
    ) -> list[dict[str, float]]:
        """Entrena la red con los datos proporcionados.

        Args:
            X: Entradas
            y: Salidas (enteros para clasificación, listas para regresión)
            epochs: Número de épocas
            batch_size: Tamaño del mini-batch
            validation_split: Fracción de datos para validación
            callback: Función llamada al final de cada época(epoch, train_loss, val_loss)

        Returns:
            Historial de entrenamiento
        """
        # Dividir en train/validation
        n = len(X)
        n_val = int(n * validation_split)
        indices = list(range(n))
        random.shuffle(indices)

        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        X_train = [X[i] for i in train_indices]
        y_train = [y[i] for i in train_indices]
        X_val = [X[i] for i in val_indices]
        y_val = [y[i] for i in val_indices]

        history = []

        for epoch in range(epochs):
            # Shuffle training data
            combined = list(zip(X_train, y_train))
            random.shuffle(combined)
            X_train_shuffled = [x for x, _ in combined]
            y_train_shuffled = [y_ for _, y_ in combined]

            # Mini-batch training
            total_loss = 0.0
            n_batches = 0
            for i in range(0, len(X_train_shuffled), batch_size):
                batch_X = X_train_shuffled[i : i + batch_size]
                batch_y = y_train_shuffled[i : i + batch_size]

                batch_loss = 0.0
                for x, y_ in zip(batch_X, batch_y):
                    batch_loss += self.train_step(x, y_)
                total_loss += batch_loss / len(batch_X)
                n_batches += 1

            train_loss = total_loss / max(n_batches, 1)

            # Validation loss
            val_loss = 0.0
            if X_val:
                for x, y_ in zip(X_val, y_val):
                    predicted = self.network.predict(x)
                    val_loss += self._loss_fn(predicted, y_)
                val_loss /= len(X_val)

            history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })

            if callback:
                callback(epoch + 1, train_loss, val_loss)

        return history

    def evaluate(
        self,
        X: list[list[float]],
        y: list[int],
    ) -> dict[str, float]:
        """Evalúa la precisión en un dataset."""
        correct = 0
        total = 0
        for x, y_ in zip(X, y):
            predicted = self.network.predict(x)
            predicted_class = predicted.index(max(predicted))
            if predicted_class == y_:
                correct += 1
            total += 1
        return {"accuracy": correct / max(total, 1), "correct": correct, "total": total}
