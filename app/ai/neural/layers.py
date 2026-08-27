"""Capas neuronales base para A.R.I.A: implementación desde cero.

Incluye Dense (totalmente conectada), activaciones y utilidades matemáticas
sin depender de NumPy o TensorFlow.
"""
import math
import random
from abc import ABC, abstractmethod


def _xavier_init(fan_in: int, fan_out: int) -> list[list[float]]:
    """Inicialización Xavier/Glorot para pesos."""
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return [
        [random.uniform(-limit, limit) for _ in range(fan_out)]
        for _ in range(fan_in)
    ]


def _zeros(size: int) -> list[float]:
    return [0.0] * size


class Layer(ABC):
    """Capa base abstracta."""

    @abstractmethod
    def forward(self, x: list[float]) -> list[float]:
        """Propagación hacia adelante."""

    @abstractmethod
    def backward(self, grad: list[float], lr: float) -> list[float]:
        """Propagación hacia atrás con actualización de pesos."""

    @abstractmethod
    def output_size(self) -> int:
        """Tamaño de la salida de la capa."""


class Dense(Layer):
    """Capa totalmente conectada: output = input @ weights + bias."""

    def __init__(self, input_size: int, output_size: int) -> None:
        self.input_size = input_size
        self.output_size_ = output_size
        self.weights = _xavier_init(input_size, output_size)
        self.bias = _zeros(output_size)
        # Cache para backprop
        self._last_input: list[float] = []

    def output_size(self) -> int:
        return self.output_size_

    def forward(self, x: list[float]) -> list[float]:
        self._last_input = x
        output = []
        for j in range(self.output_size_):
            total = self.bias[j]
            for i in range(self.input_size):
                total += x[i] * self.weights[i][j]
            output.append(total)
        return output

    def backward(self, grad: list[float], lr: float) -> list[float]:
        """Backprop: calcula gradiente de entrada y actualiza pesos."""
        grad_input = _zeros(self.input_size)
        for i in range(self.input_size):
            for j in range(self.output_size_):
                grad_input[i] += grad[j] * self.weights[i][j]
                # Actualizar peso
                self.weights[i][j] -= lr * grad[j] * self._last_input[i]
            # No actualizar bias aquí (se hace por j)
        for j in range(self.output_size_):
            self.bias[j] -= lr * grad[j]
        return grad_input


class Activation(Layer):
    """Capa de activación (sin parámetros entrenables)."""

    def __init__(self, func: str = "relu") -> None:
        self.func = func
        self._last_output: list[float] = []

    def output_size(self) -> int:
        return 0  # Se determina dinámicamente

    def forward(self, x: list[float]) -> list[float]:
        if self.func == "relu":
            self._last_output = [max(0.0, v) for v in x]
        elif self.func == "sigmoid":
            self._last_output = [1.0 / (1.0 + math.exp(-max(-500, min(500, v)))) for v in x]
        elif self.func == "tanh":
            self._last_output = [math.tanh(v) for v in x]
        elif self.func == "softmax":
            max_val = max(x)
            exp_vals = [math.exp(v - max_val) for v in x]
            total = sum(exp_vals)
            self._last_output = [v / total for v in exp_vals]
        elif self.func == "leaky_relu":
            self._last_output = [v if v > 0 else 0.01 * v for v in x]
        else:
            raise ValueError(f"Activación desconocida: {self.func}")
        return self._last_output

    def backward(self, grad: list[float], lr: float) -> list[float]:
        if self.func == "relu":
            return [g if v > 0 else 0.0 for g, v in zip(grad, self._last_output)]
        elif self.func == "sigmoid":
            return [g * v * (1 - v) for g, v in zip(grad, self._last_output)]
        elif self.func == "tanh":
            return [g * (1 - v * v) for g, v in zip(grad, self._last_output)]
        elif self.func == "softmax":
            # Softmax + cross-entropy ya está en el loss
            return grad
        elif self.func == "leaky_relu":
            return [g if v > 0 else 0.01 * g for g, v in zip(grad, self._last_output)]
        return grad


class Embedding(Layer):
    """Capa de embedding: convierte IDs de tokens a vectores densos."""

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # Inicialización aleatoria pequeña
        self.embeddings = [
            [random.gauss(0, 0.1) for _ in range(embedding_dim)]
            for _ in range(vocab_size)
        ]
        self._last_id: int = 0

    def output_size(self) -> int:
        return self.embedding_dim

    def forward(self, x: list[float]) -> list[float]:
        """x contiene un solo ID (float)."""
        self._last_id = int(x[0])
        if 0 <= self._last_id < self.vocab_size:
            return self.embeddings[self._last_id]
        return _zeros(self.embedding_dim)

    def backward(self, grad: list[float], lr: float) -> list[float]:
        """Actualiza solo el embedding usado."""
        if 0 <= self._last_id < self.vocab_size:
            for d in range(self.embedding_dim):
                self.embeddings[self._last_id][d] -= lr * grad[d]
        return [0.0]  # No hay gradiente para el ID


class Flatten(Layer):
    """Aplana una lista de listas a una lista simple."""

    def __init__(self) -> None:
        self._input_shape: tuple[int, ...] = ()

    def output_size(self) -> int:
        return 0  # Dinámico

    def forward(self, x: list[float]) -> list[float]:
        self._input_shape = (len(x),)
        return x

    def backward(self, grad: list[float], lr: float) -> list[float]:
        return grad
