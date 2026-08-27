"""Red neuronal secuencial para A.R.I.A: conecta capas en cadena.

Soporta forward pass (inferencia) y backward pass (entrenamiento).
"""
import json
from pathlib import Path

from app.ai.neural.layers import Dense, Layer


class SequentialNetwork:
    """Red neuronal secuencial: capa1 -> capa2 -> ... -> capaN."""

    def __init__(self, layers: list[Layer] | None = None) -> None:
        self.layers: list[Layer] = layers or []

    def add(self, layer: Layer) -> "SequentialNetwork":
        """Añade una capa al final. Devuelve self para encadenar."""
        self.layers.append(layer)
        return self

    def forward(self, x: list[float]) -> list[float]:
        """Propagación hacia adelante a través de todas las capas."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: list[float], lr: float) -> None:
        """Propagación hacia atrás: actualiza pesos de todas las capas."""
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)

    def predict(self, x: list[float]) -> list[float]:
        """Inferencia (sin gradientes)."""
        return self.forward(x)

    def summary(self) -> str:
        """Resumen de la arquitectura."""
        lines = ["Red Neuronal Sequential:", "=" * 40]
        total_params = 0
        for i, layer in enumerate(self.layers):
            name = type(layer).__name__
            if isinstance(layer, Dense):
                out_size = layer.output_size()
                params = layer.input_size * out_size + out_size
                total_params += params
                lines.append(
                    f"  [{i}] {name}: {layer.input_size} -> {out_size} "
                    f"({params:,} parámetros)"
                )
            else:
                lines.append(f"  [{i}] {name}")
        lines.append("=" * 40)
        lines.append(f"Total parámetros: {total_params:,}")
        return "\n".join(lines)

    def save(self, path: Path | str) -> None:
        """Guarda los pesos de la red a un archivo JSON."""
        data = {"layers": []}
        for layer in self.layers:
            if isinstance(layer, Dense):
                data["layers"].append({
                    "type": "Dense",
                    "input_size": layer.input_size,
                    "output_size": layer.output_size_,
                    "weights": layer.weights,
                    "bias": layer.bias,
                })
            else:
                data["layers"].append({"type": type(layer).__name__})
        Path(path).write_text(json.dumps(data, ensure_ascii=False))

    def load(self, path: Path | str) -> "SequentialNetwork":
        """Carga los pesos desde un archivo JSON."""
        data = json.loads(Path(path).read_text())
        for i, layer_data in enumerate(data["layers"]):
            if i >= len(self.layers):
                break
            if layer_data["type"] == "Dense" and isinstance(self.layers[i], Dense):
                self.layers[i].weights = layer_data["weights"]
                self.layers[i].bias = layer_data["bias"]
        return self

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, idx: int) -> Layer:
        return self.layers[idx]
