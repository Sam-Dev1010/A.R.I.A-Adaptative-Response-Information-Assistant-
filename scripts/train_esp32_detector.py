#!/usr/bin/env python3
"""Entrena el detector de frases del nodo de IA local del ESP32.

Construye un clasificador binario de frases (texto) pensado para correr en la
placa con TensorFlow Lite Micro. El nodo recibe una frase por Serial, la
convierte en 64 características de n-gramas sobre los bytes UTF-8, la pasa por
una red densa 64 -> 32 -> 1 y, si el resultado supera el umbral, avisa a
A.R.I.A por HTTP ("evento detectado").

Fuente de datos: el dataset de frases en español ya existente en el repo
(``data/training_data.json`` cuyos ``texts`` + ``intents`` usa el clasificador
de intenciones). Este script deriva una etiqueta binaria: ``1`` si la frase
requiere una acción del asistente (intención ``COMANDO``), ``0`` en caso
contrario. De este modo el detector en la placa discrimina "frase que dispara
una alerta" frente a texto de conversación ordinario.

Características (64 dims, todas en [0,1] para facilitar la cuantización INT8):
  - f[0] : longitud de la frase en bytes, normalizada (límite 200).
  - f[1..63] : cuenta normalizada de los 63 bigramas de bytes más frecuentes
    del corpus. El vocabulario (los propios bigramas + su máximo normalizador)
    se guarda en ``features_data.h`` para que el firmware C los reproduzca
    byte a byte sin depender de Python.

Red: Dense(64) -> ReLU -> Dense(32) -> ReLU -> Dense(1) -> Sigmoid.

Salidas (en ``data/detector_esp32/``):
  - detector.keras            modelo Keras completo
  - calibracion.npy           dataset representativo para la calibración INT8
  - features.json             vocabulario + parámetros de extracción
  - detector.tflite           modelo cuantizado INT8 (si --export)
  - model_data.h              header C++ del modelo (en firmware/lacal-IA-node)
  - features_data.h           header C++ del vocabulario (en el firmware)

Validación numérica de la cuantización: compara la predicción del modelo float
(.keras) contra la del modelo TFLite INT8 sobre las mismas muestras y reporta
el error medio absoluto. Debe ser pequeño (~<0.02) para considerarse válido.

Ejemplos:
  python scripts/train_esp32_detector.py
  python scripts/train_esp32_detector.py --export
  python scripts/train_esp32_detector.py --dataset data/neural_demo/training_data.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Importamos el pipeline de conversión refactorizado (evita duplicar lógica).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_tflite import (
    _convertir_a_tflite,
    _generar_header,
    _resumen,
)

RUTA_POR_DEFECTO_DS = Path(__file__).resolve().parent.parent / "data" / "neural" / "training_data.json"
RUTA_SALIDA = Path(__file__).resolve().parent.parent / "data" / "detector_esp32"
RUTA_FIRMWARE = Path(__file__).resolve().parent.parent / "firmware" / "lacal-IA-node" / "src"

N_FEATURES = 64      # 1 (longitud) + 63 (bigramas)
LIMITE_LONG = 200    # longitud máxima de frase (bytes) para normalizar
ETIQUETA_POSITIVA = "COMANDO"  # intención que dispara la alerta

_AYUDA_DEPS = (
    "Faltan dependencias de Python. Instala TensorFlow y NumPy:\n"
    "  pip install tensorflow numpy\n"
    "y vuelve a ejecutar el script (o usa el contenedor de TF 2.19)."
)


def _abrir(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Entrena el detector de frases binario 64->32->1 para el nodo de IA "
            "local del ESP32 (características de n-gramas sobre bytes UTF-8)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default=str(RUTA_POR_DEFECTO_DS),
        help="Path al dataset JSON {texts, intents} (por defecto el de neural).",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default=str(RUTA_SALIDA),
        help="Directorio de salida para los artefactos de entrenamiento.",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=200,
        help="Épocas de entrenamiento (por defecto 200).",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.2,
        help="Fracción de datos de validación (por defecto 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria (por defecto 42).",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Además de entrenar, convierte a TFLite INT8 y regenera los "
        "headers C++ del firmware (model_data.h + features_data.h).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Entrena y valida, pero no escribe ningún archivo.",
    )
    return parser.parse_args(argv)


# ------------------------- Extracción de features --------------------------

def _minusc_byte(b: int) -> int:
    """Minúscula ASCII (un solo byte). Los bytes ≥128 se dejan igual."""
    if 0x41 <= b <= 0x5A:
        return b + 0x20
    return b


def _bigramas(byte_seq: bytes) -> list[tuple[int, int]]:
    """Bigramas de bytes (minúscula-insensible) de la secuencia."""
    seq = bytes(_minusc_byte(b) for b in byte_seq)
    return [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]


def _derivar_vocabulario(textos: list[str], n_slots: int = N_FEATURES - 1):
    """Construye el vocabulario determinista: los ``n_slots`` bigramas de bytes
    más frecuentes del corpus y, por cada bigrama, su máximo normalizador
    (= la mayor cuenta de ese bigrama dentro de una sola frase del corpus).

    Devuelve SIEMPRE exactamente ``n_slots`` bigramas: si el corpus tiene menos,
    rellena con bigramas nulos (0x00,0x00) que nunca coinciden (feature 0).
    Así los headers C++ del firmware conservan el tamaño fijo ``kNumFeatures``.
    """
    from collections import Counter

    contadores_por_frase = [Counter(_bigramas(t.encode("utf-8"))) for t in textos]
    frecuencias_globales: Counter = Counter()
    for c in contadores_por_frase:
        frecuencias_globales.update(c)

    elegidos = frecuencias_globales.most_common(n_slots)
    # Desempate determinista por clave (byte_a, byte_b) para reproducibilidad.
    elegidos.sort(key=lambda kv: (kv[0], kv[1]))

    vocabulario = [bigrama for bigrama, _ in elegidos]
    maximos = []
    for bigrama, _ in elegidos:
        max_c = max((c[bigrama] for c in contadores_por_frase), default=0)
        maximos.append(max_c if max_c > 0 else 1)

    # Relleno hasta n_slots para tamaño fijo.
    while len(vocabulario) < n_slots:
        vocabulario.append((0x00, 0x00))
        maximos.append(1)

    return vocabulario, maximos


def extraer_features(texto: str, vocabulario: list, maximos: list) -> list[float]:
    """Convierte una frase en el vector de 64 features (todos en [0,1])."""
    from collections import Counter

    byte_seq = texto.encode("utf-8")
    feats = [0.0] * N_FEATURES
    feats[0] = min(len(byte_seq), LIMITE_LONG) / LIMITE_LONG
    contador = Counter(_bigramas(byte_seq))
    for i, bg in enumerate(vocabulario):
        feats[i + 1] = min(contador[bg] / maximos[i], 1.0)
    return feats


# ------------------------- Carga y etiquetado -------------------------------

def _cargar_datos(dataset: Path, etiqueta_positiva: str = ETIQUETA_POSITIVA):
    """Carga {texts, intents} y deriva la etiqueta binaria."""
    import numpy as np

    with open(dataset, encoding="utf-8") as fh:
        datos = json.load(fh)
    textos = datos["texts"]
    intenciones = datos["intents"]
    if not isinstance(textos, list) or not isinstance(intenciones, list):
        raise SystemExit("El dataset debe tener listas 'texts' e 'intents'.")
    if len(textos) != len(intenciones):
        raise SystemExit(f"Desajuste: {len(textos)} textos vs {len(intenciones)} intents.")
    labels = np.asarray([1.0 if i == etiqueta_positiva else 0.0 for i in intenciones])
    return textos, labels


def _construir_matriz(textos: list[str], vocabulario: list, maximos: list):
    import numpy as np

    return np.asarray(
        [extraer_features(t, vocabulario, maximos) for t in textos], dtype=np.float32
    )


def _resumen_dataset(textos, labels) -> None:
    import numpy as np

    positivas = int(np.sum(labels))
    print(
        f"[datos] {len(textos)} frases | positivas (COMANDO)={positivas} "
        f"({positivas / len(textos):.1%}) | negativas={len(textos) - positivas}"
    )


# ------------------------- Modelo y entrenamiento ---------------------------

def _construir_modelo():
    import tensorflow as tf

    modelo = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(N_FEATURES,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


def _entrenar(X, y, epochs: int, split: float, seed: int):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    modelo = _construir_modelo()
    historia = modelo.fit(
        X,
        y,
        epochs=epochs,
        validation_split=split,
        batch_size=16,
        verbose=0,
    )
    return modelo, historia


# ------------------------- Validación de cuantización -----------------------

def _validar_cuantizacion(keras_model, X, blob: bytes) -> float:
    """Compara float vs INT8 sobre las primeras muestras; devuelve MAE."""
    import numpy as np
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_content=blob)
    interp.allocate_tensors()
    entrada = interp.get_input_details()[0]
    salida = interp.get_output_details()[0]
    entrada_int = entrada["dtype"] == np.int8

    errores = []
    for muestra in X[:50]:
        pred_float = float(keras_model.predict(muestra[None, :], verbose=0)[0, 0])
        if entrada_int:
            zp, s = entrada["quantization"][1], entrada["quantization"][2]
            m = np.round(muestra / s + zp).astype(np.int8)
            interp.set_tensor(entrada["index"], m[None, :])
        else:
            interp.set_tensor(entrada["index"], muestra[None, :].astype(np.float32))
        interp.invoke()
        out = interp.get_tensor(salida["index"])
        pred_int = float(out.reshape(-1)[0])
        errores.append(abs(pred_float - pred_int))

    mae = float(np.mean(errores))
    print(
        f"[quant] Validación INT8 vs float (50 muestras): MAE = {mae:.5f} "
        f"({'OK' if mae < 0.02 else '¡REVISAR!'})"
    )
    return mae


# ------------------------- Exportación de headers ---------------------------

def _generar_features_header(vocabulario: list, maximos: list, N: int = N_FEATURES) -> str:
    """Header C++ con el vocabulario de bigramas para reproducir las features."""
    lin_aa = "  " + ", ".join(f"0x{a:02x}" for a, _ in vocabulario)
    lin_bb = "  " + ", ".join(f"0x{b:02x}" for _, b in vocabulario)
    lin_mx = "  " + ", ".join(str(m) for m in maximos)
    return (
        "// Vocabulario de bigramas de bytes y máximos normalizadores.\n"
        "// Generado por scripts/train_esp32_detector.py.\n"
        "// Lo consume calcular_features() en main.cpp (entrada del modelo).\n"
        f"// {N} features: [0]=longitud normalizada, [1..{N - 1}]=bigramas.\n"
        "#ifndef FEATURES_DATA_H_\n"
        "#define FEATURES_DATA_H_\n\n"
        "#include <Arduino.h>\n\n"
        f"constexpr int kNumFeatures = {N};\n"
        f"constexpr uint8_t g_vocab_a[] = {{\n{lin_aa}\n}};\n"
        f"constexpr uint8_t g_vocab_b[] = {{\n{lin_bb}\n}};\n"
        f"constexpr uint16_t g_vocab_max[] = {{\n{lin_mx}\n}};\n\n"
        f"#endif // FEATURES_DATA_H_\n"
    )


def _guardar_features_json(vocabulario: list, maximos: list, ruta: Path) -> None:
    datos = {
        "n_features": N_FEATURES,
        "limite_long": LIMITE_LONG,
        "bigramas": [[int(a), int(b)] for a, b in vocabulario],
        "maximos": maximos,
    }
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _abrir(argv)

    try:
        import numpy as np
    except ImportError:
        print(_AYUDA_DEPS, file=sys.stderr)
        return 1

    dataset = Path(args.dataset)
    if not dataset.exists():
        print(f"ERROR: no existe el dataset '{dataset}'.", file=sys.stderr)
        return 1

    salida_dir = Path(args.outdir)

    print(f"[dataset] {dataset}")
    textos, labels = _cargar_datos(dataset)
    _resumen_dataset(textos, labels)

    vocabulario, maximos = _derivar_vocabulario(textos)
    print(f"[features] {len(vocabulario)} bigramas de {N_FEATURES - 1} usados.")

    X = _construir_matriz(textos, vocabulario, maximos)

    modelo, historia = _entrenar(X, labels, args.epochs, args.split, args.seed)
    val_acc = float(historia.history["val_accuracy"][-1])
    acc = float(historia.history["accuracy"][-1])
    print(f"[train] acc={acc:.3f} | val_acc={val_acc:.3f} | épocas={len(historia.history['loss'])}")

    if args.dry_run:
        print("[dry-run] No se escribió ningún artefacto.")
        return 0

    # ----- Guardar artefactos -----
    salida_dir.mkdir(parents=True, exist_ok=True)
    ruta_keras = salida_dir / "detector.keras"
    ruta_calib = salida_dir / "calibracion.npy"
    ruta_features = salida_dir / "features.json"
    ruta_tflite = salida_dir / "detector.tflite"

    modelo.save(str(ruta_keras))
    np.save(ruta_calib, X[:200])
    _guardar_features_json(vocabulario, maximos, ruta_features)
    print(f"[save] {ruta_keras}")
    print(f"[save] {ruta_calib}")
    print(f"[save] {ruta_features}")

    # ----- Convertir a TFLite (siempre se valida la cuantización) -----
    blob = _convertir_a_tflite(
        modelo, quantize=True, full_int8=False, representative_data=X[:200]
    )
    ruta_tflite.write_bytes(blob)
    print(f"[save] {ruta_tflite}")
    _validar_cuantizacion(modelo, X, blob)

    if not args.export:
        print("[ok] Entrenamiento y validación de cuantización completados.")
        print("     Añade --export para regenerar los headers C++ del firmware.")
        return 0

    # ----- Headers del firmware -----
    ruta_model_h = RUTA_FIRMWARE / "model_data.h"
    ruta_features_h = RUTA_FIRMWARE / "features_data.h"
    ruta_model_h.parent.mkdir(parents=True, exist_ok=True)
    ruta_model_h.write_text(_generar_header(blob), encoding="utf-8")
    ruta_features_h.write_text(_generar_features_header(vocabulario, maximos), encoding="utf-8")
    print(f"[export] {ruta_model_h}")
    print(f"[export] {ruta_features_h}")
    _resumen(blob, None)
    return 0


if __name__ == "__main__":
    sys.exit(main())

