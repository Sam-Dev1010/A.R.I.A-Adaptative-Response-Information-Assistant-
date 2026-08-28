#!/usr/bin/env python3
"""Convierte una red neuronal entrenada a TensorFlow Lite para el ESP32.

Toma un modelo Keras (`.keras` o `.h5`) y genera un archivo ``.tflite``
cuantizado a INT8, listo para ejecutarse con ``TensorFlowLite_ESP32`` en el
nodo de IA local. El binario se empaqueta en el header C++ del firmware:

    firmware/lacal-IA-node/src/model_data.h       (repo A.R.I.A)
    src/model_data.h                              (repo Local-IA-ESP32)

El header se sobreescribe en cada ejecución y mantiene la firma del esquema
FlatBuffers de TensorFlow Lite (``TFL3``).

Cuantización:
- Por defecto se usa ``Optimize.DEFAULT`` con dataset representativo: pesos y
  activaciones van a INT8, pero la entrada/salida siguen en float. El firmware
  escribe en ``input->data.f[]`` y lee ``output->data.f[]`` sin cambios.
- ``--full-int8`` además obliga a I/O en ``int8`` (el firmware deberá usar los
  parámetros de cuantización con ``input->data.int8[]``).

Ejemplos:
    python scripts/export_tflite.py red/entrenada.keras
    python scripts/export_tflite.py red/entrenada.h5 \\
        --representative datos/calibracion.npy -n 200
    python scripts/export_tflite.py red/entrenada.keras \\
        -o firmware/lacal-IA-node/src/model_data.h
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

_ENCABEZADO = """\
#ifndef MODEL_DATA_H_
#define MODEL_DATA_H_

#include <Arduino.h>

// Modelo cuantizado a TensorFlow Lite INT8, generado por:
//   scripts/export_tflite.py
// Advertencia: este archivo se SOBRESCRIBE en cada exportacion. No edits a mano.
const unsigned char g_model[] PROGMEM = {
"""

_AYUDA_DEPS = (
    "Faltan dependencias de Python. Instala TensorFlow y NumPy:\n"
    "  pip install tensorflow numpy\n"
    "y vuelve a ejecutar el script."
)


def _default_output() -> Path:
    """Header C++ por defecto del firmware según el repo donde viva el script."""
    root = Path(__file__).resolve().parent.parent
    return root / "firmware" / "lacal-IA-node" / "src" / "model_data.h"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convierte un modelo .keras/.h5 a TensorFlow Lite INT8 y "
            "regenera el header C++ del firmware ESP32."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Ejemplo: python scripts/export_tflite.py red/entrenada.keras",
    )
    parser.add_argument(
        "model",
        metavar="MODELO",
        help="Ruta al modelo Keras entrenado (.keras o .h5).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(_default_output()),
        help=(
            "Header C++ de salida. Por defecto "
            "firmware/lacal-IA-node/src/model_data.h."
        ),
    )
    parser.add_argument(
        "-r",
        "--representative",
        metavar="DATOS",
        help=(
            "Dataset representativo para la calibración INT8 (.npy o .npz). "
            "Sin este archivo se generan muestras aleatorias (solo para probar "
            "el pipeline; la precisión real necesita datos de entrenamiento)."
        ),
    )
    parser.add_argument(
        "-n",
        "--samples",
        type=int,
        default=100,
        metavar="N",
        help="Número de muestras usadas para calibrar (por defecto 100).",
    )
    parser.add_argument(
        "--full-int8",
        action="store_true",
        help=(
            "Cuantización INT8 total: entrada y salida también en int8. "
            "El firmware deberá usar los parámetros de cuantización."
        ),
    )
    parser.add_argument(
        "--no-quant",
        action="store_true",
        help="Exporta sin cuantización (float32). Solo para depuración.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Convierte y valida, pero no escribe el header.",
    )
    return parser.parse_args(argv)


def _cargar_representative(ruta: str | Path, samples: int):
    """Carga un dataset representativo (.npy o .npz) para calibración INT8.

    Devuelve un array NumPy (N, ...) recortado a ``samples`` muestras.
    """
    import numpy as np

    ruta = Path(ruta)
    if not ruta.exists():
        raise SystemExit(f"No existe el dataset representativo: {ruta}")
    data = np.load(ruta)
    array = data
    if isinstance(data, np.lib.npyio.NpzFile):
        if "arr_0" in data:
            array = data["arr_0"]
        elif data.files:
            array = data[data.files[0]]
            print(f"[datos] Usando '{data.files[0]}' del .npz")
        data.close()
    muestras = np.asarray(array)
    if muestras.size == 0:
        raise SystemExit("El dataset representativo está vacío.")
    print(f"[datos] {len(muestras)} muestras cargadas desde {ruta}")
    return muestras[:samples]


def _verificar_firma_tfl3(blob: bytes) -> None:
    """Comprueba la cabecera FlatBuffers: el identificador 'TFL3' en el offset 4.

    Observa que la ante-palabra (offset 0, tamaño del preámbulo) puede variar
    entre versiones de TensorFlow (``0x1c``/28 o ``0x20``/32); el contrato que
    exige TensorFlow Lite Micro (y el firmware, ``isValidTFLiteModel()``) es
    únicamente el identificador ``TFL3`` en la posición 4.
    """
    if len(blob) < 32 or blob[4:8] != b"TFL3":
        raise SystemExit(
            "ERROR: el binario generado no tiene la firma TFL3 de TensorFlow "
            "Lite. No se escribe el header."
        )


def _generar_header(blob: bytes) -> str:
    """Empaqueta el binario como arreglo PROGMEM en un header C++."""
    _verificar_firma_tfl3(blob)
    bytes_list = [f"0x{b:02x}" for b in blob]
    lineas = [
        "  " + ", ".join(bytes_list[i : i + 12])
        for i in range(0, len(bytes_list), 12)
    ]
    cuerpo = ",\n".join(lineas)
    pie = (
        f"}};\n"
        f"const int g_model_len = {len(blob)};\n"
        f"\n#endif\n"
    )
    return _ENCABEZADO + "\n" + cuerpo + "\n" + pie


def _convertir_a_tflite(
    keras_model,
    *,
    quantize: bool = True,
    full_int8: bool = False,
    representative_data=None,
) -> bytes:
    """Convierte un modelo Keras a un binario TensorFlow Lite.

    Función reutilizable (importable) que encapsula todo el pipeline de
    conversión y cuantización:

    - ``quantize``: por defecto ``True``. Cuantiza pesos y activaciones a INT8
      con ``Optimize.DEFAULT`` manteniendo I/O en float (el firmware alimenta
      ``input->data.f[]`` y lee ``output->data.f[]`` sin cambios).
    - ``full_int8``: además obliga a que la entrada/salida vayan en int8.
    - ``representative_data``: iterable de arrays (N, ...) usado para calibrar
      la cuantización. Si es ``None`` se usa un dataset aleatorio de prueba.

    Devuelve el blob FlatBuffers listo para validar con
    ``_verificar_firma_tfl3`` y empaquetar con ``_generar_header``.
    """
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if quantize:
        if representative_data is None:
            representative_data = _representative_random(keras_model, 100)
        converter.representative_dataset = _representative_wrapper(representative_data)
        if full_int8:
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

    print("[tflite] Convirtiendo y cuantizando…")
    blob = converter.convert()
    _verificar_firma_tfl3(blob)
    return blob


def _representative_random(keras_model, samples: int):
    """Muestras aleatorias 0..1 a partir de la forma de entrada (placeholder)."""
    import numpy as np

    input_shape = list(keras_model.inputs[0].shape)
    if any(dim is None for dim in input_shape):
        raise SystemExit(
            "La entrada del modelo tiene dimensión dinámica (None). "
            "Fija la forma o pasa explícitamente batch=1 antes de exportar."
        )
    sample_shape = input_shape[1:]
    print(
        "[datos] Sin dataset representativo: calibrando con muestras aleatorias "
        "(solo prueba del pipeline)."
    )
    return np.random.default_rng(0).random((samples, *sample_shape))


def _representative_wrapper(representative_data) -> Callable[[], list]:
    """Envuelve un dataset (N, ...) en el generador que espera TF Lite."""
    import numpy as np

    muestras = np.asarray(representative_data, dtype=np.float32)

    def generar():
        for muestra in muestras:
            yield [muestra]

    return generar


def _resumen(blob: bytes, args: argparse.Namespace) -> None:
    """Imprime un resumen del modelo exportado (tensores y tamaño en flash)."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_content=blob)
    interpreter.allocate_tensors()
    entrada = interpreter.get_input_details()[0]
    salida = interpreter.get_output_details()[0]
    detalles = interpreter.get_tensor_details()

    print("\nResumen del modelo exportado:")
    print(f"  Tamaño en flash  : {len(blob)} bytes ({len(blob) / 1024:.1f} KiB)")
    if len(blob) > 1024 * 1024:
        print(
            "  ¡OJO! >1 MiB puede no caber en la partición APP por defecto "
            "del ESP32. Usa un modelo más pequeño o amplía la partición en "
            "platformio.ini."
        )
    print("  Firma TFL3       : OK")
    print(f"  Entrada          : {entrada['dtype']} {entrada['shape']}")
    q_in = entrada.get("quantization_parameters", {})
    if q_in.get("scales"):
        print(
            f"  Cuantización entr: scale={q_in['scales']} "
            f"zero_point={q_in['zero_points']}"
        )
    print(f"  Salida           : {salida['dtype']} {salida['shape']}")
    cuantos = sum(
        1 for t in detalles
        if t["dtype"].itemsize == 1 or t["quantization"][0] != 0
    )
    print(f"  Tensores cuantiz : {cuantos} de {len(detalles)}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        import numpy as np  # noqa: F401
        import tensorflow as tf
    except ImportError:
        print(_AYUDA_DEPS, file=sys.stderr)
        return 1

    modelo = Path(args.model)
    if not modelo.exists():
        print(f"ERROR: no existe el modelo '{modelo}'.", file=sys.stderr)
        return 1

    print(f"[modelo] Cargando '{modelo}' …")
    keras_model = tf.keras.models.load_model(str(modelo))

    representative_data = None
    if not args.no_quant and args.representative:
        representative_data = _cargar_representative(args.representative, args.samples)

    blob = _convertir_a_tflite(
        keras_model,
        quantize=not args.no_quant,
        full_int8=args.full_int8,
        representative_data=representative_data,
    )

    if args.dry_run:
        print("[dry-run] Header NO escrito.")
        _resumen(blob, args)
        return 0

    salida = Path(args.output)
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(_generar_header(blob), encoding="utf-8")

    print(f"[ok] Header escrito: {salida}")
    print(f"[ok] g_model_len = {len(blob)} bytes ")
    _resumen(blob, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())