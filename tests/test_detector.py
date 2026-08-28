"""Tests del detector de frases del nodo de IA local (ESP32).

Valida la extracción de features de n-gramas (64 dims) que usa la red densa
64 -> 32 -> 1 y que el firmware C reproduce con ``calcular_features()``.

Estos tests no requieren TensorFlow: solo ejercitan la parte pura de Python
(vocabulario + features + etiquetado binario).
"""
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from train_esp32_detector import (
    ETIQUETA_POSITIVA,
    N_FEATURES,
    _cargar_datos,
    _derivar_vocabulario,
    extraer_features,
)

# --- Vocabulario ------------------------------------------------------------

def test_num_features_es_64():
    # 1 (longitud) + 63 bigramas = 64 dims de entrada de la red.
    assert N_FEATURES == 64


def test_vocabulario_tiene_63_bigramas():
    textos = ["Hola", "Abre el navegador", "Gracias", "Qué hora es"]
    vocabulario, maximos = _derivar_vocabulario(textos)
    assert len(vocabulario) == 63
    assert len(maximos) == 63


def test_vocabulario_determinista():
    textos = ["Hola ARIA", "Buenos días", "Ejecuta pwd", "No funciona"]
    v1, m1 = _derivar_vocabulario(textos)
    v2, m2 = _derivar_vocabulario(textos)
    assert v1 == v2
    assert m1 == m2


# --- Feature extraction -----------------------------------------------------

def test_features_tienen_64_dims_en_rango():
    vocabulario, maximos = _derivar_vocabulario(["Hola", "Abre el navegador"])
    feats = extraer_features("Abre el navegador", vocabulario, maximos)
    assert len(feats) == N_FEATURES
    assert all(0.0 <= x <= 1.0 for x in feats)


def test_features_son_deterministas():
    vocabulario, maximos = _derivar_vocabulario(["Ejecuta ls", "Lee el archivo"])
    f1 = extraer_features("Ejecuta ls", vocabulario, maximos)
    f2 = extraer_features("Ejecuta ls", vocabulario, maximos)
    assert f1 == f2


def test_longitud_normalizada():
    vocabulario, maximos = _derivar_vocabulario(["a", "b"])
    corta = extraer_features("ab", vocabulario, maximos)
    larga = extraer_features("ab" * 10, vocabulario, maximos)
    # La feature 0 (longitud) crece con la frase (salvo saturación).
    assert larga[0] >= corta[0]


def test_features_utf8_no_se_rompen():
    # Caracteres multi-byte UTF-8 deben procesarse sin excepción.
    vocabulario, maximos = _derivar_vocabulario(["éxito", "¿Qué tal?", "mañana"])
    feats = extraer_features("¿Qué tal estás, mañana?", vocabulario, maximos)
    assert len(feats) == N_FEATURES


def test_features_se_recortan_a_1():
    # Una frase que repite un bigrama más veces que el máximo del corpus no debe
    # exceder 1.0 (paridad con el recorte en calcular_features() de main.cpp).
    vocabulario, _maximos = _derivar_vocabulario(["ab"])
    feats = extraer_features("ab" * 500, vocabulario, _maximos)
    assert all(0.0 <= x <= 1.0 for x in feats)
    # La feature del bigrama "ab" satura en 1.0 (500 repeticiones > máx=1).
    idx = vocabulario.index((0x61, 0x62))  # "ab" en ASCII
    assert feats[idx + 1] == 1.0


# --- Etiquetado binario -----------------------------------------------------

def test_cargar_datos_etiqueta_comando_como_positivo(tmp_path):
    ds = tmp_path / "ds.json"
    ds.write_text(
        json.dumps({
            "texts": ["Hola", "Abre el navegador", "Gracias", "Ejecuta ls"],
            "intents": ["SALUDO", "COMANDO", "AGRADECIMIENTO", "COMANDO"],
        }),
        encoding="utf-8",
    )
    textos, labels = _cargar_datos(ds)
    assert textos == ["Hola", "Abre el navegador", "Gracias", "Ejecuta ls"]
    assert list(labels) == [0.0, 1.0, 0.0, 1.0]
    assert ETIQUETA_POSITIVA == "COMANDO"


def test_cargar_datos_desbalance_lanza(tmp_path):
    ds = tmp_path / "ds.json"
    ds.write_text(
        json.dumps({"texts": ["Hola"], "intents": ["SALUDO", "COMANDO"]}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        # texts e intents de tamaños distintos.
        _cargar_datos(ds)
