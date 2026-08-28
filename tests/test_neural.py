"""Tests del cerebro neural: tokenizador BPE, generación GPT y guardas de calidad."""
import random

from app.ai.neural.brain import NeuralBrain
from app.ai.neural.transformer.gpt_model import GPTModel
from app.ai.neural.transformer.tokenizer_bpe import BPETokenizer

# --- Tokenizador BPE: tokens de rol atómicos ---------------------------------


def test_tokens_especiales_atomicos_al_codificar(tmp_path):
    tokenizer = BPETokenizer(vocab_size=128)
    tokenizer.train([
        "<user>Hola<assistant>Hola, soy ARIA.<eos>",
        "<user>Adiós<assistant>Hasta luego.<eos>",
    ])
    ids = tokenizer.encode("<user>Hola<assistant>")
    assert ids[0] == 5  # <user>
    assert ids[-1] == 6  # <assistant>
    assert 3 in tokenizer.encode("Hola, soy ARIA.<eos>")  # <eos>


def test_decode_roundtrip_omite_marcadores():
    tokenizer = BPETokenizer(vocab_size=128)
    tokenizer.train(["<user>Hola<assistant>Hola, soy ARIA.<eos>"])
    texto = "Hola, soy ARIA."
    assert tokenizer.decode(tokenizer.encode(texto)) == texto


def test_merges_no_absorben_tokens_especiales():
    tokenizer = BPETokenizer(vocab_size=128)
    tokenizer.train(["<user>Hola<assistant>Hola<eos>"] * 25)
    ids = tokenizer.encode("<user>Hola<assistant>Hola<eos>")
    # Los marcadores conservan su id propio aunque el corpus sea repetitivo
    assert ids[0] == 5
    assert 6 in ids
    assert ids[-1] == 3


# --- GPTModel: muestreo con máscara y corte mínimo ---------------------------


def _modelo_tiny() -> GPTModel:
    return GPTModel(vocab_size=64, embed_dim=8, num_heads=2, num_layers=1,
                    max_seq_len=64)


def test_generate_respeta_mascara_de_vocabulario():
    modelo = _modelo_tiny()
    random.seed(42)
    ids = modelo.generate([5, 8, 6], max_new_tokens=20, temperature=0.6,
                          vocab_mask_size=20)
    nuevos = ids[3:]
    assert len(nuevos) > 0
    # Sin IDs fuera del vocabulario real ni tokens de control (<unk>,<user>,<assistant>)
    assert all(0 <= i < 20 for i in nuevos)
    assert not any(i in (1, 5, 6) for i in nuevos)


def test_generate_min_new_tokens_evita_corte_temprano():
    modelo = _modelo_tiny()
    random.seed(7)
    ids = modelo.generate([5, 8, 6], max_new_tokens=12, temperature=0.2,
                          vocab_mask_size=64, repetition_penalty=1.0,
                          min_new_tokens=8)
    nuevos = ids[3:]
    assert len(nuevos) >= 8
    # Si se cortó antes del máximo, solo pudo ser por <eos> o <pad> ya cumplidos
    if len(nuevos) < 12:
        assert nuevos[-1] in (0, 3)


# --- Guarda de calidad de la respuesta (brain._is_usable_response) -----------


def _brain_tmp(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    brain = NeuralBrain(tmp_path)
    brain.initialize()
    return brain


def test_usable_response_acepta_texto_real(tmp_path):
    brain = _brain_tmp(tmp_path)
    assert brain._is_usable_response("Hola, soy ARIA. ¿En qué puedo ayudarte?")
    assert brain._is_usable_response("De nada, para eso estoy")


def test_usable_response_rechaza_basura(tmp_path):
    brain = _brain_tmp(tmp_path)
    assert not brain._is_usable_response("")
    assert not brain._is_usable_response("abc")
    assert not brain._is_usable_response(">os>os>>os>")
    assert not brain._is_usable_response("eeeeeeeeeeeeeeee")
    assert not brain._is_usable_response("eoseoseoseos")
    assert not brain._is_usable_response("a b")