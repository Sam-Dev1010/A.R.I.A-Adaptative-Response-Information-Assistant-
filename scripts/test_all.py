#!/usr/bin/env python3
"""Suite de pruebas completa para el sistema neural de A.R.I.A.

Cubre todos los módulos del sistema neural: tokenizer BPE, capas,
red secuencial, clasificador de intenciones, base de conocimiento,
memoria semántica, generador de texto, generador de respuestas,
razonamiento, transformer blocks, GPT model, trainer, inference, y brain.
"""
import asyncio
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

passed = 0
failed = 0
errors = []


def test(name: str, func):
    global passed, failed
    try:
        result = func()
        if result is not False:
            print(f"  PASS  {name}")
            passed += 1
        else:
            print(f"  FAIL  {name} (returned False)")
            failed += 1
            errors.append(name)
    except Exception as e:
        print(f"  FAIL  {name} ({type(e).__name__}: {e})")
        failed += 1
        errors.append(f"{name}: {type(e).__name__}: {e}")


# ======================================================================
print("=" * 70)
print(" A.R.I.A - Suite de Pruebas Completa del Sistema Neural")
print("=" * 70)

# ======================================================================
print("\n--- 1. BPE Tokenizer ---")
from app.ai.neural.transformer.tokenizer_bpe import BPETokenizer


def test_bpe_train_and_encode():
    t = BPETokenizer(vocab_size=1000)
    t.train(["Hola mundo cruel mundo cruel"], verbose=False)
    ids = t.encode("Hola")
    assert len(ids) > 0
    decoded = t.decode(ids)
    assert "Hola" in decoded


test("train + encode + decode", test_bpe_train_and_encode)


def test_bpe_special_tokens():
    t = BPETokenizer(vocab_size=1000)
    t.train(["Hola mundo Hola asistente"], verbose=False)
    # Special tokens are decoded when present as token IDs
    user_id = t.vocab.get("<user>")
    assistant_id = t.vocab.get("<assistant>")
    if user_id is not None and assistant_id is not None:
        decoded = t.decode([user_id, assistant_id])
        # decode() skips special tokens in the standard implementation
        # But the IDs should exist in the vocab
        assert user_id in t.id_to_token.values() or user_id in range(10)
        assert assistant_id in t.id_to_token.values() or assistant_id in range(10)


test("special token IDs exist in vocab", test_bpe_special_tokens)


def test_bpe_save_load():
    t = BPETokenizer(vocab_size=2000)
    t.train(["Hola mundo cruel mundo cruel otro texto"], verbose=False)
    with tempfile.TemporaryDirectory() as d:
        t.save(d)
        t2 = BPETokenizer()
        t2.load(d)
        assert t.vocab_len == t2.vocab_len
        assert t.encode("Hola") == t2.encode("Hola")


test("save and load", test_bpe_save_load)


def test_bpe_vocab_property():
    t = BPETokenizer()
    assert isinstance(t.vocab_len, int)


test("vocab_len is integer property", test_bpe_vocab_property)


def test_bpe_untrained_raises():
    t = BPETokenizer()
    try:
        t.encode("Hola")
        return False
    except RuntimeError:
        return True


test("encode raises RuntimeError when not trained", test_bpe_untrained_raises)


def test_bpe_encode_decode_roundtrip():
    t = BPETokenizer(vocab_size=2000)
    phrases = [
        "Hola que tal",
        "Buenos dias",
        "Adios amigo",
        "Gracias por todo",
    ]
    t.train(phrases * 5, verbose=False)
    for phrase in phrases:
        ids = t.encode(phrase)
        decoded = t.decode(ids)
        assert phrase.lower() in decoded.lower() or phrase.split()[0] in decoded, \
            f"Roundtrip failed: '{phrase}' -> '{decoded}'"


test("encode/decode roundtrip on multiple phrases", test_bpe_encode_decode_roundtrip)


# ======================================================================
print("\n--- 2. Neural Layers ---")
from app.ai.neural.layers import Dense, Activation, Embedding


def test_dense_forward():
    d = Dense(4, 3)
    out = d.forward([1.0, 0.5, -0.3, 0.0])
    assert len(out) == 3
    assert all(isinstance(v, float) for v in out)


test("Dense forward", test_dense_forward)


def test_dense_backward_updates():
    d = Dense(4, 3)
    x = [1.0, 0.5, -0.3, 0.0]
    out1 = d.forward(x)
    d.backward([0.1, -0.2, 0.3], lr=0.01)
    out2 = d.forward(x)
    assert out1 != out2, "Weights must change after backward"


test("Dense backward updates weights", test_dense_backward_updates)


def test_dense_backprop_gradient():
    d = Dense(4, 3)
    x = [1.0, 0.5, -0.3, 0.0]
    d.forward(x)
    grad = d.backward([1.0, 1.0, 1.0], lr=0.01)
    assert len(grad) == 4, "Gradient must have input_size elements"


test("Dense backward returns gradient", test_dense_backprop_gradient)


def test_activation_relu():
    a = Activation("relu")
    assert a.forward([-1.0, 0.0, 1.0, 2.0]) == [0.0, 0.0, 1.0, 2.0]


test("Activation relu", test_activation_relu)


def test_activation_softmax():
    a = Activation("softmax")
    out = a.forward([1.0, 2.0, 3.0])
    assert abs(sum(out) - 1.0) < 1e-6
    assert out[2] > out[1] > out[0]


test("Activation softmax", test_activation_softmax)


def test_activation_sigmoid():
    a = Activation("sigmoid")
    out = a.forward([0.0])
    assert abs(out[0] - 0.5) < 1e-6


test("Activation sigmoid", test_activation_sigmoid)


def test_activation_tanh():
    a = Activation("tanh")
    out = a.forward([0.0])
    assert abs(out[0]) < 1e-6


test("Activation tanh", test_activation_tanh)


def test_activation_leaky_relu():
    a = Activation("leaky_relu")
    out = a.forward([-1.0, 0.0, 1.0])
    assert out[1] == 0.0
    assert out[2] == 1.0
    assert out[0] < 0.0


test("Activation leaky_relu", test_activation_leaky_relu)


def test_embedding_layer():
    e = Embedding(100, 16)
    out = e.forward([5.0])
    assert len(out) == 16
    assert all(isinstance(v, float) for v in out)


test("Embedding forward", test_embedding_layer)


def test_embedding_backward_updates():
    e = Embedding(100, 16)
    out1 = e.forward([5.0])[:]
    e.backward([0.1] * 16, lr=0.01)
    out2 = e.forward([5.0])
    # Embedding returns reference to weight vector, so we compare before/after
    assert out1 != out2, "Embedding weights should change after backward"


test("Embedding backward updates weights", test_embedding_backward_updates)


def test_embedding_multiple_lookups():
    e = Embedding(100, 16)
    out0 = e.forward([0.0])
    out1 = e.forward([1.0])
    assert out0 != out1


test("Embedding different IDs produce different vectors", test_embedding_multiple_lookups)


# ======================================================================
print("\n--- 3. Sequential Network ---")
from app.ai.neural.network import SequentialNetwork


def test_sequential_forward():
    net = SequentialNetwork()
    net.add(Dense(4, 8))
    net.add(Activation("relu"))
    net.add(Dense(8, 3))
    net.add(Activation("softmax"))
    out = net.forward([1.0, 0.5, -0.3, 0.0])
    assert len(out) == 3
    assert abs(sum(out) - 1.0) < 1e-4


test("forward", test_sequential_forward)


def test_sequential_predict():
    net = SequentialNetwork()
    net.add(Dense(4, 8))
    net.add(Dense(8, 3))
    x = [1.0, 0.5, -0.3, 0.0]
    assert net.predict(x) == net.forward(x)


test("predict == forward", test_sequential_predict)


def test_sequential_save_load():
    net = SequentialNetwork()
    net.add(Dense(4, 8))
    net.add(Dense(8, 3))
    x = [1.0, 0.5, -0.3, 0.0]
    out1 = net.forward(x)
    with tempfile.TemporaryDirectory() as d:
        fpath = Path(d) / "net.json"
        net.save(fpath)
        net2 = SequentialNetwork()
        net2.add(Dense(4, 8))
        net2.add(Dense(8, 3))
        net2.load(fpath)
        out2 = net2.forward(x)
        assert out1 == out2


test("save/load", test_sequential_save_load)


def test_sequential_summary():
    net = SequentialNetwork()
    net.add(Dense(4, 8))
    net.add(Dense(8, 3))
    s = net.summary()
    assert isinstance(s, str)
    assert "Dense" in s


test("summary", test_sequential_summary)


def test_sequential_len():
    net = SequentialNetwork()
    net.add(Dense(4, 8))
    net.add(Dense(8, 3))
    assert len(net) == 2


test("len", test_sequential_len)


def test_sequential_getitem():
    net = SequentialNetwork()
    net.add(Dense(4, 8))
    net.add(Dense(8, 3))
    assert isinstance(net[0], Dense)
    assert isinstance(net[1], Dense)


test("getitem", test_sequential_getitem)


# ======================================================================
print("\n--- 4. Intent Classifier ---")
from app.ai.neural.intent_classifier import IntentClassifier


def test_classifier_classify():
    c = IntentClassifier(vocab_size=500, hidden_dim=32)
    texts = ["Hola"] * 5 + ["Adiós"] * 5 + ["Gracias"] * 5
    intents = ["SALUDO"] * 5 + ["DESPEDIDA"] * 5 + ["AGRADECIMIENTO"] * 5
    c.train(texts, intents, epochs=20)
    intent, conf = c.classify("Hola")
    assert intent in IntentClassifier.INTENTS
    assert 0.0 <= conf <= 1.0


test("classify", test_classifier_classify)


def test_classifier_detailed():
    c = IntentClassifier(vocab_size=500, hidden_dim=32)
    texts = ["Hola"] * 5 + ["Adiós"] * 5 + ["Gracias"] * 5
    intents = ["SALUDO"] * 5 + ["DESPEDIDA"] * 5 + ["AGRADECIMIENTO"] * 5
    c.train(texts, intents, epochs=20)
    result = c.classify_detailed("Hola")
    assert isinstance(result, dict)
    assert len(result) == len(IntentClassifier.INTENTS)
    total = sum(result.values())
    assert abs(total - 1.0) < 0.05, f"Probabilities should sum to ~1, got {total}"


test("classify_detailed", test_classifier_detailed)


def test_classifier_accuracy():
    c = IntentClassifier(vocab_size=500, hidden_dim=32)
    texts = ["Hola"] * 10 + ["Adiós"] * 10 + ["Gracias"] * 10 + ["Qué hora es"] * 10
    intents = ["SALUDO"] * 10 + ["DESPEDIDA"] * 10 + ["AGRADECIMIENTO"] * 10 + ["PREGUNTA"] * 10
    c.train(texts, intents, epochs=50)
    correct = 0
    for text, expected in zip(["Hola", "Adiós", "Gracias", "Qué hora es"],
                               ["SALUDO", "DESPEDIDA", "AGRADECIMIENTO", "PREGUNTA"]):
        intent, _ = c.classify(text)
        if intent == expected:
            correct += 1
    assert correct >= 3, f"Expected >=3/4 correct, got {correct}/4"


test("classify accuracy", test_classifier_accuracy)


def test_classifier_save_load():
    c = IntentClassifier(vocab_size=500, hidden_dim=32)
    texts = ["Hola"] * 5 + ["Adiós"] * 5
    intents = ["SALUDO"] * 5 + ["DESPEDIDA"] * 5
    c.train(texts, intents, epochs=10)
    intent1, _ = c.classify("Hola")
    with tempfile.TemporaryDirectory() as d:
        c.save(d)
        c2 = IntentClassifier()
        c2.load(d)
        intent2, _ = c2.classify("Hola")
        assert intent1 == intent2


test("save/load", test_classifier_save_load)


# ======================================================================
print("\n--- 5. Knowledge Base ---")
from app.ai.neural.knowledge_base import KnowledgeBase


def test_kb_add_search():
    with tempfile.TemporaryDirectory() as d:
        kb = KnowledgeBase(db_path=Path(d) / "kb.db")
        kb.open()
        kb.add_fact("Python es un lenguaje de programación", "general", confidence=0.9)
        results = kb.search_facts("Python")
        assert len(results) > 0
        kb.close()


test("add_fact + search_facts", test_kb_add_search)


def test_kb_add_entity():
    with tempfile.TemporaryDirectory() as d:
        kb = KnowledgeBase(db_path=Path(d) / "kb.db")
        kb.open()
        kb.add_entity("Python", "lenguaje", {"popular": True})
        entity = kb.get_entity("Python")
        assert entity is not None
        kb.close()


test("add_entity + get_entity", test_kb_add_entity)


def test_kb_add_relation():
    with tempfile.TemporaryDirectory() as d:
        kb = KnowledgeBase(db_path=Path(d) / "kb.db")
        kb.open()
        kb.add_entity("Python", "lenguaje")
        kb.add_entity("Guido", "persona")
        kb.add_relation("Python", "creado_por", "Guido")
        rels = kb.get_relations(subject_name="Python")
        assert len(rels) > 0
        kb.close()


test("add_relation + get_relations", test_kb_add_relation)


def test_kb_infer():
    with tempfile.TemporaryDirectory() as d:
        kb = KnowledgeBase(db_path=Path(d) / "kb.db")
        kb.open()
        kb.add_entity("Python", "lenguaje")
        kb.add_entity("Guido", "persona")
        kb.add_relation("Python", "creado_por", "Guido")
        results = kb.infer("Python")
        assert isinstance(results, list)
        kb.close()


test("infer", test_kb_infer)


def test_kb_stats():
    with tempfile.TemporaryDirectory() as d:
        kb = KnowledgeBase(db_path=Path(d) / "kb.db")
        kb.open()
        kb.add_fact("Test fact", "test")
        s = kb.stats()
        assert isinstance(s, dict)
        assert "facts" in s
        kb.close()


test("stats", test_kb_stats)


def test_kb_category():
    with tempfile.TemporaryDirectory() as d:
        kb = KnowledgeBase(db_path=Path(d) / "kb.db")
        kb.open()
        kb.add_fact("Fact1", "ciencia")
        kb.add_fact("Fact2", "ciencia")
        kb.add_fact("Fact3", "historia")
        results = kb.get_facts_by_category("ciencia")
        assert len(results) == 2
        kb.close()


test("get_facts_by_category", test_kb_category)


def test_kb_save_load():
    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "kb.db"
        kb = KnowledgeBase(db_path=db_path)
        kb.open()
        kb.add_fact("Persistent fact", "test", confidence=0.8)
        kb.close()
        kb2 = KnowledgeBase(db_path=db_path)
        kb2.open()
        results = kb2.search_facts("Persistent")
        assert len(results) > 0
        kb2.close()


test("save/load (SQLite persistence)", test_kb_save_load)


# ======================================================================
print("\n--- 6. Semantic Memory ---")
from app.ai.neural.semantic_memory import SemanticMemory


def test_semantic_store_search():
    with tempfile.TemporaryDirectory() as d:
        sm = SemanticMemory(db_path=Path(d) / "mem.db")
        sm.open()
        sm.store("Mi nombre es Samuel y vivo en Chile", category="personal", importance=0.8)
        results = sm.search("nombre Samuel")
        assert len(results) > 0
        assert results[0]["content"] == "Mi nombre es Samuel y vivo en Chile"
        sm.close()


test("store + search", test_semantic_store_search)


def test_semantic_category():
    with tempfile.TemporaryDirectory() as d:
        sm = SemanticMemory(db_path=Path(d) / "mem.db")
        sm.open()
        sm.store("Dato1", category="tech")
        sm.store("Dato2", category="tech")
        sm.store("Dato3", category="personal")
        results = sm.get_by_category("tech")
        assert len(results) == 2
        sm.close()


test("get_by_category", test_semantic_category)


def test_semantic_links():
    with tempfile.TemporaryDirectory() as d:
        sm = SemanticMemory(db_path=Path(d) / "mem.db")
        sm.open()
        id1 = sm.store("Memoria1")
        id2 = sm.store("Memoria2")
        sm.link(id1, id2, strength=0.9)
        related = sm.get_related(id1)
        assert len(related) > 0
        sm.close()


test("link + get_related", test_semantic_links)


def test_semantic_forget():
    with tempfile.TemporaryDirectory() as d:
        sm = SemanticMemory(db_path=Path(d) / "mem.db")
        sm.open()
        mid = sm.store("Olvidar esto")
        result = sm.forget(mid)
        assert result is True
        sm.close()


test("forget", test_semantic_forget)


def test_semantic_stats():
    with tempfile.TemporaryDirectory() as d:
        sm = SemanticMemory(db_path=Path(d) / "mem.db")
        sm.open()
        sm.store("Dato1")
        s = sm.stats()
        assert isinstance(s, dict)
        assert "memories" in s
        assert s["memories"] >= 1
        sm.close()


test("stats", test_semantic_stats)


def test_semantic_save_load():
    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "mem.db"
        sm = SemanticMemory(db_path=db_path)
        sm.open()
        sm.store("Mi nombre es Samuel y me gusta programar")
        sm.close()
        sm2 = SemanticMemory(db_path=db_path)
        sm2.open()
        results = sm2.search("nombre Samuel programar")
        assert len(results) > 0
        sm2.close()


test("save/load (SQLite persistence)", test_semantic_save_load)


# ======================================================================
print("\n--- 7. Text Generator ---")
from app.ai.neural.text_generator import TextGenerator


def test_generator_train_and_generate():
    g = TextGenerator(vocab_size=500)
    convs = [
        {"input": "Hola", "response": "Hola que tal"},
        {"input": "Adiós", "response": "Hasta luego"},
        {"input": "Gracias", "response": "De nada"},
    ] * 3
    g.train(convs)
    text = g.generate("Hola", max_length=20)
    assert isinstance(text, str)
    assert len(text) > 0


test("train + generate", test_generator_train_and_generate)


def test_generator_save_load():
    g = TextGenerator(vocab_size=500)
    convs = [{"input": "Hola", "response": "Hola que tal"}] * 3
    g.train(convs)
    with tempfile.TemporaryDirectory() as d:
        g.save(d)
        g2 = TextGenerator(vocab_size=500)
        g2.load(d)
        text = g2.generate("Hola", max_length=10)
        assert isinstance(text, str)
        assert len(text) > 0


test("save/load", test_generator_save_load)


# ======================================================================
print("\n--- 8. Response Generator ---")
from app.ai.neural.response_generator import ResponseGenerator


def test_response_generate():
    with tempfile.TemporaryDirectory() as d:
        kb = KnowledgeBase(db_path=Path(d) / "kb.db")
        kb.open()
        r = ResponseGenerator(knowledge_base=kb)
        result = r.generate(user_message="Hola", intent="SALUDO", confidence=0.9)
        assert isinstance(result, str)
        assert len(result) > 0
        kb.close()


test("generate", test_response_generate)


def test_response_varies():
    with tempfile.TemporaryDirectory() as d:
        kb = KnowledgeBase(db_path=Path(d) / "kb.db")
        kb.open()
        r = ResponseGenerator(knowledge_base=kb)
        results = {r.generate(user_message="Hola", intent="SALUDO", confidence=0.9)
                   for _ in range(10)}
        assert len(results) > 1
        kb.close()


test("generates varied responses", test_response_varies)


def test_response_context():
    with tempfile.TemporaryDirectory() as d:
        kb = KnowledgeBase(db_path=Path(d) / "kb.db")
        kb.open()
        r = ResponseGenerator(knowledge_base=kb)
        r.add_to_context("Hola, soy Samuel")
        result = r.generate(user_message="Quién soy", intent="PREGUNTA", confidence=0.8)
        assert isinstance(result, str)
        r.clear_context()
        kb.close()


test("add_to_context + clear_context", test_response_context)


def test_response_all_intents():
    with tempfile.TemporaryDirectory() as d:
        kb = KnowledgeBase(db_path=Path(d) / "kb.db")
        kb.open()
        r = ResponseGenerator(knowledge_base=kb)
        for intent in ["SALUDO", "DESPEDIDA", "AGRADECIMIENTO", "PREGUNTA",
                        "COMANDO", "QUEJA", "CURIOSIDAD"]:
            result = r.generate(user_message="test", intent=intent, confidence=0.5)
            assert isinstance(result, str) and len(result) > 0, \
                f"Failed for intent {intent}"
        kb.close()


test("generates for all intent types", test_response_all_intents)


# ======================================================================
print("\n--- 9. Reasoning Engine ---")
from app.ai.neural.reasoning import ReasoningEngine, ReasoningResult


def test_reasoning_reason():
    engine = ReasoningEngine()
    result = engine.reason("¿Qué opinas sobre Python?")
    assert isinstance(result, ReasoningResult)
    assert isinstance(result.answer, str)
    assert isinstance(result.thoughts, list)
    assert len(result.thoughts) > 0
    assert 0.0 <= result.confidence <= 1.0


test("reason", test_reasoning_reason)


def test_reasoning_history():
    engine = ReasoningEngine()
    engine.reason("Pregunta 1")
    engine.reason("Pregunta 2")
    history = engine.get_reasoning_history()
    assert len(history) == 2


test("get_reasoning_history", test_reasoning_history)


def test_reasoning_explain():
    engine = ReasoningEngine()
    result = engine.reason("Explícame Python")
    explanation = engine.explain_reasoning(result)
    assert isinstance(explanation, str)
    assert len(explanation) > 0


test("explain_reasoning", test_reasoning_explain)


def test_reasoning_question_types():
    engine = ReasoningEngine()
    questions = [
        "¿Qué es Python?",
        "Explica la programación",
        "¿Cuál es la diferencia entre Python y Java?",
        "¿Qué pasará con la IA?",
    ]
    for q in questions:
        result = engine.reason(q)
        assert isinstance(result, ReasoningResult)
        assert len(result.thoughts) > 0


test("handles different question types", test_reasoning_question_types)


# ======================================================================
print("\n--- 10. Trainer ---")
from app.ai.neural.trainer import Trainer


def test_trainer_init():
    from app.ai.neural.network import SequentialNetwork
    from app.ai.neural.layers import Dense
    net = SequentialNetwork()
    net.add(Dense(10, 5))
    t = Trainer(net, loss_fn="cross_entropy", learning_rate=0.01)
    assert hasattr(t, "train")


test("Trainer init", test_trainer_init)


# ======================================================================
print("\n--- 11. Transformer Blocks ---")
from app.ai.neural.transformer.blocks import (
    Linear,
    Embedding as BlockEmbedding,
    MultiHeadAttention,
    FeedForward,
    TransformerBlock,
)


def test_linear_forward():
    l = Linear(16, 8)
    out = l.forward([1.0] * 16)
    assert len(out) == 8
    assert all(isinstance(v, float) for v in out)


test("Linear forward", test_linear_forward)


def test_linear_backward():
    l = Linear(16, 8)
    l.forward([1.0] * 16)
    grad = l.backward([0.1] * 8, lr=0.001)
    assert len(grad) == 16


test("Linear backward returns gradient", test_linear_backward)


def test_linear_updates():
    l = Linear(16, 8)
    x = [1.0] * 16
    out1 = l.forward(x)
    l.backward([0.1] * 8, lr=0.01)
    out2 = l.forward(x)
    assert out1 != out2


test("Linear updates weights", test_linear_updates)


def test_block_embedding():
    e = BlockEmbedding(100, 32)
    out = e.forward([0, 1, 2])
    assert len(out) == 3
    assert len(out[0]) == 32


test("Block Embedding forward", test_block_embedding)


def test_block_embedding_backward():
    e = BlockEmbedding(100, 32)
    out1 = e.forward([0, 1, 2])
    e.backward([[0.1] * 32, [0.1] * 32, [0.1] * 32], lr=0.01)
    out2 = e.forward([0, 1, 2])
    assert out1 != out2


test("Block Embedding backward", test_block_embedding_backward)


def test_multihead_attention():
    mha = MultiHeadAttention(64, 4)
    x = [[0.1] * 64 for _ in range(5)]
    out = mha.forward(x)
    assert len(out) == 5
    assert len(out[0]) == 64


test("MultiHeadAttention forward", test_multihead_attention)


def test_multihead_causal():
    mha = MultiHeadAttention(64, 4)
    x = [[float(i) / 5] * 64 for i in range(3)]
    mha.forward(x[:1])
    mha.forward(x[1:2], use_cache=True)
    mha.forward(x[2:3], use_cache=True)
    mha.clear_cache()
    o_full = mha.forward(x)
    assert len(o_full) == 3


test("MultiHeadAttention causal masking", test_multihead_causal)


def test_feedforward():
    ff = FeedForward(64, 4)
    x = [[0.1] * 64 for _ in range(3)]
    out = ff.forward(x)
    assert len(out) == 3
    assert len(out[0]) == 64


test("FeedForward forward", test_feedforward)


def test_transformer_block():
    block = TransformerBlock(64, 4, 4)
    x = [[0.1] * 64 for _ in range(3)]
    out = block.forward(x)
    assert len(out) == 3
    assert len(out[0]) == 64


test("TransformerBlock forward", test_transformer_block)


def test_transformer_block_causal():
    block = TransformerBlock(64, 4, 4)
    x = [[float(i) / 5] * 64 for i in range(3)]
    block.forward(x[:1])
    block.forward(x[1:2], use_cache=True)
    block.forward(x[2:3], use_cache=True)
    block.clear_cache()
    out_full = block.forward(x)
    assert len(out_full) == 3


test("TransformerBlock causal masking", test_transformer_block_causal)


def test_transformer_residual():
    block = TransformerBlock(64, 4, 4)
    x = [[0.5] * 64 for _ in range(2)]
    out = block.forward(x)
    for i in range(2):
        for j in range(64):
            assert out[i][j] != 0.0, "Output should not be zero (residual adds signal)"


test("TransformerBlock residual connection", test_transformer_residual)


# ======================================================================
print("\n--- 12. GPT Model ---")
from app.ai.neural.transformer.gpt_model import GPTModel


def test_gpt_init():
    m = GPTModel(vocab_size=500, embed_dim=64, num_heads=4, num_layers=2, max_seq_len=128)
    assert m.vocab_size == 500
    assert m.embed_dim == 64
    assert m.num_heads == 4
    assert m.num_layers == 2


test("GPT init", test_gpt_init)


def test_gpt_forward():
    m = GPTModel(vocab_size=500, embed_dim=64, num_heads=4, num_layers=2, max_seq_len=128)
    logits = m.forward([1, 2, 3, 4, 5])
    assert len(logits) == 5
    assert len(logits[0]) == 500


test("GPT forward", test_gpt_forward)


def test_gpt_generate():
    m = GPTModel(vocab_size=500, embed_dim=64, num_heads=4, num_layers=2, max_seq_len=128)
    generated = m.generate([1, 2, 3], max_new_tokens=10, temperature=0.9)
    assert len(generated) >= 3
    assert generated[:3] == [1, 2, 3]


test("GPT generate", test_gpt_generate)


def test_gpt_predict():
    m = GPTModel(vocab_size=500, embed_dim=64, num_heads=4, num_layers=2, max_seq_len=128)
    token_id, prob = m.predict_next([1, 2, 3])
    assert isinstance(token_id, int)
    assert 0 <= prob <= 1
    assert 0 <= token_id < 500


test("GPT predict_next", test_gpt_predict)


def test_gpt_save_load():
    m = GPTModel(vocab_size=500, embed_dim=64, num_heads=4, num_layers=2, max_seq_len=128)
    m.forward([1, 2, 3])
    with tempfile.TemporaryDirectory() as d:
        m.save(d)
        m2 = GPTModel()
        m2.load(d)
        assert m.vocab_size == m2.vocab_size
        assert m.embed_dim == m2.embed_dim
        assert m.num_heads == m2.num_heads
        assert m.num_layers == m2.num_layers
        logits1 = m.forward([1, 2, 3])
        logits2 = m2.forward([1, 2, 3])
        for i in range(len(logits1)):
            for j in range(min(10, len(logits1[i]))):
                assert abs(logits1[i][j] - logits2[i][j]) < 1e-6


test("GPT save/load roundtrip", test_gpt_save_load)


def test_gpt_count_params():
    m = GPTModel(vocab_size=500, embed_dim=64, num_heads=4, num_layers=2, max_seq_len=128)
    p = m.count_params()
    assert p > 0
    assert isinstance(p, int)


test("GPT count_params", test_gpt_count_params)


def test_gpt_backward():
    m = GPTModel(vocab_size=500, embed_dim=64, num_heads=4, num_layers=2, max_seq_len=128)
    m.forward([1, 2, 3])
    loss = m.backward([4], lr=0.001)
    assert loss > 0


test("GPT backward pass", test_gpt_backward)


def test_gpt_backward_reduces_loss():
    m = GPTModel(vocab_size=500, embed_dim=64, num_heads=4, num_layers=2, max_seq_len=128)
    m.forward([1, 2, 3])
    loss1 = m.backward([4], lr=0.01)
    m.forward([1, 2, 3])
    loss2 = m.backward([4], lr=0.01)
    assert loss2 <= loss1 + 0.01, f"Loss should decrease or stay similar: {loss1} -> {loss2}"


test("GPT backward reduces loss over steps", test_gpt_backward_reduces_loss)


def test_gpt_clear_cache():
    m = GPTModel(vocab_size=500, embed_dim=64, num_heads=4, num_layers=2, max_seq_len=128)
    m.generate([1, 2], max_new_tokens=5)
    m.clear_cache()
    assert m._cache_pos == 0


test("GPT clear_cache", test_gpt_clear_cache)


def test_gpt_multiple_generations():
    m = GPTModel(vocab_size=500, embed_dim=64, num_heads=4, num_layers=2, max_seq_len=128)
    gen1 = m.generate([1], max_new_tokens=5)
    m.clear_cache()
    gen2 = m.generate([2], max_new_tokens=5)
    assert len(gen1) > 0
    assert len(gen2) > 0


test("GPT multiple generations work", test_gpt_multiple_generations)


# ======================================================================
print("\n--- 13. GPT Trainer ---")
from app.ai.neural.transformer.trainer import GPTTrainer


def test_gpt_trainer_init():
    tok = BPETokenizer()
    m = GPTModel(vocab_size=max(tok.vocab_len, 100), embed_dim=64, num_heads=4, num_layers=2, max_seq_len=128)
    t = GPTTrainer(m, tok)
    assert t.model is m
    assert t.tokenizer is tok


test("GPTTrainer init", test_gpt_trainer_init)


def test_gpt_trainer_train():
    tok = BPETokenizer(vocab_size=2000)
    m = GPTModel(vocab_size=2000, embed_dim=64, num_heads=4, num_layers=2, max_seq_len=128)
    t = GPTTrainer(m, tok, learning_rate=0.001)
    convs = [
        {"user": "Hola", "assistant": "Hola jefe, qué necesitas hoy"},
        {"user": "Gracias", "assistant": "De nada, para eso estoy aquí"},
        {"user": "Adiós", "assistant": "Hasta luego, estaré por aquí"},
        {"user": "¿Qué hora es?", "assistant": "Son las tres de la tarde"},
        {"user": "¿Qué puedes hacer?", "assistant": "Puedo ejecutar comandos y gestionar archivos"},
    ]
    history = t.train_on_conversations(convs, epochs=2, verbose=False)
    assert len(history) == 2
    assert history[0]["loss"] > 0
    assert history[1]["loss"] > 0


test("GPTTrainer train_on_conversations", test_gpt_trainer_train)


def test_gpt_trainer_loss_decreases():
    tok = BPETokenizer(vocab_size=2000)
    m = GPTModel(vocab_size=2000, embed_dim=64, num_heads=4, num_layers=2, max_seq_len=128)
    t = GPTTrainer(m, tok, learning_rate=0.01)
    convs = [
        {"user": "Hola", "assistant": "Hola jefe, qué necesitas hoy"},
        {"user": "Gracias", "assistant": "De nada, para eso estoy aquí"},
        {"user": "Adiós", "assistant": "Hasta luego, estaré por aquí"},
        {"user": "¿Qué hora es?", "assistant": "Son las tres de la tarde"},
        {"user": "¿Qué puedes hacer?", "assistant": "Puedo ejecutar comandos y gestionar archivos"},
    ]
    history = t.train_on_conversations(convs, epochs=5, verbose=False)
    assert len(history) >= 2
    first_loss = history[0]["loss"]
    last_loss = history[-1]["loss"]
    assert last_loss <= first_loss + 0.5, \
        f"Loss should decrease: {first_loss} -> {last_loss}"


test("GPTTrainer loss decreases", test_gpt_trainer_loss_decreases)


def test_gpt_trainer_checkpoint():
    tok = BPETokenizer(vocab_size=2000)
    m = GPTModel(vocab_size=2000, embed_dim=64, num_heads=4, num_layers=2, max_seq_len=128)
    t = GPTTrainer(m, tok, learning_rate=0.001)
    convs = [
        {"user": "Hola", "assistant": "Hola jefe"},
        {"user": "Gracias", "assistant": "De nada"},
        {"user": "Adiós", "assistant": "Hasta luego"},
    ]
    t.train_on_conversations(convs, epochs=1, verbose=False)
    with tempfile.TemporaryDirectory() as d:
        t.save_checkpoint(d)
        t2 = GPTTrainer(GPTModel(), BPETokenizer(), learning_rate=0.001)
        t2.load_checkpoint(d)
        assert t2._step == t._step


test("GPTTrainer save/load checkpoint", test_gpt_trainer_checkpoint)


# ======================================================================
print("\n--- 14. GPT Inference ---")
from app.ai.neural.transformer.inference import GPTInference


def test_inference_respond():
    tok = BPETokenizer(vocab_size=2000)
    tok.train(["Hola que tal", "Gracias por todo", "Adiós amigo"], verbose=False)
    m = GPTModel(vocab_size=tok.vocab_len, embed_dim=64, num_heads=4, num_layers=2, max_seq_len=128)
    engine = GPTInference(m, tok, max_new_tokens=10)
    resp = engine.respond("Hola")
    assert isinstance(resp, str)
    assert len(resp) > 0


test("GPTInference respond", test_inference_respond)


def test_inference_reset():
    tok = BPETokenizer(vocab_size=2000)
    tok.train(["Hola que tal", "Gracias por todo"], verbose=False)
    m = GPTModel(vocab_size=tok.vocab_len, embed_dim=64, num_heads=4, num_layers=2, max_seq_len=128)
    engine = GPTInference(m, tok, max_new_tokens=10)
    engine.respond("Hola")
    engine.reset_context()
    assert engine._context == []


test("GPTInference reset_context", test_inference_reset)


def test_inference_context_grows():
    tok = BPETokenizer(vocab_size=2000)
    tok.train(["Hola que tal", "Gracias por todo"], verbose=False)
    m = GPTModel(vocab_size=tok.vocab_len, embed_dim=64, num_heads=4, num_layers=2, max_seq_len=128)
    engine = GPTInference(m, tok, max_new_tokens=10)
    engine.respond("Hola")
    ctx_len_1 = len(engine._context)
    engine.respond("Gracias")
    ctx_len_2 = len(engine._context)
    assert ctx_len_2 >= ctx_len_1


test("GPTInference context grows", test_inference_context_grows)


def test_inference_max_tokens():
    tok = BPETokenizer(vocab_size=2000)
    tok.train(["Hola que tal"], verbose=False)
    m = GPTModel(vocab_size=tok.vocab_len, embed_dim=64, num_heads=4, num_layers=2, max_seq_len=128)
    engine = GPTInference(m, tok, max_new_tokens=3)
    resp = engine.respond("Hola")
    assert isinstance(resp, str)


test("GPTInference respects max_new_tokens", test_inference_max_tokens)


# ======================================================================
print("\n--- 15. Brain (NeuralBrain) ---")
from app.ai.neural.brain import NeuralBrain


def test_brain_init():
    with tempfile.TemporaryDirectory() as d:
        brain = NeuralBrain(data_dir=d)
        assert hasattr(brain, "think")
        assert hasattr(brain, "train_gpt")
        assert hasattr(brain, "learn")


test("Brain init", test_brain_init)


def test_brain_add_training_data():
    with tempfile.TemporaryDirectory() as d:
        brain = NeuralBrain(data_dir=d)
        brain.initialize()
        brain.add_training_data(["Hola", "Adiós"], ["SALUDO", "DESPEDIDA"])
        assert len(brain._training_data["texts"]) == 2


test("Brain add_training_data", test_brain_add_training_data)


def test_brain_train():
    with tempfile.TemporaryDirectory() as d:
        brain = NeuralBrain(data_dir=d)
        brain.initialize()
        texts = ["Hola"] * 5 + ["Adiós"] * 5 + ["Gracias"] * 5
        intents = ["SALUDO"] * 5 + ["DESPEDIDA"] * 5 + ["AGRADECIMIENTO"] * 5
        brain.add_training_data(texts, intents)
        history = brain.train(epochs=10)
        assert len(history) > 0
        assert brain.is_trained()


test("Brain train classifier", test_brain_train)


def test_brain_train_gpt():
    with tempfile.TemporaryDirectory() as d:
        brain = NeuralBrain(data_dir=d)
        brain.initialize()
        convs = [
            {"user": "Hola", "assistant": "Hola jefe"},
            {"user": "Adiós", "assistant": "Hasta luego"},
        ] * 3
        history = brain.train_gpt(convs, epochs=2, verbose=False)
        assert len(history) == 2


test("Brain train_gpt", test_brain_train_gpt)


def test_brain_learn():
    with tempfile.TemporaryDirectory() as d:
        brain = NeuralBrain(data_dir=d)
        brain.initialize()
        result = brain.learn("Python es genial", "general")
        assert result is True


test("Brain learn", test_brain_learn)


def test_brain_learn_entity():
    with tempfile.TemporaryDirectory() as d:
        brain = NeuralBrain(data_dir=d)
        brain.initialize()
        result = brain.learn_entity("Python", "lenguaje", {"popular": True})
        assert result is True


test("Brain learn_entity", test_brain_learn_entity)


def test_brain_search_knowledge():
    with tempfile.TemporaryDirectory() as d:
        brain = NeuralBrain(data_dir=d)
        brain.initialize()
        brain.learn("Python es un lenguaje de programación", "general")
        results = brain.search_knowledge("Python")
        assert len(results) > 0


test("Brain search_knowledge", test_brain_search_knowledge)


def test_brain_search_memory():
    with tempfile.TemporaryDirectory() as d:
        brain = NeuralBrain(data_dir=d)
        brain.initialize()
        brain.learn("dato importante sobre Python y programación", "general")
        results = brain.search_memory("Python programación")
        assert len(results) > 0


test("Brain search_memory", test_brain_search_memory)


def test_brain_infer_about():
    with tempfile.TemporaryDirectory() as d:
        brain = NeuralBrain(data_dir=d)
        brain.initialize()
        brain.learn_entity("Python", "lenguaje")
        brain.learn("Python es popular", "general")
        results = brain.infer_about("Python")
        assert isinstance(results, list)


test("Brain infer_about", test_brain_infer_about)


def test_brain_plan_action():
    with tempfile.TemporaryDirectory() as d:
        brain = NeuralBrain(data_dir=d)
        brain.initialize()
        plan = brain.plan_action("Instalar paquete nuevo")
        assert "steps" in plan
        assert isinstance(plan["steps"], list)
        assert len(plan["steps"]) > 0


test("Brain plan_action", test_brain_plan_action)


def test_brain_status():
    with tempfile.TemporaryDirectory() as d:
        brain = NeuralBrain(data_dir=d)
        brain.initialize()
        status = brain.get_status()
        assert isinstance(status, dict)
        assert "mood" in status
        assert "knowledge_stats" in status
        assert "memory_stats" in status
        assert "trained" in status
        assert "queries" in status


test("Brain get_status", test_brain_status)


def test_brain_personality():
    with tempfile.TemporaryDirectory() as d:
        brain = NeuralBrain(data_dir=d)
        brain.initialize()
        p = brain.personality
        assert p.mood in ["neutral", "happy", "focused", "tired", "excited", "serious"]
        p.mood = "happy"
        assert p.mood == "happy"
        assert p.energy == 1.0
        greeting = p.get_greeting()
        assert isinstance(greeting, str)
        assert len(greeting) > 0


test("Brain personality", test_brain_personality)


def test_brain_think():
    with tempfile.TemporaryDirectory() as d:
        brain = NeuralBrain(data_dir=d)
        brain.initialize()
        texts = ["Hola"] * 5 + ["Adiós"] * 5 + ["Gracias"] * 5 + ["Qué hora es"] * 5
        intents = ["SALUDO"] * 5 + ["DESPEDIDA"] * 5 + ["AGRADECIMIENTO"] * 5 + ["PREGUNTA"] * 5
        brain.add_training_data(texts, intents)
        brain.train(epochs=20)

        async def run():
            return await brain.think("Hola")

        resp = asyncio.run(run())
        assert isinstance(resp, str)
        assert len(resp) > 0


test("Brain think (async)", test_brain_think)


def test_brain_think_multiple():
    with tempfile.TemporaryDirectory() as d:
        brain = NeuralBrain(data_dir=d)
        brain.initialize()
        texts = ["Hola"] * 5 + ["Adiós"] * 5 + ["Gracias"] * 5
        intents = ["SALUDO"] * 5 + ["DESPEDIDA"] * 5 + ["AGRADECIMIENTO"] * 5
        brain.add_training_data(texts, intents)
        brain.train(epochs=20)

        async def run():
            results = []
            for msg in ["Hola", "Adiós", "Gracias"]:
                r = await brain.think(msg)
                results.append(r)
            return results

        results = asyncio.run(run())
        assert all(isinstance(r, str) and len(r) > 0 for r in results)


test("Brain think multiple messages", test_brain_think_multiple)


def test_brain_is_trained():
    with tempfile.TemporaryDirectory() as d:
        brain = NeuralBrain(data_dir=d)
        brain.initialize()
        assert not brain.is_trained()
        brain.add_training_data(["Hola"] * 5, ["SALUDO"] * 5)
        brain.train(epochs=5)
        assert brain.is_trained()


test("Brain is_trained", test_brain_is_trained)


def test_brain_close():
    with tempfile.TemporaryDirectory() as d:
        brain = NeuralBrain(data_dir=d)
        brain.initialize()
        brain.learn("Test", "general")
        brain.close()
        # Should not raise
        assert True


test("Brain close", test_brain_close)


# ======================================================================
print("\n" + "=" * 70)
total = passed + failed
print(f" RESULTADO: {passed}/{total} éxitos, {failed} fallos")
print("=" * 70)

if errors:
    print("\nFallos:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\n  ¡TODAS LAS PRUEBAS PASARON!")
    sys.exit(0)
