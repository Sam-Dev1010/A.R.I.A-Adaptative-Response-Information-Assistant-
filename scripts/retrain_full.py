#!/usr/bin/env python3
"""Re-entrena el modelo GPT de ARIA con el corpus ampliado (combinado).

Escribe el modelo en los directorios de datos de talk_to_aria (data/aria_gpt)
y de train_neural (data/neural), así ambos flujos usan el cerebro mejorado.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_mod(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    print("=" * 60, flush=True)
    print(" A.R.I.A - Re-entrenamiento completo del GPT (corpus ampliado)", flush=True)
    print("=" * 60, flush=True)

    tn = load_mod("tn", str(ROOT / "scripts" / "train_neural.py"))
    ta = load_mod("ta", str(ROOT / "scripts" / "talk_to_aria.py"))

    # Combinar conversaciones (sin duplicados)
    convs = list(tn.CONVERSATIONS)
    seen = {(c["user"], c["assistant"]) for c in convs}
    for c in ta.TRAINING_CONVERSATIONS:
        key = (c["user"], c["assistant"])
        if key not in seen:
            seen.add(key)
            convs.append(c)

    # Combinar textos extra (sin duplicados)
    extra_texts = list(tn.EXTRA_TEXTS)
    seen_t = set(extra_texts)
    for t in ta.EXTRA_TEXTS:
        if t not in seen_t:
            seen_t.add(t)
            extra_texts.append(t)

    # === Corpus creciente aprendido de internet (data/neural/corpus.json) ===
    corpus_file = ROOT / "data" / "neural" / "corpus.json"
    if corpus_file.exists():
        try:
            corpus = json.loads(corpus_file.read_text())
            for p in corpus.get("conversations", []):
                user, asst = p.get("user", ""), p.get("assistant", "")
                if user and asst and (user, asst) not in seen:
                    seen.add((user, asst))
                    convs.append({"user": user, "assistant": asst})
            for t in corpus.get("extra_texts", []):
                if t and t not in seen_t:
                    seen_t.add(t)
                    extra_texts.append(t)
            print(f"Corpus aprendido de internet: "
                  f"{len(convs)} conv | {len(extra_texts)} textos", flush=True)
        except (json.JSONDecodeError, OSError) as exc:  # noqa: BLE001
            print(f"[aviso] no se pudo leer corpus.json: {exc}", flush=True)

    print(f"Conversaciones: {len(convs)} | Textos extra: {len(extra_texts)}", flush=True)

    from app.ai.neural.brain import NeuralBrain

    for data_dir_name in ("data/aria_gpt", "data/neural"):
        data_dir = ROOT / data_dir_name
        print(f"\n>>> Entrenando en {data_dir} ...", flush=True)
        brain = NeuralBrain(data_dir)
        brain.initialize()

        t0 = time.time()
        brain.train_gpt(
            conversations=convs,
            extra_texts=extra_texts,
            epochs=12,
            verbose=True,
        )
        elapsed = time.time() - t0
        elapsed = time.time() - t0
        print(f"<<< {data_dir} listo en {elapsed:.1f}s | params={brain.gpt_model.count_params():,}", flush=True)
        brain.close()

    print("\n¡Re-entrenamiento completo terminado con éxito!", flush=True)


if __name__ == "__main__":
    main()
