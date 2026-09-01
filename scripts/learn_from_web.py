#!/usr/bin/env python3
"""Aprendizaje autónomo de A.R.I.A desde internet: recolecta texto real y lo
convierte en corpus de entrenamiento (conversaciones + textos de conocimiento).

Metas de fase 2 de A.R.I.A como "su propio LLM":
  1. ARIA baja texto real del internet (Wikipedia en español por defecto).
  2. Convierte cada artículo en pares de preguntas/respuestas naturales.
  3. Acumula esos pares en data/neural/corpus.json (el corpus creciente).
  4. El entrenamiento (retrain_full.py / train_neural.py) se alimenta del corpus.

Uso:
    python scripts/learn_from_web.py --articulos 20            # 20 artículos aleatorios
    python scripts/learn_from_web.py --tema "inteligencia artificial" # busca un tema
    python scripts/learn_from_web.py --articulos 5 --solo-conversaciones

Opciones:
    --articulos N     cuántos artículos bajar (default 10)
    --tema T          frase para buscar artículos del tema (default: aleatorio)
    --max-conv N      máx. pares de conversación extraídos por artículo (default 3)
    --solo-texto      solo acumular textos de conocimiento, sin pares Q&A
    --verboso         más detalle en la salida

El corpus se guarda en data/neural/corpus.json (ignorado por git, como el resto
de data/). Puedes borrarlo y empezar de cero si quieres re-entrenar el tokenizer.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path
from urllib import request
from urllib.error import URLError, HTTPError

ROOT = Path(__file__).resolve().parent.parent
CORPUS_FILE = ROOT / "data" / "neural" / "corpus.json"

USER_AGENT = "A.R.I.A-learning/0.2 (asistente local; contacto: samuel@localhost)"
WIKI_BASE = "https://es.wikipedia.org/api/rest_v1/page"

# Frases para generar preguntas a partir del extracto de un artículo.
# Se rellenan con el tema/título y el contexto del extracto.
QUESTION_PATTERNS = [
    "¿Qué es {tema}?",
    "Explícame {tema} en tus palabras",
    "¿Qué sabes de {tema}?",
    "Cuéntame sobre {tema}",
    "¿Puedes darme información sobre {tema}?",
    "¿Qué me puedes contar de {tema}?",
]


def _fetch_json(url: str, timeout: int = 20) -> dict | None:
    """Baja y parsea un JSON con un User-Agent respetuoso."""
    req = request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (URLError, HTTPError, ValueError) as exc:  # noqa: BLE001
        print(f"  [error] no se pudo bajar {url}: {exc}", file=sys.stderr)
        return None


def _fetch_extract(title: str) -> str | None:
    """Baja el resumen (extract) de un artículo por su título."""
    url = f"{WIKI_BASE}/summary/{title.replace(' ', '_')}"
    data = _fetch_json(url)
    if not data:
        return None
    extract = data.get("extract") or data.get("description") or None
    return extract.strip() if extract else None


def _fetch_random_extract() -> dict[str, str] | None:
    """Baja un artículo aleatorio y devuelve {tema, texto}."""
    data = _fetch_json(f"{WIKI_BASE}/random/summary")
    if not data:
        return None
    title = data.get("title") or ""
    extract = data.get("extract") or ""
    if not title or not extract:
        return None
    return {"tema": title, "texto": extract.strip()}


def fetch_by_topic(topic: str) -> dict[str, str] | None:
    """Busca un artículo por tema usando el extracto de la API REST."""
    url = f"{WIKI_BASE}/summary/{topic.replace(' ', '_')}"
    data = _fetch_json(url)
    if not data:
        return None
    title = data.get("title") or topic
    extract = data.get("extract") or ""
    if not extract:
        return None
    return {"tema": title, "texto": extract.strip()}


# Longitud mínima (caracteres) del extracto para considerarlo aprendizaje útil.
MIN_EXTRACT_LEN = 200


def _split_sentences(text: str) -> list[str]:
    """Divide texto en oraciones completas (terminadas en . ! ?)."""
    import re

    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if len(p.split()) >= 6 and p[-1:] in ".!?"]


def build_pairs(article: dict[str, str], max_pairs: int = 3) -> list[dict[str, str]]:
    """Convierte un artículo en pares pregunta→respuesta naturales y variados.

    Solo se usa un extracto suficientemente largo (evita ruido de artículos
    diminutos). Genera una única pregunta general con la respuesta completa y,
    si el extracto tiene oraciones completas, hasta `max_pairs` preguntas de
    detalle. No se generan variantes casi-duplicadas de la misma pregunta.
    """
    tema = article.get("tema", "").strip()
    texto = article.get("texto", "").strip()
    if not texto or len(texto) < MIN_EXTRACT_LEN:
        return []

    pairs: list[dict[str, str]] = []

    # 1. Una sola pregunta general con el extracto completo (no variantes duplicadas).
    pairs.append({"user": f"¿Qué es {tema}?", "assistant": texto})
    pairs.append({"user": "Cuéntame sobre {tema}".format(tema=tema), "assistant": texto})

    # 2. Preguntas de detalle solo con oraciones completas.
    sentences = _split_sentences(texto)
    if len(sentences) >= 2:
        for sent in sentences:
            if len(pairs) >= max_pairs + 2:
                break
            pairs.append({"user": f"Explícame más sobre {tema}", "assistant": sent})

    return pairs


def load_corpus() -> dict[str, list]:
    """Carga el corpus creciente, o uno vacío si no existe."""
    if CORPUS_FILE.exists():
        try:
            return json.loads(CORPUS_FILE.read_text())
        except (json.JSONDecodeError, OSError):  # noqa: BLE001
            pass
    return {"conversations": [], "extra_texts": []}


def save_corpus(corpus: dict[str, list]) -> None:
    CORPUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_FILE.write_text(json.dumps(corpus, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articulos", type=int, default=10)
    parser.add_argument("--tema", type=str, default=None)
    parser.add_argument("--max-conv", type=int, default=3)
    parser.add_argument("--solo-texto", action="store_true")
    parser.add_argument("--verboso", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print(" A.R.I.A - Aprendiendo del internet (corpus de entrenamiento)")
    print("=" * 60)

    corpus = load_corpus()
    n_inicial_conv = len(corpus["conversations"])
    n_inicial_texto = len(corpus["extra_texts"])
    print(f"  Corpus actual: {n_inicial_conv} conversaciones, {n_inicial_texto} textos\n")

    nuevos_conv = 0
    nuevos_texto = 0

    for i in range(args.articulos):
        if args.tema and i == 0:
            # Refresca el tema en la primera iteración y luego sigue aleatorio;
            # si falla la búsqueda del tema, cae a aleatorio.
            article = fetch_by_topic(args.tema)
        else:
            article = _fetch_random_extract()
        if not article:
            print(f"[{i+1}/{args.articulos}] (sin contenido)")
            time.sleep(1)
            continue

        tema = article["tema"]
        texto = article["texto"]

        # Saltar extractos demasiado cortos (aportan más ruido que conocimiento).
        if len(texto) < MIN_EXTRACT_LEN:
            if args.verboso:
                print(f"  [{i+1}/{args.articulos}] '{tema}' (extracto corto, omitido)")
            time.sleep(1)
            continue

        # Acumular texto de conocimiento (siempre).
        if texto not in corpus["extra_texts"]:
            corpus["extra_texts"].append(texto)
            nuevos_texto += 1

        if not args.solo_texto:
            # Convertir a pares de conversación (deduplicado por (user,assistant)).
            for pair in build_pairs(article, max_pairs=args.max_conv):
                key = (pair["user"], pair["assistant"])
                if any((c.get("user"), c.get("assistant")) == key
                       for c in corpus["conversations"]):
                    continue
                corpus["conversations"].append(pair)
                nuevos_conv += 1

        if args.verboso:
            print(f"  [{i+1}/{args.articulos}] + '{tema}' ({len(texto)} chars)")
        time.sleep(1)  # cortesía al servidor de Wikipedia

    save_corpus(corpus)
    print(f"\n  Aprendidos: +{nuevos_conv} conversaciones, +{nuevos_texto} textos")
    print(f"  Total corpus: {len(corpus['conversations'])} conv, {len(corpus['extra_texts'])} textos")
    print(f"  Guardado en {CORPUS_FILE}")
    print("\nAhora re-entrena con:")
    print("  python scripts/retrain_full.py")


if __name__ == "__main__":
    main()
