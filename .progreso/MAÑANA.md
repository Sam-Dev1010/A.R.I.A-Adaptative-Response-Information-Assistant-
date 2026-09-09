# MAÑANA — arranque rápido (Sesión 5)

Léeme primero y arranca sin perder tiempo. (Detalles en `SESION_2026-09-09.md`)

## Estado
- Entrenador **por lotes vectorizado ~10.3x** (B=32) con equivalencia bit-exacta
  (B=1) y batch-mean (B>1) verificada; pytest 285 passed.
- Toca la **fase de datos**: el freno del GPT local es corpus, no capacidad.

## La misión de mañana
1. **Escalar corpus**: correr `/tmp/harvest_masivo.py` (throttling + reanudable)
   hasta juntar varios MB de texto en español (Wikipedia/knowledge base).
2. **Reentrenar `scaled_model_v3`** con el trainer por lotes:
   `/tmp/train_scaled_v3.py` (checkpoints en `/tmp/train_v3_state.json`).
   En la sesión anterior tardaba un rato; con el nuevo trainer ya es viable.
3. **Validar generación**: `python /tmp/gen_v3.py` (o talk_to_aria tras
   promocionar) — sin colapso a `<unk>` ni bucles. Decidir promoción en ESTADO.
4. Si el BPE aún fragmenta raro, revisar sus merges.

## Comandos útiles
```bash
source .venv/bin/activate
python /tmp/equivalence_test.py           # contrato por lotes (debe salir ~e-15)
pytest                                   # 285 tests
python /tmp/harvest_masivo.py            # continuar descarga (reanudable)
python /tmp/train_scaled_v3.py           # reentrenar v3 con lotes (B=32)
python scripts/talk_to_aria.py           # hablar con ARIA local
```

## Recordatorio importante
- NO tocar `brain.py`, `train_neural.py`, `talk_to_aria.py` sin preguntar.
- `data/*` gitignored; `trainer_batch.py` ya está commiteado.
- Guardar progreso al final: `SESION_YYYY-MM-DD.md` + `ESTADO.md`.