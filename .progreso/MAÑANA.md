# MAÑANA — arranque rápido (Sesión 6)

Léeme primero y arranca sin perder tiempo. (Detalles en `SESION_2026-09-12.md`)

## Estado
- Trainer por lotes **~10.3x** (B=32), equivalencia VERDE (e-15/e-16), pytest 285 passed.
- Corpus escalado a **3.11 MB / 3342 textos** (`data/neural/corpus_masivo.json`).
- `scaled_model_v3` reentrenándose por chunks reanudables; **pasada 1 completada**,
  loss en atractor ~7.69. Falta dar muchas pasadas para ver descenso.

## La misión de hoy
1. **Decidir modelo a seguir**: el BPE word-aware (4096, `new_bpe_tokenizer`)
   gana de calle (0 unk, pérdida 5.59 tras 3 pasadas vs 6.88 en 12 del OLD),
   pero la generación aún no monta palabras. Elegir entre:
   - dar más pasadas a `scaled_v3_bpe4096` (lr 1e-2) hasta que las palabras
     superen a los espacios en el muestreo, o
   - subir capacidad (más capas/embed) con el BPE nuevo.
2. **Validar cada 2-3 pasadas**: `gen_v3.py` adaptado o inline probe (top-tokens
   tras "la ciudad de " → hoy p~0.001 por palabra).
3. Seguir con pasadas del OLD (6.88) es opcional — el BPE nuevo lo supera ya.
4. Decidir promoción en ESTADO.

## Comandos útiles (sesión entera en un comando — ver receta exacta en SESION_2026-09-12.md)
```bash
source .venv/bin/activate
python /tmp/equivalence_test.py                     # contrato por lotes (~e-15)
pytest                                             # 285 tests
python data/neural/rebuild_bpe_v3.py --vocab 4096  # re-entrenar el BPE word-aware
python data/neural/retrain_bpe.py --tok data/neural/new_bpe_tokenizer \
    --windows data/neural/v3_bpe4096_windows.json \
    --out data/neural/scaled_v3_bpe4096 --vocab 4096 --max-min 95 --lr 1e-2
```

## Ojo (¡importante!)
- El repo canónico es `/home/samuel/A.R.I.A-main` (PUNTO; git + .venv). El tool
  a veces escribe en `/home/samuel/A.R.I-A-main` (GUION). Tras escribir un script
  con Write: `cp <GUION-arch> <PUNTO-arch>`. Los entrenamientos solo caen en el PUNTO.
- NO tocar `brain.py`, `train_neural.py`, `talk_to_aria.py` sin preguntar.
- `data/*` gitignored; `trainer_batch.py`, `trainer.py` y `.progreso` commiteados.
- Guardar progreso al final: `SESION_YYYY-MM-DD.md` + `ESTADO.md`.