# MAÑANA — arranque rápido (Sesión 6)

Léeme primero y arranca sin perder tiempo. (Detalles en `SESION_2026-09-12.md`)

## Estado
- Trainer por lotes **~10.3x** (B=32), equivalencia VERDE (e-15/e-16), pytest 285 passed.
- Corpus escalado a **3.11 MB / 3342 textos** (`data/neural/corpus_masivo.json`).
- `scaled_model_v3` reentrenándose por chunks reanudables; **pasada 1 completada**,
  loss en atractor ~7.69. Falta dar muchas pasadas para ver descenso.

## La misión de hoy
1. **Seguir el reentrenado a lr 1e-2** (¡el que rompe el atractor!): reset
   `done`/`step`, una pasada completa (~70 min) hasta done=5200, repetir.
   Loss actual ~6.88 (11 pasadas). Meta: loss < 6.3.
2. **Validar cada 2-3 pasadas**: `gen_v3.py` (`--temp 0.8`; a temp baja hoy solo
   emite espacios → señal de que aún no hay palabras).
3. Si se estanca o se vuelve ruidoso: revisar codificación BPE (fragmenta palabras).
4. Decidir promoción en ESTADO (sin `<unk>`, sin bucles bigráficos).

## Comandos útiles (sesión entera en un comando — ver receta exacta en SESION_2026-09-12.md)
```bash
source .venv/bin/activate
python /tmp/equivalence_test.py                     # contrato por lotes (~e-15)
pytest                                             # 285 tests
python data/neural/harvest_masivo.py               # continuar corpus (reanudable)
python data/neural/retrain_v3.py --max-min 68 --chunk 400 --batch 32 --lr 1e-2
python data/neural/gen_v3.py --prompt "..." --temp 0.8 --seed 7
```

## Ojo (¡importante!)
- El repo canónico es `/home/samuel/A.R.I.A-main` (PUNTO; git + .venv). El tool
  a veces escribe en `/home/samuel/A.R.I-A-main` (GUION). Tras escribir un script
  con Write: `cp <GUION-arch> <PUNTO-arch>`. Los entrenamientos solo caen en el PUNTO.
- NO tocar `brain.py`, `train_neural.py`, `talk_to_aria.py` sin preguntar.
- `data/*` gitignored; `trainer_batch.py`, `trainer.py` y `.progreso` commiteados.
- Guardar progreso al final: `SESION_YYYY-MM-DD.md` + `ESTADO.md`.