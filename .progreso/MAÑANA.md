# MAÑANA — arranque rápido (Sesión 6)

Léeme primero y arranca sin perder tiempo. (Detalles en `SESION_2026-09-12.md`)

## Estado
- Trainer por lotes **~10.3x** (B=32), equivalencia VERDE (e-15/e-16), pytest 285 passed.
- Corpus escalado a **3.11 MB / 3342 textos** (`data/neural/corpus_masivo.json`).
- `scaled_model_v3` reentrenándose por chunks reanudables; **pasada 1 completada**,
  loss en atractor ~7.69. Falta dar muchas pasadas para ver descenso.

## La misión de hoy
1. **Seguir el reentrenado**: reset `done`/`step`, lanzar tramos de ~6.5 min hasta
   done=5749, repetir. Meta: loss < 7.0 antes de valorar generación.
2. **Validar cada pocas pasadas**: `gen_v3.py` (`--prompt`/`--temp`/`--seed`).
3. Si la loss no baja de ~7.3 tras 5-10 pasadas: probar una pasada a lr 1e-3 estable.
4. Decidir promoción en ESTADO (sin `<unk>`, sin bucles bigráficos).

## Comandos útiles (sesión entera en un comando — ver receta exacta en SESION_2026-09-12.md)
```bash
source .venv/bin/activate
python /tmp/equivalence_test.py                     # contrato por lotes (~e-15)
pytest                                             # 285 tests
python data/neural/harvest_masivo.py               # continuar corpus (reanudable)
python data/neural/retrain_v3.py --max-min 6.5 --chunk 400 --batch 32 --lr 2e-4
python data/neural/gen_v3.py --prompt "..." --temp 0.8 --seed 7
```

## Ojo (¡importante!)
- El repo canónico es `/home/samuel/A.R.I.A-main` (PUNTO; git + .venv). El tool
  a veces escribe en `/home/samuel/A.R.I-A-main` (GUION). Tras escribir un script
  con Write: `cp <GUION-arch> <PUNTO-arch>`. Los entrenamientos solo caen en el PUNTO.
- NO tocar `brain.py`, `train_neural.py`, `talk_to_aria.py` sin preguntar.
- `data/*` gitignored; `trainer_batch.py`, `trainer.py` y `.progreso` commiteados.
- Guardar progreso al final: `SESION_YYYY-MM-DD.md` + `ESTADO.md`.