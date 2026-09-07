# MAÑANA — arranque rápido (Sesión 4)

Léeme primero y arranca sin perder tiempo. (Detalles en `SESION_2026-09-07.md`)

## Estado
- ARIA al 100% en PC (284/284 tests) y ESP32.
- Entrenador vectorizado (~18x), generación sin colapso a `<unk>`, y **guardián
  anti-bucles** activo (ARIA ya no diría basura si el GPT local degenera).
- Veredicto con datos: **el modelo escalado (1.35M) NO se promociona** — sigue
  degenerando con el corpus actual (~210 conversaciones). El freno es DATA, no
  capacidad.

## La decisión estratégica pendiente (preguntar al usuario)
El GPT local condicionado NO produce diálogo útil con el corpus actual. Hay que
elegir camino:
1. **Invertir en corpus grande**: miles de diálogos/textos en español fuentes
   abiertas (ej: descargar más artículos web limpios, corpus abiertos). Coste:
   tiempo de descarga y de entrenamiento, pero es la vía honesta si queremos
   "su propio LLM".
2. **GPT local experimental**: mantenerlo como I+D con respaldo y enfocar la
   inversión en el clasificador (fiable) y la memoria/autoaprendizaje.
3. **Revisar tokenizer BPE**: los merges raros ("qprogram", "Armin") alimentan
   la degeneración; vale la pena inspeccionar el preprocesado.

## Comandos útiles
```bash
source .venv/bin/activate
python scripts/talk_to_aria.py      # hablar con ARIA local
python scripts/retrain_full.py      # reentrenar todo con corpus (limpio)
python scripts/learn_from_web.py --tema "X" --articulos N
pytest                              # 284 tests
```

## Recordatorio importante
- NO tocar `brain.py`, `train_neural.py`, `talk_to_aria.py` y demás archivos
  que venían modificados del usuario SIN preguntar.
- Guardar progreso al final creando `.progreso/SESION_YYYY-MM-DD.md` y
  actualizando `.progreso/ESTADO.md`.