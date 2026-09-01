# MAÑANA — arranque rápido (Sesión 3)

Léeme primero y arranca sin perder tiempo. (Detalles en `SESION_2026-09-01.md`)

## Estado
- ARIA al 100% en PC (283/283 tests) y ESP32.
- Entrenador **vectorizado y commiteado** (~18x más rápido, ya no es cuello de
  botella).
- Fix de generación commiteado (el GPT ya no colapsa a `<unk>`).
- Corpus web **cargado y CURADO** (9 conv/9 textos limpios), modelo actual
  reentrenado con él.
- Pendiente: mejorar la CALIDAD de generación libre (sigue débil/repetitiva).

## Prioridades de la sesión (en orden)
1. **Tunear params de generación** sin reentrenar (más rápido de probar):
   inventariar combos de `temperature` / `top_k` / `top_p` /
   `repetition_penalty` contra el modelo actual y medir % de salidas útiles y
   no degeneradas.
2. **Entrenar bien el modelo escalado (embed 128, 4 layers, 8 heads, 1.35M
   params)**: usar LR más baja (~1e-4) con decay y más épocas, sobre el corpus
   limpio. En la prueba previa con LR 3e-4 la loss saltaba (overfit).
3. **Comparar generación** actual vs escalado con el MIMO set de prompts;
   decidir si promover el modelo grande como cerebro principal (cambiar la
   config por defecto en `app/ai/neural/brain.py`).
4. **Crecer el corpus con datos limpios**: `python scripts/learn_from_web.py
   --tema "<tema>" --articulos N` con temas relevantes; reentrenar.

## Comandos útiles
```bash
source .venv/bin/activate
python scripts/learn_from_web.py --tema "programación" --articulos 5
python scripts/retrain_full.py      # reentrenar todo con corpus (limpio)
python scripts/talk_to_aria.py      # hablar con ARIA local
pytest                              # 283 tests
```

## Recordatorio importante
- NO tocar `brain.py`, `train_neural.py`, `talk_to_aria.py` y demás archivos
  que venían modificados del usuario SIN preguntar.
- Guardar progreso al final creando `.progreso/SESION_YYYY-MM-DD.md` y
  actualizando `.progreso/ESTADO.md`.
