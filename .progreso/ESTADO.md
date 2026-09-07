# ESTADO ACTUAL DE A.R.I.A (canónica)

Actualizado: 2026-09-07 (Sesión 3)
Ver histórico detallado en `SESION_2026-09-07.md` y sesiones anteriores.

## Resumen ejecutivo
ARIA funciona **al 100% en el PC** (284/284 tests) y en el ESP32. El entrenador
del GPT local quedó vectorizado (~18x), la generación ya no colapsa a `<unk>`,
y se protegió a ARIA contra los bucles degenerados del GPT local (guardián
reforzado). El **modelo escalado NO se promociona** (basado en datos: sigue
degenerando; el corpus es demasiado pequeño).

## Qué funciona
- **PC completo**: cerebro neural GPT local (377,472 params, desplegado) + voz
  + tools + memoria + GUI web. Sin API keys para el cerebro local.
- **ESP32**: nodo de IA local TensorFlow Lite Micro (inferencia offline).
- **Aprendizaje del internet**: `learn_from_web.py` (filtra extractos cortos y
  duplicados sintéticos) + `retrain_full.py` fusiona el corpus al reentrenar.
- **Clasificador de intenciones**: fiable (reglas + red, confianza ≥0.9).
- **Guardián de respuestas**: detecta bucles degenerados de bigramas → ARIA
  usa el fallback elegante en vez de hablar basura.

## Datos / modelos actuales
- Modelo GPT desplegado: vocab 2048, embed 64, 4 heads, 2 layers, seq 256 =
  377,472 params. Reentrenado con corpus limpio.
- Corpus aprendido (limpio): `data/neural/corpus.json` = **59 conversaciones +
  20 textos** (artículos completos y temas relevantes).
- Base en código: 118 conv + 48 textos (train_neural) + 71 conv/27 textos
  (talk_to_aria). Total reentrenado ≈ 210 conv + 71 textos.

## CUADRO DE MANDO — pendientes priorizados
1. **[ALTA] Decidir futuro del GPT local**: ¿invertir en corpus MUCHO mayor
   (miles de diálogos) o tratarlo como experimental y enfocar en lo fiable
   (clasificador + memoria)? Datos de hoy: escalar capacidad con ~200
   conversaciones no basta (overfit/bucles).
2. **[MEDIA] Revisar tokenizer BPE**: merges que fragmentan palabras de forma
   rara ("qprogram", "Armin") que alimentan la degeneración.
3. **[BAJA] Limpiar experimentos**: `data/scaled_model*` no están desplegados
   (se pueden borrar).
4. **[MEDIA] Correr pytest tras cada cambio** (hoy 284/284 OK).

## Notas de manejo
- `data/*` está gitignored (solo `data/.gitkeep` se versiona). Modelos y corpus
  son datos locales.
- Los archivos `brain.py`, `train_neural.py`, `talk_to_aria.py` traían cambios
  SIN commitear del usuario: NO tocarlos sin preguntar.
- Interfaz de chat local: `python scripts/talk_to_aria.py`.
- Monitor ESP32: `pio device monitor -b 115200` (requiere terminal propia).