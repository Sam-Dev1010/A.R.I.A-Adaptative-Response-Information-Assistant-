# ESTADO ACTUAL DE A.R.I.A (canónica)

Actualizado: 2026-09-01 (Sesión 2)
Ver histórico detallado en `SESION_2026-09-01.md` y `SESION_2026-08-29.md`.

## Resumen ejecutivo
ARIA funciona **al 100% en el PC** (283/283 tests) y en el ESP32. La Fase 2
(que sea su propio LLM que aprende del internet) avanzó: el entrenador quedó
vectorizado (**~18x más rápido**, resuelto el cuello de botella), se corrigió
un bug que hacía colapsar la generación a `<unk>`, y el corpus del internet se
creció y luego se **curó** (menos ruido y duplicados).

## Qué funciona
- **PC completo**: cerebro neural GPT local (377,472 params) + voz + tools +
  memoria + GUI web. Sin API keys para el cerebro local.
- **ESP32**: nodo de IA local con TensorFlow Lite Micro (inferencia offline).
- **Aprendizaje del internet**: `scripts/learn_from_web.py` (ahora filtra
  extractos cortos y evita preguntas duplicadas) + `retrain_full.py` fusiona el
  corpus al reentrenar.

## Datos / modelos actuales
- Modelo GPT desplegado: vocab 2048, embed 64, 4 heads, 2 layers, seq 256 =
  377,472 params. Reentrenado con corpus limpio: loss 5.757 / ppl 316.46.
- Corpus aprendido (curado): `data/neural/corpus.json` = 9 conversaciones + 9
  textos (artículos completos, sin duplicados sintéticos).
- Base de entrenamiento en código: 118 conv + 48 textos (hardcoded) +
  talk_to_aria (71 conv/27 textos). Total reentrenado: 160 conv + 60 textos.

## CUADRO DE MANDO — pendientes priorizados
1. **[ALTA] Tunear params de generación** (temperature/top_k/top_p/repetición)
   sobre el modelo actual para reducir la degeneración repetitiva.
2. **[ALTA] Entrenar bien el modelo escalado (128/4/8, 1.35M params)**: la
   prueba mostró que mejora la fluidez de generación, pero con loss inestable
   → probar LR más baja + decay y más épocas con el corpus limpio.
3. **[MEDIA] Decidir si promover el modelo escalado** como cerebro principal
   (cambiar config por defecto en `app/ai/neural/brain.py`).
4. **[MEDIA] Crecer el corpus con datos Web limpios** (usar el nuevo
   `learn_from_web.py` con varios `--tema` relevantes).
5. **[MEDIA] Correr pytest tras cada cambio** (hoy 283/283 OK).

## Notas de manejo
- `data/*` está gitignored (solo `data/.gitkeep` se versiona). Los modelos y
  corpus son datos locales.
- Los archivos `brain.py`, `train_neural.py`, `talk_to_aria.py`, etc. traían
  cambios SIN commitear del usuario: NO tocarlos sin preguntar.
- Interfaz de chat local: `python scripts/talk_to_aria.py`.
- Monitor ESP32: `pio device monitor -b 115200` (requiere terminal propia).
