# ESTADO ACTUAL DE A.R.I.A (canónica)

Actualizado: 2026-09-09 (Sesión 4)
Ver histórico detallado en `SESION_2026-09-09.md` y sesiones anteriores.

## Resumen ejecutivo
ARIA funciona **al 100% en el PC** (285/285 tests) y en el ESP32. El
entrenador del GPT local quedó **por lotes vectorizado ~10.3x** (B=32) con
equivalencia bit-exacta demostrada (B=1) y batch-mean (B>1), listo para
reentrenar con un corpus MUCHO mayor. El modelo escalado sigue SIN promocionar
y la fase actual es de **escalado de corpus/datos**.

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
1. **[ALTA] Escalar corpus**: recolectar varios MB de texto en español (Wikipedia
   es) y reentrenar `scaled_model_v3` con el trainer por lotes (B=32, ~10x más
   rápido). Datos: el freno del GPT local es DATA, no capacidad.
2. **[ALTA] Decidir promoción**: validar generación del escalado (sin `<unk>` ni
   bucles) y decidir si se despliega; si no mejora, GPT local queda experimental.
3. **[MEDIA] Revisar tokenizer BPE**: merges que fragmentan palabras de forma
   rara ("qprogram", "Armin") que alimentan la degeneración.
4. **[BAJA] Limpiar experimentos**: `data/scaled_model*` no desplegados.
5. **[ALTA] Correr pytest tras cada cambio** (hoy 285/285 OK) y
   `/tmp/equivalence_test.py` (contrato por lotes) antes de tocar trainer.

## Notas de manejo
- `data/*` está gitignored (solo `data/.gitkeep` se versiona). Modelos y corpus
  son datos locales.
- Los archivos `brain.py`, `train_neural.py`, `talk_to_aria.py` traían cambios
  SIN commitear del usuario: NO tocarlos sin preguntar.
- Interfaz de chat local: `python scripts/talk_to_aria.py`.
- Monitor ESP32: `pio device monitor -b 115200` (requiere terminal propia).