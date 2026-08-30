# ESTADO ACTUAL DE A.R.I.A (canónica)

Actualizado: 2026-08-29 (Sesión 1)
Ver histórico detallado en `SESION_2026-08-29.md`.

## Resumen ejecutivo
ARIA funciona **al 100% en el PC** (282/282 tests) y el firmware ya está
**subido al ESP32**. Comenzó la **Fase 2** (que sea su propio LLM que aprende
del internet). El pipeline de aprendizaje del internet está creado y funciona.
Queda pendiente la decisión estratégica sobre el entrenador (ver abajo).

## Qué funciona
- **PC completo**: cerebro neural GPT local (377,472 params) + voz + tools +
  memoria + GUI web. Sin API keys para el cerebro local.
- **ESP32**: nodo de IA local con TensorFlow Lite Micro (inferencia offline).
- **Aprendizaje del internet**: `scripts/learn_from_web.py` baja Wikipedia en
  español y acumula corpus en `data/neural/corpus.json`. `retrain_full.py` ya
  fusiona ese corpus al reentrenar.

## Datos / modelos actuales
- Modelo GPT: vocab 2048, embed 64, 4 heads, 2 layers, seq 256 = 377,472 params.
- Corpus aprendido: `data/neural/corpus.json` = 8 conv + 2 textos.
- Base de entrenamiento (en código): 118 conv + 48 textos (hardcoded).
- Total al retrain: 126 conv + 50 textos.

## CUADRO DE MANDO — pendientes priorizados
1. **[CRÍTICO] Decidir estrategia del entrenador** (NumPy vs framework vs
   congelar). El entrenador actual tarda ~3 min/época con 118 secuencias →
   inviable para escalar. Mostrar Etapa 1 de NumPy al usuario para que decida.
2. **[ALTA] Vectorizar el forward** del transformer (Etapa 1) y validar
   equivalencia de salidas con tests.
3. **[ALTA] Reentrenar el GPT** con el corpus ampliado y medir loss/perplexidad
   antes vs después.
4. **[MEDIA] Escalar capacidad del modelo** (embed_dim, capas) en CPU.
5. **[MEDIA] Evaluar calidad de generación libre** frente al baseline.
6. **[MEDIA] Correr pytest** tras cada cambio (hoy 282/282 OK).

## Notas de manejo
- `data/*` está gitignored (solo `data/.gitkeep` se versiona). Los modelos y
  corpus son datos locales.
- Los archivos `brain.py`, `train_neural.py`, `talk_to_aria.py`, etc. ya venían
  con cambios SIN commitear del usuario: NO tocarlos sin preguntar.
- Interfaz de chat local: `python scripts/talk_to_aria.py`.
- Monitor ESP32: `pio device monitor -b 115200` (requiere terminal propia).
