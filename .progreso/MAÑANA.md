# MAÑANA — arranque rápido (Sesión 2)

Léeme primero y arranca sin perder tiempo. (Detalles en `SESION_2026-08-29.md`)

## Estado
- ARIA al 100% en PC (282/282 tests) y en ESP32.
- Pipeline de aprendizaje del internet CREADO y funcionando.
- **Pendiente: decisión del entrenador** (el usuario eligió "que le explique
  más antes de decidir" y NO decidió aún).

## El hallazgo que marca el rumbo
El entrenador Python-puro tarda ~3 min/época con solo 118 secuencias →
inviable para "aprender del internet" (miles de secuencias).

## Plan de la sesión (en orden)
1. **Etapa 1 - vectorizar el forward** (`app/ai/neural/transformer/gpt_model.py`):
   - Reescribir `forward()` con NumPy (matmul/broadcast) en vez de lazos.
   - VALIDAR: las salidas deben ser idénticas al actual. Usar los tests
     `pytest tests/test_neural.py` como red de seguridad + comparar manualmente.
   - Medir tiempo por época antes (~168s) vs después. Mostrar la mejora al
     usuario para que confirme la Opción 1 (NumPy).
2. **Si el usuario confirma**: Etapa 2 - vectorizar el `backward()`:
   - Validar con gradientes numéricos (finite differences).
   - Reentrenar el MISMO modelo actual y comparar loss/accuracy (debe quedar
     igual o mejor, y mucho más rápido).
3. **Reentrenar** con el corpus ampliado: `python scripts/retrain_full.py`
   (hoy ~40-45 min; tras optimizar debería bajar a minutos).
4. Medir loss/ppl/acc ANTES vs DESPUÉS.
5. `pytest` al final para no romper nada.

## Comandos útiles
```bash
source .venv/bin/activate
python scripts/learn_from_web.py --articulos 20          # crecer corpus (opcional)
python scripts/retrain_full.py                           # reentrenar todo
python scripts/talk_to_aria.py                           # hablar con ARIA local
pytest                                                   # 282 tests
```

## Recordatorio importante
- NO tocar `brain.py`, `train_neural.py`, `talk_to_aria.py` y los demás
  archivos que ya venían modificados del usuario SIN preguntar.
- Guardar progreso al final de cada día creando `.progreso/SESION_YYYY-MM-DD.md`
  y actualizando `.progreso/ESTADO.md`.
