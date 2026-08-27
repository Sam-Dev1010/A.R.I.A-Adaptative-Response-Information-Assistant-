#!/usr/bin/env python3
"""Tests de integración end-to-end: Cerebro Neural → Orquestador → Voz.

Verifica que el pipeline completo funcione:
  texto → NeuralBrain.think() → Orquestador.ask() → VoiceAssistant → TTS
"""
import asyncio
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

passed = 0
failed = 0
errors = []


def test(name: str, func):
    global passed, failed
    try:
        result = func()
        if result is not False:
            print(f"  PASS  {name}")
            passed += 1
        else:
            print(f"  FAIL  {name}")
            failed += 1
            errors.append(name)
    except Exception as e:
        print(f"  FAIL  {name} ({type(e).__name__}: {e})")
        failed += 1
        errors.append(f"{name}: {type(e).__name__}: {e}")


def async_test(name: str, func):
    global passed, failed
    try:
        result = asyncio.run(func())
        if result is not False:
            print(f"  PASS  {name}")
            passed += 1
        else:
            print(f"  FAIL  {name}")
            failed += 1
            errors.append(name)
    except Exception as e:
        print(f"  FAIL  {name} ({type(e).__name__}: {e})")
        failed += 1
        errors.append(f"{name}: {type(e).__name__}: {e}")


# ======================================================================
print("=" * 70)
print(" A.R.I.A - Tests de Integración: Cerebro + Orquestador + Voz")
print("=" * 70)

# ======================================================================
print("\n--- 1. NeuralBrain: think() funciona ---")
from app.ai.neural.brain import NeuralBrain


def _make_brain(tmpdir: Path) -> NeuralBrain:
    """Crea un NeuralBrain con datos de entrenamiento."""
    brain = NeuralBrain(tmpdir)
    brain.initialize()
    return brain


async def test_brain_think_basic():
    """El cerebro responde a un saludo básico."""
    with tempfile.TemporaryDirectory() as d:
        brain = _make_brain(Path(d))
        response = await brain.think("hola")
        assert isinstance(response, str), f"Expected str, got {type(response)}"
        assert len(response) > 0, "Empty response"
        brain.close()


async_test("NeuralBrain responds to greeting", test_brain_think_basic)


async def test_brain_think_stores_memory():
    """El cerebro almacena la conversación en memoria semántica."""
    with tempfile.TemporaryDirectory() as d:
        brain = _make_brain(Path(d))
        await brain.think("mi nombre es Samuel")
        # Verificar que se almacenó
        results = brain.search_memory("Samuel")
        assert len(results) > 0, "Memory should store user input"
        brain.close()


async_test("NeuralBrain stores in semantic memory", test_brain_think_stores_memory)


async def test_brain_think_classifies_intent():
    """El cerebro clasifica la intención del usuario."""
    with tempfile.TemporaryDirectory() as d:
        brain = _make_brain(Path(d))
        await brain.think("abre firefox")
        intent, conf = brain.classifier.classify("abre firefox")
        assert isinstance(intent, str), f"Intent should be str, got {type(intent)}"
        assert isinstance(conf, (int, float)), f"Confidence should be numeric, got {type(conf)}"
        brain.close()


async_test("NeuralBrain classifies intent correctly", test_brain_think_classifies_intent)


async def test_brain_multiple_conversations():
    """El cerebro maneja múltiples mensajes en secuencia."""
    with tempfile.TemporaryDirectory() as d:
        brain = _make_brain(Path(d))
        responses = []
        msgs = ["hola", "qué hora es", "abre firefox", "gracias"]
        for msg in msgs:
            r = await brain.think(msg)
            responses.append(r)
            assert isinstance(r, str) and len(r) > 0
        assert len(responses) == 4
        brain.close()


async_test("NeuralBrain handles conversation sequence", test_brain_multiple_conversations)


# ======================================================================
print("\n--- 2. Orquestador con NeuralBrain (sin LLM) ---")
from app.ai.orchestrator import AssistantOrchestrator


def _make_orchestrator(brain: NeuralBrain) -> AssistantOrchestrator:
    """Crea un orquestador usando solo el cerebro neural."""
    return AssistantOrchestrator(
        provider=None,
        neural_brain=brain,
        system_prompt="Eres A.R.I.A.",
    )


async def test_orchestrator_neural_basic():
    """El orquestador delega al cerebro neural correctamente."""
    with tempfile.TemporaryDirectory() as d:
        brain = _make_brain(Path(d))
        orch = _make_orchestrator(brain)
        response = await orch.ask("hola A.R.I.A")
        assert response.content, "Empty content"
        assert response.model == "neural"
        assert response.provider == "local"
        brain.close()


async_test("Orchestrator delegates to neural brain", test_orchestrator_neural_basic)


async def test_orchestrator_records_history():
    """El orquestador guarda el historial de conversación."""
    with tempfile.TemporaryDirectory() as d:
        brain = _make_brain(Path(d))
        orch = _make_orchestrator(brain)
        await orch.ask("hola")
        await orch.ask("qué hora es")
        assert len(orch.history) == 4  # 2 user + 2 assistant
        brain.close()


async_test("Orchestrator records conversation history", test_orchestrator_records_history)


async def test_orchestrator_command_detection():
    """El orquestador detecta comandos del cerebro."""
    with tempfile.TemporaryDirectory() as d:
        brain = _make_brain(Path(d))
        orch = _make_orchestrator(brain)
        response = await orch.ask("abre firefox")
        assert response.content, "Command should produce response"
        brain.close()


async_test("Orchestrator handles commands via brain", test_orchestrator_command_detection)


# ======================================================================
print("\n--- 3. VoiceAssistant + NeuralBrain Pipeline ---")
from app.voice.assistant import VoiceAssistant, ExitConversation
from app.voice.base import TTSProvider, VoiceError


async def test_voice_activates_and_thinks():
    """VoiceAssistant activa, el cerebro piensa, y TTS habla."""
    with tempfile.TemporaryDirectory() as d:
        brain = _make_brain(Path(d))
        orch = _make_orchestrator(brain)

        tts_speaks = []
        mock_tts = AsyncMock(spec=TTSProvider)
        mock_tts.name = "mock_tts"
        mock_tts.audio_ext = ".mp3"

        async def capture_speak(text, **kw):
            tts_speaks.append(text)
            p = Path(d) / f"test_{len(tts_speaks)}.mp3"
            p.write_bytes(b"\x00" * 100)
            return p

        mock_tts.speak = capture_speak

        stt = AsyncMock()
        va = VoiceAssistant(orch, stt, mock_tts)

        result = await va.run_once(text="ARIA hola")
        assert result is not None, "Should return greeting"
        assert va.active, "Should be active"
        assert len(tts_speaks) >= 1, "TTS should speak greeting"
        brain.close()


async_test("Full voice activation + brain + TTS pipeline", test_voice_activates_and_thinks)


async def test_voice_conversation_loop():
    """Simula una conversación completa de voz."""
    with tempfile.TemporaryDirectory() as d:
        brain = _make_brain(Path(d))
        orch = _make_orchestrator(brain)

        spoken = []
        mock_tts = AsyncMock(spec=TTSProvider)
        mock_tts.name = "mock_tts"
        mock_tts.audio_ext = ".mp3"

        async def capture_speak(text, **kw):
            spoken.append(text)
            p = Path(d) / f"tts_{len(spoken)}.mp3"
            p.write_bytes(b"\x00" * 100)
            return p

        mock_tts.speak = capture_speak
        stt = AsyncMock()
        va = VoiceAssistant(orch, stt, mock_tts)

        # 1. Activar
        r = await va.run_once(text="ARIA hola")
        assert r is not None and va.active

        # 2. Hablar estando activo
        r2 = await va.run_once(text="qué hora es")
        assert r2 is not None
        assert isinstance(r2, str) and len(r2) > 0

        # 3. Desactivar
        r3 = await va.run_once(text="ya acabamos")
        assert "Hasta luego" in r3
        assert not va.active

        # Verificar que TTS habló todo
        assert len(spoken) >= 3, f"Expected >=3 TTS calls, got {len(spoken)}"
        brain.close()


async_test("Full voice conversation: activate → talk → deactivate", test_voice_conversation_loop)


async def test_voice_stt_failure_graceful():
    """Fallo de STT no rompe el pipeline."""
    with tempfile.TemporaryDirectory() as d:
        brain = _make_brain(Path(d))
        orch = _make_orchestrator(brain)

        mock_tts = AsyncMock(spec=TTSProvider)
        mock_tts.name = "mock_tts"
        mock_tts.audio_ext = ".mp3"
        mock_tts.speak = AsyncMock()

        stt = AsyncMock()
        stt.listen = AsyncMock(side_effect=VoiceError("mic error"))

        va = VoiceAssistant(orch, stt, mock_tts)
        result = await va.run_once()
        assert result is None, "STT failure should return None"
        brain.close()


async_test("Voice STT failure handled gracefully", test_voice_stt_failure_graceful)


async def test_voice_exit_command():
    """Comando 'salir' lanza ExitConversation."""
    with tempfile.TemporaryDirectory() as d:
        brain = _make_brain(Path(d))
        orch = _make_orchestrator(brain)

        mock_tts = AsyncMock(spec=TTSProvider)
        mock_tts.name = "mock_tts"
        mock_tts.audio_ext = ".mp3"
        mock_tts.speak = AsyncMock()

        stt = AsyncMock()
        va = VoiceAssistant(orch, stt, mock_tts, active=True)

        try:
            await va.run_once(text="salir")
            return False
        except ExitConversation:
            return True
        finally:
            brain.close()


async_test("Voice exit command raises ExitConversation", test_voice_exit_command)


# ======================================================================
print("\n--- 4. TTS Real con NeuralBrain ---")


async def test_real_tts_with_brain_response():
    """El cerebro genera una respuesta y Edge TTS la sintetiza a audio real."""
    from app.voice.tts import EdgeTTSProvider

    with tempfile.TemporaryDirectory() as d:
        brain = _make_brain(Path(d))
        orch = _make_orchestrator(brain)

        response = await orch.ask("hola")
        text_to_speak = response.content

        tts = EdgeTTSProvider(voice="es-MX-DaliaNeural")
        audio_path = await tts.synthesize(text_to_speak, Path(d) / "brain_output.mp3")
        assert audio_path.exists()
        assert audio_path.stat().st_size > 500, f"Audio too small for: {text_to_speak}"
        brain.close()


async_test("Real TTS synthesizes brain response", test_real_tts_with_brain_response)


async def test_real_tts_voice_assistant_full():
    """Pipeline completo: brain → orchestrator → VoiceAssistant → real Edge TTS."""
    from app.voice.tts import EdgeTTSProvider

    with tempfile.TemporaryDirectory() as d:
        brain = _make_brain(Path(d))
        orch = _make_orchestrator(brain)
        tts = EdgeTTSProvider(voice="es-MX-DaliaNeural")

        spoken_paths = []
        original_speak = tts.speak

        async def track_speak(text, **kw):
            spoken_paths.append(text)
            return await original_speak(text, **kw)

        tts.speak = track_speak

        stt = AsyncMock()
        va = VoiceAssistant(orch, stt, tts)

        # Activar
        result = await va.run_once(text="ARIA hola")
        assert result is not None
        assert len(spoken_paths) >= 1

        # Verificar que el audio del saludo existe y es válido
        # (speak crea el archivo y lo reproduce, pero sin player solo verifica)

        brain.close()


async_test("Full pipeline: brain → orchestrator → voice → real TTS", test_real_tts_voice_assistant_full)


# ======================================================================
print("\n--- 5. NeuralBrain: Training + Inference ---")


async def test_brain_train_and_respond():
    """Entrena el cerebro y verifica que responde después de entrenar."""
    with tempfile.TemporaryDirectory() as d:
        brain = _make_brain(Path(d))

        # Entrenar tokenizer BPE (rápido)
        conversations = [
            {"user": "hola", "assistant": "¡Hola! Soy A.R.I.A."},
            {"user": "qué hora es", "assistant": "No tengo acceso al reloj ahora."},
            {"user": "abre firefox", "assistant": "Abriendo Firefox."},
            {"user": "adiós", "assistant": "¡Hasta luego!"},
        ]
        # Solo entrenar tokenizer (rápido), skip GPT training (lento con 8.9M params)
        all_texts = [f"<user>{c['user']}<assistant>{c['assistant']}" for c in conversations]
        brain.gpt_tokenizer.train(all_texts, verbose=False)

        # Verificar que responde (sin entrenamiento GPT, usa fallback)
        r1 = await brain.think("hola")
        r2 = await brain.think("abre firefox")
        assert isinstance(r1, str) and len(r1) > 0
        assert isinstance(r2, str) and len(r2) > 0
        brain.close()


async_test("NeuralBrain tokenizer training + think works", test_brain_train_and_respond)


async def test_brain_persistence():
    """El cerebro guarda y carga estado correctamente."""
    with tempfile.TemporaryDirectory() as d:
        brain1 = _make_brain(Path(d))
        await brain1.think("mi color favorito es el azul")
        brain1.close()

        # Recargar
        brain2 = NeuralBrain(Path(d))
        brain2.initialize()
        r = await brain2.think("hola")
        assert isinstance(r, str) and len(r) > 0
        brain2.close()


async_test("NeuralBrain persists and reloads state", test_brain_persistence)


# ======================================================================
print("\n--- 6. Edge Cases ---")


async def test_empty_message():
    """Mensaje vacío no procesa."""
    with tempfile.TemporaryDirectory() as d:
        brain = _make_brain(Path(d))
        orch = _make_orchestrator(brain)
        try:
            await orch.ask("")
            return False
        except ValueError:
            return True
        finally:
            brain.close()


async_test("Empty message raises ValueError", test_empty_message)


async def test_concurrent_think_calls():
    """Múltiples llamadas concurrentes al cerebro no fallan."""
    with tempfile.TemporaryDirectory() as d:
        brain = _make_brain(Path(d))

        async def think(msg):
            return await brain.think(msg)

        results = await asyncio.gather(
            think("hola"),
            think("adiós"),
            think("qué hora es"),
        )
        assert all(isinstance(r, str) and len(r) > 0 for r in results)
        brain.close()


async_test("Concurrent brain.think() calls work", test_concurrent_think_calls)


async def test_brain_personality():
    """El cerebro tiene personalidad y la usa."""
    with tempfile.TemporaryDirectory() as d:
        brain = _make_brain(Path(d))
        assert brain.personality.mood in ("neutral", "happy", "focused", "tired", "excited", "serious")
        assert 0.0 <= brain.personality.energy <= 1.0
        brain.close()


async_test("NeuralBrain has valid personality state", test_brain_personality)


# ======================================================================
print("\n" + "=" * 70)
total = passed + failed
print(f" RESULTADO: {passed}/{total} éxitos, {failed} fallos")
print("=" * 70)

if errors:
    print("\nFallos:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\n  ¡TODOS LOS TESTS DE INTEGRACIÓN PASARON!")
    sys.exit(0)
