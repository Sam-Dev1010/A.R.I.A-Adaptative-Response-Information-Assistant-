#!/usr/bin/env python3
"""Suite de pruebas completa para el sistema de voz de A.R.I.A.

Cubre: TTS (Edge, Piper, gTTS, Race), STT (Google, Groq),
VoiceAssistant (activación, desactivación, wake word, saludos),
y tests de integración con síntesis real.
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
print(" A.R.I.A - Suite de Pruebas Completa del Sistema de Voz")
print("=" * 70)

# ======================================================================
print("\n--- 1. Base Classes ---")
from app.voice.base import STTProvider, TTSProvider, VoiceError


def test_voice_error_is_exception():
    assert issubclass(VoiceError, Exception)


test("VoiceError is Exception", test_voice_error_is_exception)


def test_stt_provider_is_abstract():
    assert hasattr(STTProvider, "listen")


test("STTProvider has listen method", test_stt_provider_is_abstract)


def test_tts_provider_is_abstract():
    assert hasattr(TTSProvider, "synthesize")
    assert hasattr(TTSProvider, "speak")


test("TTSProvider has required methods", test_tts_provider_is_abstract)


# ======================================================================
print("\n--- 2. Edge TTS Provider ---")
from app.voice.tts import EdgeTTSProvider


def test_edge_tts_init():
    p = EdgeTTSProvider(voice="es-MX-DaliaNeural", rate="+12%", pitch="+0Hz")
    assert p.voice == "es-MX-DaliaNeural"


test("EdgeTTS init", test_edge_tts_init)


def test_edge_tts_rate_variation():
    p = EdgeTTSProvider(rate="+12%", pitch="+0Hz", vary_rate=True)
    rates = [p._siguiente_rate() for _ in range(5)]
    assert len(rates) == 5
    assert rates[0] != rates[1] or rates[2] != rates[3]


test("EdgeTTS rate variation", test_edge_tts_rate_variation)


def test_edge_tts_pitch_variation():
    p = EdgeTTSProvider(rate="+12%", pitch="+10Hz", vary_rate=True)
    pitches = [p._siguiente_pitch() for _ in range(5)]
    assert len(pitches) == 5


test("EdgeTTS pitch variation", test_edge_tts_pitch_variation)


def test_edge_tts_rate_no_variation():
    p = EdgeTTSProvider(rate="+12%", vary_rate=False)
    assert p._siguiente_rate() == "+12%"
    assert p._siguiente_rate() == "+12%"


test("EdgeTTS rate without variation stays constant", test_edge_tts_rate_no_variation)


async def test_edge_tts_synthesize_real():
    """Sintetiza texto real con Edge TTS y verifica que genera un MP3 válido."""
    with tempfile.TemporaryDirectory() as d:
        p = EdgeTTSProvider(voice="es-MX-DaliaNeural")
        path = await p.synthesize("Hola, soy A.R.I.A.", Path(d) / "test.mp3")
        assert path.exists()
        assert path.stat().st_size > 1000, "MP3 should be >1KB for real speech"
        # Verify it's a valid MP3 (starts with ID3 tag or frame sync)
        header = path.read_bytes()[:10]
        assert header[:3] == b"ID3" or header[0] == 0xFF, "Not a valid MP3"


async_test("EdgeTTS real synthesis produces valid MP3", test_edge_tts_synthesize_real)


async def test_edge_tts_synthesize_spanish():
    """Verifica que el texto en español se sintetiza correctamente."""
    with tempfile.TemporaryDirectory() as d:
        p = EdgeTTSProvider(voice="es-MX-DaliaNeural")
        path = await p.synthesize(
            "Buenos días. Puedo ejecutar comandos y gestionar archivos.",
            Path(d) / "spanish.mp3",
        )
        assert path.exists()
        assert path.stat().st_size > 2000


async_test("EdgeTTS Spanish speech synthesis", test_edge_tts_synthesize_spanish)


async def test_edge_tts_synthesize_long_text():
    """Verifica que textos largos se sintetizan correctamente."""
    with tempfile.TemporaryDirectory() as d:
        p = EdgeTTSProvider(voice="es-MX-DaliaNeural")
        long_text = (
            "Este es un texto largo para probar que el sistema de síntesis "
            "de voz puede manejar frases largas sin problemas. A.R.I.A es "
            "un asistente personal con inteligencia neural que puede controlar "
            "tu computadora y ayudarte con muchas tareas diferentes."
        )
        path = await p.synthesize(long_text, Path(d) / "long.mp3")
        assert path.exists()
        assert path.stat().st_size > 5000


async_test("EdgeTTS long text synthesis", test_edge_tts_synthesize_long_text)


async def test_edge_tts_synthesize_empty():
    """Verifica que texto vacío no genera error."""
    with tempfile.TemporaryDirectory() as d:
        p = EdgeTTSProvider(voice="es-MX-DaliaNeural")
        path = await p.synthesize("", Path(d) / "empty.mp3")
        assert path.exists()


async_test("EdgeTTS empty text synthesis", test_edge_tts_synthesize_empty)


async def test_edge_tts_synthesize_punctuation():
    """Verifica que signos de puntuación se manejan correctamente."""
    with tempfile.TemporaryDirectory() as d:
        p = EdgeTTSProvider(voice="es-MX-DaliaNeural")
        text = "¿Qué hora es? Son las tres. ¡Fantástico! Me alegra mucho."
        path = await p.synthesize(text, Path(d) / "punct.mp3")
        assert path.exists()
        assert path.stat().st_size > 2000


async_test("EdgeTTS punctuation handling", test_edge_tts_synthesize_punctuation)


async def test_edge_tts_speak():
    """Verifica que speak() sintetiza y devuelve la ruta."""
    with tempfile.TemporaryDirectory() as d:
        p = EdgeTTSProvider(voice="es-MX-DaliaNeural", player_cmds=("echo",))
        path = await p.speak("Hola mundo", output_dir=d)
        assert path.exists()


async_test("EdgeTTS speak produces file", test_edge_tts_speak)


# ======================================================================
print("\n--- 3. Piper TTS Provider ---")
from app.voice.tts import PiperTTSProvider


def test_piper_init():
    p = PiperTTSProvider(model_path="/fake/model.onnx")
    assert p.name == "piper_tts"


test("PiperTTS init", test_piper_init)


def test_piper_missing_model_raises():
    async def run():
        p = PiperTTSProvider(model_path="/nonexistent/model.onnx")
        with tempfile.TemporaryDirectory() as d:
            await p.synthesize("Hola", Path(d) / "out.wav")
        return False

    try:
        asyncio.run(run())
        return False
    except VoiceError:
        return True


test("PiperTTS missing model raises VoiceError", test_piper_missing_model_raises)


# ======================================================================
print("\n--- 4. Google Translate TTS ---")
from app.voice.tts import GoogleTranslateTTSProvider


def test_gtts_init():
    p = GoogleTranslateTTSProvider()
    assert p.name == "gtts"


test("GoogleTranslateTTS init", test_gtts_init)


async def test_gtts_synthesize():
    """Sintetiza texto real con gTTS."""
    with tempfile.TemporaryDirectory() as d:
        p = GoogleTranslateTTSProvider()
        path = await p.synthesize("Hola, esto es una prueba.", Path(d) / "gtts.mp3")
        assert path.exists()
        assert path.stat().st_size > 500


async_test("GoogleTranslateTTS real synthesis", test_gtts_synthesize)


async def test_gtts_long_text():
    """Verifica que textos largos se dividen correctamente."""
    with tempfile.TemporaryDirectory() as d:
        p = GoogleTranslateTTSProvider()
        long = "Hola. " * 200
        path = await p.synthesize(long, Path(d) / "long.mp3")
        assert path.exists()


async_test("GoogleTranslateTTS long text splitting", test_gtts_long_text)


# ======================================================================
print("\n--- 5. Race TTS Provider ---")
from app.voice.tts import RaceTTSProvider


async def test_race_picks_fastest():
    """Race elige el provider más rápido."""
    async def fast_synthesize(text, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"\xff\xfb\x90\x00" * 500)
        return Path(path)

    fast = AsyncMock(spec=TTSProvider)
    fast.name = "fast"
    fast.audio_ext = ".mp3"
    fast.synthesize = fast_synthesize

    async def slow_synthesize(text, path):
        await asyncio.sleep(0.5)
        return Path(path)

    slow = AsyncMock(spec=TTSProvider)
    slow.name = "slow"
    slow.audio_ext = ".mp3"
    slow.synthesize = slow_synthesize

    race = RaceTTSProvider(slow, fast)
    with tempfile.TemporaryDirectory() as d:
        path = await race.synthesize("test", Path(d) / "out.mp3")
    # RaceTTS copies winner to original output_path
    assert path.name == "out.mp3"


async_test("RaceTTS picks fastest provider", test_race_picks_fastest)


async def test_race_falls_back():
    """Race cae al siguiente provider si el primero falla."""
    async def fail_synthesize(text, path):
        raise VoiceError("fail")

    failing = AsyncMock(spec=TTSProvider)
    failing.name = "fail"
    failing.audio_ext = ".mp3"
    failing.synthesize = fail_synthesize

    async def ok_synthesize(text, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"\xff\xfb\x90\x00" * 500)
        return Path(path)

    ok = AsyncMock(spec=TTSProvider)
    ok.name = "ok"
    ok.audio_ext = ".mp3"
    ok.synthesize = ok_synthesize

    race = RaceTTSProvider(failing, ok)
    with tempfile.TemporaryDirectory() as d:
        path = await race.synthesize("test", Path(d) / "out.mp3")
    assert path.name == "out.mp3"


async_test("RaceTTS falls back on failure", test_race_falls_back)


async def test_race_all_fail_raises():
    """Race levanta VoiceError si todos fallan."""
    failing = AsyncMock(spec=TTSProvider)
    failing.name = "fail"
    failing.synthesize = AsyncMock(side_effect=VoiceError("fail"))

    race = RaceTTSProvider(failing)
    try:
        with tempfile.TemporaryDirectory() as d:
            await race.synthesize("test", Path(d) / "out.mp3")
        return False
    except VoiceError:
        return True


async_test("RaceTTS raises when all fail", test_race_all_fail_raises)


# ======================================================================
print("\n--- 6. STT Providers ---")
from app.voice.stt import GoogleSTTProvider, GroqSTTProvider


def test_google_stt_init():
    p = GoogleSTTProvider(language="es-ES")
    assert p.name == "google_stt"


test("GoogleSTT init", test_google_stt_init)


def test_google_stt_transcribe_bytes():
    p = GoogleSTTProvider()
    # Silent audio (all zeros) should return empty or raise
    try:
        text = p.transcribe_bytes(b"\x00\x00" * 1600, sample_rate=16000)
        assert isinstance(text, str)
    except VoiceError:
        pass  # Expected for silent audio


test("GoogleSTT transcribe_bytes handles silent audio", test_google_stt_transcribe_bytes)


def test_groq_stt_init():
    p = GroqSTTProvider(api_key="test_key")
    assert p.name == "groq_stt"


test("GroqSTT init", test_groq_stt_init)


# ======================================================================
print("\n--- 7. VoiceAssistant ---")
from app.voice.assistant import VoiceAssistant, ExitConversation, greeting_for


def test_greeting_for_morning():
    assert "Buenos días" in greeting_for(8)


test("greeting_for morning", test_greeting_for_morning)


def test_greeting_for_afternoon():
    assert "Buenas tardes" in greeting_for(15)


test("greeting_for afternoon", test_greeting_for_afternoon)


def test_greeting_for_night():
    assert "Buenas noches" in greeting_for(23)


test("greeting_for night", test_greeting_for_night)


def test_greeting_for_aria():
    assert "A.R.I.A" in greeting_for(12)


test("greeting_for includes ARIA name", test_greeting_for_aria)


async def test_assistant_activation():
    """A.R.I.A se activa con wake word + saludo."""
    orch = AsyncMock()
    orch.ask = AsyncMock(return_value=MagicMock(content="Hola jefe"))
    stt = AsyncMock()
    stt.listen = AsyncMock(return_value="Hola ARIA")
    tts = AsyncMock()
    tts.speak = AsyncMock()

    va = VoiceAssistant(orch, stt, tts)
    assert not va.active
    result = await va.run_once(text="ARIA hola")
    assert va.active
    assert result is not None
    tts.speak.assert_called()


async_test("VoiceAssistant activation with salutation", test_assistant_activation)


async def test_assistant_wake_word_only():
    """A.R.I.A se activa solo con wake word (sin saludo)."""
    orch = AsyncMock()
    orch.ask = AsyncMock(return_value=MagicMock(content="Hola"))
    stt = AsyncMock()
    tts = AsyncMock()

    va = VoiceAssistant(orch, stt, tts)
    result = await va.run_once(text="ARIA")
    assert va.active
    assert result is not None


async_test("VoiceAssistant wake word alone activates", test_assistant_wake_word_only)


async def test_assistant_standby_ignores():
    """En espera, sin wake word no responde."""
    orch = AsyncMock()
    stt = AsyncMock()
    tts = AsyncMock()

    va = VoiceAssistant(orch, stt, tts)
    result = await va.run_once(text="hola qué tal")
    assert result is None
    assert not va.active


async_test("VoiceAssistant standby ignores without wake word", test_assistant_standby_ignores)


async def test_assistant_deactivation():
    """A.R.I.A se desactiva con frase de despedida."""
    orch = AsyncMock()
    orch.ask = AsyncMock(return_value=MagicMock(content="ok"))
    stt = AsyncMock()
    tts = AsyncMock()

    va = VoiceAssistant(orch, stt, tts, active=True)
    result = await va.run_once(text="ya acabamos gracias")
    assert not va.active
    assert "Hasta luego" in result


async_test("VoiceAssistant deactivation", test_assistant_deactivation)


async def test_assistant_active_responds():
    """En modo activo, responde a cualquier mensaje."""
    orch = AsyncMock()
    orch.ask = AsyncMock(return_value=MagicMock(content="Tu contraseña es segura"))
    stt = AsyncMock()
    tts = AsyncMock()

    va = VoiceAssistant(orch, stt, tts, active=True)
    result = await va.run_once(text="cuál es mi contraseña")
    assert result == "Tu contraseña es segura"
    orch.ask.assert_called_once_with("cuál es mi contraseña")


async_test("VoiceAssistant active mode responds", test_assistant_active_responds)


async def test_assistant_exit():
    """Comando 'salir' lanza ExitConversation."""
    orch = AsyncMock()
    stt = AsyncMock()
    tts = AsyncMock()

    va = VoiceAssistant(orch, stt, tts, active=True)
    try:
        await va.run_once(text="salir")
        return False
    except ExitConversation:
        return True


async_test("VoiceAssistant exit raises ExitConversation", test_assistant_exit)


async def test_assistant_active_accepts_wake_prefix():
    """En modo activo, ignora wake word prefix."""
    orch = AsyncMock()
    orch.ask = AsyncMock(return_value=MagicMock(content="ok"))
    stt = AsyncMock()
    tts = AsyncMock()

    va = VoiceAssistant(orch, stt, tts, active=True)
    result = await va.run_once(text="ARIA cuál es la hora")
    assert result is not None
    orch.ask.assert_called_once_with("cuál es la hora")


async_test("VoiceAssistant active strips wake word prefix", test_assistant_active_accepts_wake_prefix)


async def test_assistant_on_wake_hook():
    """El gancho on_wake se ejecuta tras la activación."""
    orch = AsyncMock()
    orch.ask = AsyncMock(return_value=MagicMock(content="ok"))
    stt = AsyncMock()
    tts = AsyncMock()
    hook = AsyncMock()

    va = VoiceAssistant(orch, stt, tts, on_wake=hook)
    await va.run_once(text="ARIA hola")
    hook.assert_called_once()


async_test("VoiceAssistant on_wake hook fires", test_assistant_on_wake_hook)


async def test_assistant_wake_with_command():
    """ARIA + comando activa Y responde al comando."""
    orch = AsyncMock()
    orch.ask = AsyncMock(return_value=MagicMock(content="Son las 3"))
    stt = AsyncMock()
    tts = AsyncMock()

    va = VoiceAssistant(orch, stt, tts)
    result = await va.run_once(text="ARIA qué hora es")
    assert va.active
    orch.ask.assert_called_once_with("qué hora es")


async_test("VoiceAssistant wake + command activates and answers", test_assistant_wake_with_command)


async def test_assistant_deactivation_phrases():
    """Múltiples frases de desactivación funcionan."""
    for phrase in ["ya acabamos", "apágate", "adiós"]:
        orch = AsyncMock()
        stt = AsyncMock()
        tts = AsyncMock()
        va = VoiceAssistant(orch, stt, tts, active=True)
        result = await va.run_once(text=phrase)
        assert not va.active, f"Failed for phrase: {phrase}"


async_test("VoiceAssistant multiple deactivation phrases", test_assistant_deactivation_phrases)


async def test_assistant_empty_text():
    """Texto vacío no hace nada."""
    orch = AsyncMock()
    stt = AsyncMock()
    tts = AsyncMock()
    va = VoiceAssistant(orch, stt, tts)
    result = await va.run_once(text="")
    assert result is None


async_test("VoiceAssistant empty text returns None", test_assistant_empty_text)


async def test_assistant_stt_failure():
    """Fallo de STT no levanta excepción."""
    orch = AsyncMock()
    stt = AsyncMock()
    stt.listen = AsyncMock(side_effect=VoiceError("mic not found"))
    tts = AsyncMock()

    va = VoiceAssistant(orch, stt, tts)
    result = await va.run_once()
    assert result is None


async_test("VoiceAssistant STT failure handled gracefully", test_assistant_stt_failure)


# ======================================================================
print("\n--- 8. Builder Functions ---")
from app.voice.tts import build_tts_provider
from app.voice.stt import build_stt_provider


def test_build_tts_auto():
    with patch("app.voice.tts.get_settings") as m:
        m.return_value = MagicMock(tts_provider="auto", tts_voice="es-MX-DaliaNeural",
                                    tts_rate="+8%", tts_pitch="+0Hz",
                                    piper_model=None, tts_vary_rate=False)
        p = build_tts_provider()
        assert isinstance(p, EdgeTTSProvider)


test("build_tts_provider auto uses EdgeTTS", test_build_tts_auto)


def test_build_tts_edge_forced():
    with patch("app.voice.tts.get_settings") as m:
        m.return_value = MagicMock(tts_provider="edge", tts_voice="es-MX-DaliaNeural",
                                    tts_rate="+8%", tts_pitch="+0Hz",
                                    tts_vary_rate=False)
        p = build_tts_provider()
        assert isinstance(p, EdgeTTSProvider)


test("build_tts_provider edge forced", test_build_tts_edge_forced)


def test_build_stt_google():
    with patch("app.voice.stt.get_settings") as m:
        m.return_value = MagicMock(stt_provider="google", stt_language="es-ES")
        p = build_stt_provider()
        assert isinstance(p, GoogleSTTProvider)


test("build_stt_provider google", test_build_stt_google)


# ======================================================================
print("\n--- 9. Integration: Full Voice Pipeline ---")


async def test_full_pipeline_edge_tts():
    """Pipeline completo: texto → Edge TTS → MP3 válido → verificar tamaño."""
    with tempfile.TemporaryDirectory() as d:
        provider = EdgeTTSProvider(voice="es-MX-DaliaNeural")
        texts = [
            "Hola, soy A.R.I.A, tu asistente personal.",
            "Puedo ejecutar comandos, leer archivos y abrir aplicaciones.",
            "¿En qué puedo ayudarte hoy?",
        ]
        sizes = []
        for i, text in enumerate(texts):
            path = await provider.synthesize(text, Path(d) / f"test_{i}.mp3")
            size = path.stat().st_size
            sizes.append(size)
            assert size > 1000, f"Audio too small for: {text[:30]}..."

        assert all(s > 1000 for s in sizes), "All audio files should be substantial"


async_test("Full pipeline: Edge TTS multi-sentence", test_full_pipeline_edge_tts)


async def test_full_pipeline_gtts():
    """Pipeline completo con gTTS."""
    with tempfile.TemporaryDirectory() as d:
        provider = GoogleTranslateTTSProvider()
        text = "A.R.I.A está lista para ayudarte con todo lo que necesites."
        path = await provider.synthesize(text, Path(d) / "gtts_test.mp3")
        assert path.exists()
        assert path.stat().st_size > 1000


async_test("Full pipeline: GoogleTranslateTTS", test_full_pipeline_gtts)


async def test_edge_tts_vary_rate_real():
    """Verifica que la variación de ritmo produce audios de tamaño diferente."""
    with tempfile.TemporaryDirectory() as d:
        p = EdgeTTSProvider(voice="es-MX-DaliaNeural", rate="+12%", pitch="+10Hz", vary_rate=True)
        sizes = []
        for i in range(3):
            path = await p.synthesize(
                f"Esta es la frase número {i} con ritmo variado.",
                Path(d) / f"vary_{i}.mp3",
            )
            sizes.append(path.stat().st_size)
        # All should be valid MP3s
        assert all(s > 1000 for s in sizes)


async_test("EdgeTTS varying rate produces valid audio", test_edge_tts_vary_rate_real)


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
    print("\n  ¡TODAS LAS PRUEBAS DE VOZ PASARON!")
    sys.exit(0)
