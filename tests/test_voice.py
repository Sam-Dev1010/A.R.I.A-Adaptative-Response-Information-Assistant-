"""Tests de STT, TTS y el asistente de voz (FASES 5-7)."""
from pathlib import Path

import pytest

from app.ai.orchestrator import AssistantOrchestrator
from app.ai.providers.base import LLMProvider
from app.ai.schemas import ChatResponse, ChatRole, TokenUsage
from app.voice.assistant import ExitConversation, VoiceAssistant, greeting_for
from app.voice.base import STTProvider, TTSProvider, VoiceError
from app.voice.stt import GoogleSTTProvider
from app.voice.tts import EdgeTTSProvider, PiperTTSProvider, build_tts_provider


class FakeRecognizer:
    """Reemplazo de speech_recognition.Recognizer."""

    def __init__(self, text: str) -> None:
        self._text = text

    def recognize_google(self, audio, language=None) -> str:
        return self._text


@pytest.mark.asyncio
async def test_google_stt_listen_returns_text(monkeypatch):
    provider = GoogleSTTProvider(language="es-ES")

    class FakeAudio:
        pass

    monkeypatch.setattr(provider, "_record", lambda: FakeAudio())
    monkeypatch.setattr(provider, "_get_recognizer", lambda: FakeRecognizer("hola sia"))

    text = await provider.listen()

    assert text == "hola sia"


@pytest.mark.asyncio
async def test_google_stt_unknown_audio_raises(monkeypatch):
    import speech_recognition as sr

    provider = GoogleSTTProvider()

    class BoomRecognizer(FakeRecognizer):
        def recognize_google(self, audio, language=None):
            raise sr.UnknownValueError

    monkeypatch.setattr(provider, "_record", lambda: object())
    monkeypatch.setattr(provider, "_get_recognizer", lambda: BoomRecognizer(""))

    with pytest.raises(VoiceError, match="No se entendió"):
        await provider.listen()


def test_google_stt_transcribe_bytes(monkeypatch):
    provider = GoogleSTTProvider(language="es-ES")
    monkeypatch.setattr(provider, "_get_recognizer", lambda: FakeRecognizer("hola sia"))

    text = provider.transcribe_bytes(b"\x00\x00" * 160, sample_rate=16000)

    assert text == "hola sia"


class FakeCommunicate:
    """Reemplazo de edge_tts.Communicate (constructor + save async)."""

    def __init__(self, text: str, voice: str, *, rate: str = "+0%", pitch: str = "+0Hz") -> None:
        self._text = text
        self.rate = rate
        self.pitch = pitch

    async def save(self, path) -> None:
        self.saved_path = path


@pytest.mark.asyncio
async def test_edge_tts_synthesize_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr("edge_tts.Communicate", FakeCommunicate)
    provider = EdgeTTSProvider()

    out = tmp_path / "audio.mp3"
    path = await provider.synthesize("Hola", out)

    assert path == out


@pytest.mark.asyncio
async def test_edge_tts_speak_uses_available_player(tmp_path, monkeypatch):
    class RealisticFake(FakeCommunicate):
        async def save(self, path):
            Path(path).write_bytes(b"audio")
            self.saved_path = path

    played = []

    def fake_play(self, path):
        played.append(path)

    monkeypatch.setattr("edge_tts.Communicate", RealisticFake)
    monkeypatch.setattr(EdgeTTSProvider, "_play", fake_play)
    provider = EdgeTTSProvider()

    path = await provider.speak("Hola", output_dir=tmp_path)

    assert path.exists()
    assert len(played) == 1


@pytest.mark.asyncio
async def test_edge_tts_passes_rate_and_pitch(tmp_path, monkeypatch):
    last = {}

    class CapturingFake(FakeCommunicate):
        async def save(self, path):
            Path(path).write_bytes(b"audio")
            self.saved_path = path

    def fake_communicate(text, voice, **kwargs):
        last["rate"] = kwargs.get("rate")
        last["pitch"] = kwargs.get("pitch")
        return CapturingFake(text, voice, **kwargs)

    def fake_play(self, path):
        pass

    monkeypatch.setattr("edge_tts.Communicate", fake_communicate)
    monkeypatch.setattr(EdgeTTSProvider, "_play", fake_play)
    provider = EdgeTTSProvider(rate="+20%", pitch="+30Hz")

    await provider.speak("Hola", output_dir=tmp_path)

    assert last == {"rate": "+20%", "pitch": "+30Hz"}


# --- VoiceAssistant ---


class FakeSTT(STTProvider):
    name = "fake_stt"

    def __init__(self, text: str | list[str]) -> None:
        self._texts = [text] if isinstance(text, str) else list(text)

    async def listen(self, *, language=None) -> str:
        return self._texts.pop(0)


class FakeTTS(TTSProvider):
    name = "fake_tts"

    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def synthesize(self, text, output_path) -> Path:
        return Path(output_path)

    async def speak(self, text, *, output_dir=None) -> Path:
        self.spoken.append(text)
        return Path("fake.mp3")


class EchoProvider(LLMProvider):
    name = "echo"
    model = "echo-model"

    async def chat(self, messages, tools=None) -> ChatResponse:
        user = next(m.content for m in reversed(messages) if m.role is ChatRole.USER)
        return ChatResponse(
            content=f"ARIA dice: {user}",
            model=self.model,
            provider=self.name,
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1),
        )


def _assistant(stt: STTProvider, tts: FakeTTS, wake_word: str | None = "aria") -> VoiceAssistant:
    orchestrator = AssistantOrchestrator(EchoProvider())
    return VoiceAssistant(orchestrator, stt, tts, wake_word=wake_word)


@pytest.mark.asyncio
async def test_activation_with_salutation_greets():
    tts = FakeTTS()
    assistant = _assistant(FakeSTT("aria buenos días"), tts)

    reply = await assistant.run_once()

    assert reply == "¡Buenos días! Soy A.R.I.A, ¿en qué puedo ayudarle, jefe?"
    assert tts.spoken == [reply]
    assert assistant.active is True


@pytest.mark.asyncio
async def test_wake_word_alone_activates_with_time_greeting():
    tts = FakeTTS()
    assistant = _assistant(FakeSTT("aria"), tts)

    reply = await assistant.run_once()

    assert reply in {greeting_for(7), greeting_for(14), greeting_for(22)}
    assert tts.spoken == [reply]
    assert assistant.active is True


@pytest.mark.asyncio
async def test_standby_answers_command_with_wake_word():
    tts = FakeTTS()
    assistant = _assistant(FakeSTT("aria qué hora es"), tts)

    reply = await assistant.run_once()

    assert reply == "ARIA dice: qué hora es"
    assert len(tts.spoken) == 2
    assert tts.spoken[1] == reply
    assert assistant.active is True


@pytest.mark.asyncio
async def test_on_wake_fires_after_activation():
    tts = FakeTTS()
    fired = []
    orchestrator = AssistantOrchestrator(EchoProvider())

    async def hook():
        fired.append(True)

    assistant = VoiceAssistant(orchestrator, FakeSTT("aria hola"), tts, on_wake=hook)
    await assistant.run_once()

    assert fired == [True]
    assert assistant.active is True


@pytest.mark.asyncio
async def test_standby_ignores_without_wake_word():
    tts = FakeTTS()
    assistant = _assistant(FakeSTT("hola a todos"), tts)

    reply = await assistant.run_once()

    assert reply is None
    assert tts.spoken == []
    assert assistant.active is False


@pytest.mark.parametrize(
    ("hour", "expected"),
    [
        (0, "¡Buenas noches!"),
        (4, "¡Buenas noches!"),
        (5, "¡Buenos días!"),
        (9, "¡Buenos días!"),
        (11, "¡Buenos días!"),
        (12, "¡Buenas tardes!"),
        (15, "¡Buenas tardes!"),
        (19, "¡Buenas tardes!"),
        (20, "¡Buenas noches!"),
        (23, "¡Buenas noches!"),
    ],
)
def test_greeting_for_by_hour(hour, expected):
    assert greeting_for(hour).startswith(expected)


@pytest.mark.asyncio
async def test_run_once_wake_word_alone_greets():
    tts = FakeTTS()
    assistant = _assistant(FakeSTT("aria"), tts)

    reply = await assistant.run_once()

    assert reply in {greeting_for(7), greeting_for(14), greeting_for(22)}
    assert "puedo ayudarle" in reply
    assert tts.spoken == [reply]


@pytest.mark.asyncio
async def test_active_answers_without_wake_word():
    tts = FakeTTS()
    assistant = _assistant(FakeSTT("aria hola"), tts)
    await assistant.run_once()

    reply = await assistant.run_once(text="¿qué hora es?")

    assert reply == "ARIA dice: ¿qué hora es?"
    assert assistant.active is True


@pytest.mark.asyncio
async def test_active_accepts_wake_word_prefix():
    tts = FakeTTS()
    assistant = _assistant(FakeSTT("aria hola"), tts)
    await assistant.run_once()

    reply = await assistant.run_once(text="ARIA, ¿qué hora es?")

    assert reply == "ARIA dice: ¿qué hora es?"


@pytest.mark.asyncio
async def test_deactivation_puts_standby():
    tts = FakeTTS()
    assistant = _assistant(FakeSTT("aria hola"), tts)
    await assistant.run_once()

    reply = await assistant.run_once(text="ARIA ya acabamos gracias")

    assert reply == "¡Hasta luego! Estaré en espera."
    assert tts.spoken[-1] == reply
    assert assistant.active is False

    later = await assistant.run_once(text="¿qué hora es?")
    assert later is None


@pytest.mark.asyncio
async def test_no_wake_responds_always():
    tts = FakeTTS()
    assistant = _assistant(FakeSTT("qué hora es"), tts, wake_word=None)

    reply = await assistant.run_once()

    assert reply == "ARIA dice: qué hora es"


@pytest.mark.asyncio
async def test_run_once_exit_raises():
    tts = FakeTTS()
    assistant = _assistant(FakeSTT("aria hola"), tts)
    await assistant.run_once()

    with pytest.raises(ExitConversation):
        await assistant.run_once(text="salir")


@pytest.mark.asyncio
async def test_run_once_stt_failure_returns_none():
    class BrokenSTT(STTProvider):
        name = "broken"

        async def listen(self, *, language=None) -> str:
            raise VoiceError("sin micrófono")

    reply = await _assistant(BrokenSTT(), FakeTTS()).run_once()

    assert reply is None


@pytest.mark.asyncio
async def test_run_loop_activate_deactivate_and_exit():
    tts = FakeTTS()
    assistant = _assistant(
        FakeSTT(["aria hola", "aria ya acabamos gracias", "aria hola", "salir"]), tts
    )

    await assistant.run_loop()

    assert tts.spoken == [
        "¡Hola! Soy A.R.I.A, ¿en qué puedo ayudarle, jefe?",
        "¡Hasta luego! Estaré en espera.",
        "¡Hola! Soy A.R.I.A, ¿en qué puedo ayudarle, jefe?",
    ]

# --- Piper TTS local ---


class _FakePiperResult:
    returncode = 0
    stderr = b""


async def test_piper_synthesize_creates_wav(tmp_path, monkeypatch):
    model = tmp_path / "modelo.onnx"
    model.write_bytes(b"pesos")
    captured = {}

    def fake_run(cmd, input=None, capture_output=True, timeout=30, check=False):
        captured["cmd"] = cmd
        captured["input"] = input
        out = Path(cmd[cmd.index("--output-file") + 1])
        out.write_bytes(b"WAVDATA")
        return _FakePiperResult()

    monkeypatch.setattr("app.voice.tts.subprocess.run", fake_run)
    monkeypatch.setattr("app.voice.tts.shutil.which", lambda name: "/usr/bin/piper")

    provider = PiperTTSProvider(model)
    path = await provider.synthesize("Hola", tmp_path / "out.wav")

    assert path.read_bytes() == b"WAVDATA"
    assert provider.name == "piper_tts"
    assert "--model" in captured["cmd"]
    assert str(model) in captured["cmd"]
    assert captured["input"] == b"Hola"


async def test_piper_missing_model_raises(tmp_path):
    provider = PiperTTSProvider(tmp_path / "noexiste.onnx")

    with pytest.raises(VoiceError):
        await provider.synthesize("Hola", tmp_path / "out.wav")


def test_build_tts_provider_prefers_piper_when_available(tmp_path, monkeypatch):
    from app.core.config import Settings

    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")
    monkeypatch.setattr("app.voice.tts.shutil.which", lambda name: "/usr/bin/piper")

    settings = Settings(tts_provider="piper", piper_model=model)
    tts = build_tts_provider(settings)

    assert isinstance(tts, PiperTTSProvider)
    assert tts.audio_ext == ".wav"


def test_build_tts_provider_auto_uses_natural_edge_voice(tmp_path, monkeypatch):
    """auto = Edge (neuronal): gtts suena robótico y ya no compite por defecto."""
    from app.core.config import Settings

    monkeypatch.setattr("app.voice.tts.shutil.which", lambda name: None)

    settings = Settings(tts_provider="auto", piper_model=tmp_path / "m.onnx")
    tts = build_tts_provider(settings)

    assert isinstance(tts, EdgeTTSProvider)
    assert tts.voice == settings.tts_voice


def test_build_tts_provider_edge_forced(tmp_path):
    from app.core.config import Settings

    settings = Settings(tts_provider="edge", piper_model=tmp_path / "m.onnx")
    tts = build_tts_provider(settings)

    assert isinstance(tts, EdgeTTSProvider)


# --- Google Translate TTS + carrera de proveedores ---


class _FakeResponse:
    def __init__(self, content: bytes = b"MP3", status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code


async def test_gtts_synthesize_writes_mp3(tmp_path, monkeypatch):
    from app.voice.tts import GoogleTranslateTTSProvider

    captured = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["params"] = params
        return _FakeResponse(b"GTTSDATA")

    monkeypatch.setattr("httpx.get", fake_get)

    provider = GoogleTranslateTTSProvider()
    path = await provider.synthesize("Hola", tmp_path / "out.mp3")

    assert path.read_bytes() == b"GTTSDATA"
    assert provider.name == "gtts"
    assert captured["params"]["q"] == "Hola"
    assert captured["params"]["tl"] == "es"


async def test_gtts_splits_long_text_into_requests(tmp_path, monkeypatch):
    from app.voice.tts import GoogleTranslateTTSProvider

    peticiones: list[str] = []

    def fake_get(url, params=None, headers=None, timeout=None):
        peticiones.append(params["q"])
        return _FakeResponse(b"X")

    monkeypatch.setattr("httpx.get", fake_get)

    texto_largo = "palabra " * 60  # 480 chars
    await GoogleTranslateTTSProvider().synthesize(texto_largo, tmp_path / "out.mp3")

    assert len(peticiones) >= 3
    assert all(len(p) <= 200 for p in peticiones)
    assert " ".join(peticiones).split() == texto_largo.split()


class _InstantTTS(TTSProvider):
    name = "instant"
    audio_ext = ".mp3"

    async def synthesize(self, text, output_path):
        from pathlib import Path

        Path(output_path).write_bytes(b"RAPIDO")
        return Path(output_path)

    async def speak(self, text, *, output_dir=None):
        raise AssertionError


class _SlowTTS(TTSProvider):
    name = "slow"
    audio_ext = ".mp3"

    async def synthesize(self, text, output_path):
        import asyncio
        from pathlib import Path

        await asyncio.sleep(30)
        Path(output_path).write_bytes(b"LENTO")
        return Path(output_path)

    async def speak(self, text, *, output_dir=None):
        raise AssertionError


class _BrokenTTS(TTSProvider):
    name = "broken"
    audio_ext = ".mp3"

    async def synthesize(self, text, output_path):
        raise VoiceError("fallo simulado")

    async def speak(self, text, *, output_dir=None):
        raise AssertionError


async def test_race_picks_fastest_provider(tmp_path):
    from app.voice.tts import RaceTTSProvider

    race = RaceTTSProvider(_SlowTTS(), _InstantTTS())
    path = await race.synthesize("Hola", tmp_path / "out.mp3")

    assert path.read_bytes() == b"RAPIDO"
    # el temporal del perdedor se limpia
    assert not list(tmp_path.glob("*_slow*"))


async def test_race_falls_back_when_first_fails(tmp_path):
    from app.voice.tts import RaceTTSProvider

    race = RaceTTSProvider(_BrokenTTS(), _InstantTTS())
    path = await race.synthesize("Hola", tmp_path / "out.mp3")

    assert path.read_bytes() == b"RAPIDO"


async def test_race_raises_when_all_fail(tmp_path):
    from app.voice.tts import RaceTTSProvider

    race = RaceTTSProvider(_BrokenTTS(), _BrokenTTS())

    with pytest.raises(VoiceError):
        await race.synthesize("Hola", tmp_path / "out.mp3")
