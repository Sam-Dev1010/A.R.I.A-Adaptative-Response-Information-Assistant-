"""Tests de música y control multimedia (desktop_tools extra)."""
import asyncio

from app.tools.desktop_tools import MediaControlTool, PlayMusicTool


class FakeProc:
    def __init__(self, stdout=b"", returncode=0):
        self._stdout = stdout
        self.returncode = returncode

    async def communicate(self):
        return (self._stdout, b"")


def _patch_playerctl(monkeypatch, cmd_log, stdout=b""):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/playerctl")

    async def fake_exec(*args, **kwargs):
        cmd_log.append(list(args))
        return FakeProc(stdout=stdout)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)


async def test_play_music_falls_back_to_browser(monkeypatch):
    opened = {}
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr("webbrowser.open", lambda url: opened.setdefault("url", url))

    result = await PlayMusicTool().execute(song="paranoid black sabbath")

    assert "Busqué" in result
    assert "youtube.com" in opened["url"]
    assert "paranoid" in opened["url"]


async def test_play_music_with_ytdlp_and_mpv(monkeypatch):
    calls = {}

    async def fake_exec(*args, **kwargs):
        calls.setdefault("procs", []).append([str(a) for a in args])
        if any("ytsearch1" in str(a) for a in args):
            return FakeProc(stdout=b"https://youtu.be/fake\n")
        return FakeProc()

    monkeypatch.setattr(
        "shutil.which",
        lambda name: f"/usr/bin/{name}" if name in ("yt-dlp", "mpv") else None,
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    result = await PlayMusicTool().execute(song="war pigs")

    assert result == "Reproduciendo 'war pigs'."
    ytdlp_cmd, mpv_cmd = calls["procs"]
    assert "ytsearch1:war pigs" in ytdlp_cmd
    assert mpv_cmd[-1] == "https://youtu.be/fake"


async def test_media_control_volume_clamped(monkeypatch):
    ran = []
    _patch_playerctl(monkeypatch, ran)

    result = await MediaControlTool().execute(action="volume", value=250)

    assert "100%" in result and "250%" not in result
    assert ran[0] == ["/usr/bin/playerctl", "volume", "1.00"]


async def test_media_control_without_playerctl(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)

    result = await MediaControlTool().execute(action="pause")

    assert "playerctl no está instalado" in result


async def test_media_control_next(monkeypatch):
    ran = []
    _patch_playerctl(monkeypatch, ran)

    result = await MediaControlTool().execute(action="next")

    assert "pista siguiente" in result
    assert ran[0] == ["/usr/bin/playerctl", "next"]


async def test_media_control_status(monkeypatch):
    ran = []
    _patch_playerctl(monkeypatch, ran, stdout=b"Playing\n")

    result = await MediaControlTool().execute(action="status")

    assert "Estado: Playing" in result
