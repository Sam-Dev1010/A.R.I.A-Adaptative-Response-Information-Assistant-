"""Tests de la configuración central."""
from app.core.config import Settings, get_settings


def test_defaults():
    settings = Settings()
    assert settings.app_name == "A.R.I.A"
    assert settings.environment == "development"
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.log_level == "INFO"


def test_env_override(monkeypatch):
    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    settings = Settings()
    assert settings.port == 9000
    assert settings.log_level == "DEBUG"


def test_is_production():
    assert not Settings().is_production
    assert Settings(environment="production").is_production


def test_get_settings_is_cached():
    assert get_settings() is get_settings()


def test_safe_dict_never_exposes_secrets():
    settings = Settings(api_key="sk-secreto", llm_token="abc", password="1234", host="x")
    safe = settings.safe_dict()
    values = " ".join(str(v) for v in safe.values())
    assert "sk-secreto" not in values
    assert "abc" not in values
    assert "1234" not in values
    assert safe["host"] == "x"
