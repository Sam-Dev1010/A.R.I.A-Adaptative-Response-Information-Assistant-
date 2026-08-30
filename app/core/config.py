"""Configuración central de A.R.I.A.

Toda la configuración proviene de variables de entorno o del archivo ``.env``.
Nunca se deben almacenar API keys u otros secretos en el código.
"""
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Campos que nunca deben exponerse en logs o respuestas.
_SECRET_MARKERS = ("key", "token", "secret", "password", "passwd")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "A.R.I.A"
    app_version: str = "0.1.0"
    environment: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    data_dir: Path = Path("data")

    # LLM (FASE 2)
    llm_provider: str = "openai_compatible"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""  # secreto: nunca se loguea ni se expone
    llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: float = 60.0
    llm_system_prompt: str = ""  # vacío = prompt por defecto del orquestador
    llm_max_tokens: int = 256  # respuestas cortas = más rápidas

    # LLM respaldo (failover): si el principal falla, se usa este (FASE 10)
    llm_fallback_provider: str = ""  # vacío = sin respaldo
    llm_fallback_base_url: str = ""
    llm_fallback_api_key: str = ""  # secreto
    llm_fallback_model: str = ""

    # Esfuerzo de razonamiento (none|low|medium|high) para gpt-oss/qwen3:
    # menos razonamiento = respuestas mucho más rápidas.
    llm_reasoning_effort: str = ""
    llm_fallback_reasoning_effort: str = ""

    # Herramientas (FASE 3)
    tools_enabled: str = ""  # comas; vacío = todas habilitadas
    tools_auto_confirm: str = ""  # comas; CONFIRM que no piden confirmación

    # Memoria (FASE 4)
    memory_enabled: bool = True  # SQLite persistente en DATA_DIR/sia.db

    # Identidad y autoaprendizaje
    # A.R.I.A sabe quién la creó. Acepta ARIA_CREATOR_NAME (nuevo) o el
    # histórico SIA_CREATOR_NAME para no romper .env ya configurados.
    aria_creator_name: str = Field(
        default="Samuel",
        validation_alias=AliasChoices("ARIA_CREATOR_NAME", "SIA_CREATOR_NAME"),
    )
    auto_learn_enabled: bool = True  # extrae hechos del usuario tras cada charla
    auto_curiosity_enabled: bool = True  # investiga sola temas nuevos en la web

    # Charla espontánea: SIA toma la iniciativa y habla sin que le pregunten
    proactive_enabled: bool = True
    proactive_min_minutes: int = 8   # espera mínima entre comentarios (aleatoria)
    proactive_max_minutes: int = 25  # espera máxima; se sortea entre min y max
    proactive_max_per_hour: int = 3  # presupuesto por hora para no ser pesada
    proactive_quiet_start: int = 23  # hora local: empieza el silencio nocturno
    proactive_quiet_end: int = 8     # hora local: vuelve a poder hablar

    # Voz (FASES 5-7)
    stt_language: str = "es-ES"
    stt_provider: str = "google"  # google | groq (whisper-turbo, ~3x más rápido)
    stt_groq_base_url: str = "https://api.groq.com/openai/v1"
    stt_groq_model: str = "whisper-large-v3-turbo"
    stt_groq_api_key: str = ""  # secreto; vacío = reutiliza LLM_FALLBACK_API_KEY si es Groq
    tts_provider: str = "auto"  # auto | piper | edge
    piper_model: Path = Path("data/piper/es_MX-ald-medium.onnx")
    tts_voice: str = "es-MX-DaliaNeural"
    tts_rate: str = "+8%"  # ritmo ligeramente vivo, sin sonar apresurada
    tts_pitch: str = "+0Hz"  # sin cambios de tono: el pitch alto suena robótico
    wake_word: str = "aria"  # vacío = responder sin palabra de activación

    # Identificación de hablante (quién le habla): 100 % local (speakeronnx)
    speaker_id_enabled: bool = True          # deshabilita el análisis de voz
    speaker_id_threshold: float = 0.55       # similitud mínima para dar por identificado
    speaker_id_dir: Path = Path("data/speakers")  # donde guarda las huellas de voz
    speaker_default_authority: str = ""      # vacío = se usará aria_creator_name

    # Acceso remoto: si se define, los WebSockets exigen ?token=… (fuera de casa)
    access_token: str = ""

    # Motor Neural (red neuronal propia)
    neural_enabled: bool = True  # habilita el motor neural de razonamiento
    neural_data_dir: Path = Path("data/neural")  # directorio del modelo entrenado
    neural_auto_train: bool = True  # entrena automáticamente al iniciar si hay datos
    neural_use_llm_fallback: bool = True  # usar LLM externo si el neural no puede

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    def safe_dict(self) -> dict:
        """Dict de configuración apto para logs: excluye cualquier valor secreto."""
        return {
            key: value
            for key, value in self.model_dump().items()
            if not any(marker in key.lower() for marker in _SECRET_MARKERS)
        }


@lru_cache
def get_settings() -> Settings:
    """Devuelve la configuración única del proceso (cargada una sola vez)."""
    return Settings()
