"""Construcción de los componentes de IA y herramientas a partir de Settings."""
from app.ai.auto_curiosity import CuriosityEngine
from app.ai.neural.brain import NeuralBrain
from app.ai.orchestrator import AssistantOrchestrator
from app.ai.personality import build_personality_prompt
from app.ai.providers.base import LLMProvider
from app.ai.providers.fallback import FallbackProvider
from app.ai.providers.openai_compatible import OpenAICompatibleProvider
from app.ai.self_learner import SelfLearner
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.memory.manager import MemoryManager
from app.tools.builtins import BUILTIN_TOOLS
from app.tools.desktop_tools import (
    MediaControlTool,
    OpenAppTool,
    OpenFolderTool,
    PlayMusicTool,
)
from app.tools.dev_tools import RunCommandTool, SystemUpdateTool
from app.tools.file_tools import (
    CreateFileTool,
    CreateFolderTool,
    DeletePathTool,
    ListFilesTool,
    ReadFileTool,
)
from app.tools.memory_tools import ForgetFactTool, RememberFactTool
from app.tools.network_tools import GetWeatherTool, SearchWikipediaTool, WebSearchTool
from app.tools.phone_tools import (
    NavigateTool,
    NotifyPhoneTool,
    OpenPhoneAppTool,
    PhoneCallTool,
    PhoneClipboardTool,
    PhoneContactsTool,
    PhoneStatusTool,
    PhoneTorchTool,
    PhoneVibrateTool,
    PhoneVolumeTool,
    SendEmailTool,
    SendSmsTool,
    SetAlarmTool,
    WhatsAppTool,
)
from app.tools.policy import ToolPolicy
from app.tools.presence_tools import GetPresenceTool
from app.tools.registry import ToolRegistry
from app.tools.study_tool import DeepStudyTool

logger = get_logger("sia.ai")

_PROVIDERS: dict[str, type[LLMProvider]] = {
    "openai_compatible": OpenAICompatibleProvider,
}


def _parse_names(value: str) -> set[str] | None:
    """Convierte una lista separada por comas en un set (None si está vacía)."""
    names = {name.strip() for name in value.split(",") if name.strip()}
    return names or None


def _build_single(
    settings: Settings,
    *,
    base_url: str,
    api_key: str,
    model: str,
    reasoning_effort: str = "",
) -> LLMProvider:
    provider_cls = _PROVIDERS.get(settings.llm_provider)
    if provider_cls is None:
        raise ValueError(
            f"Proveedor LLM desconocido: {settings.llm_provider!r}. "
            f"Disponibles: {sorted(_PROVIDERS)}"
        )
    extra_body = {"reasoning_effort": reasoning_effort} if reasoning_effort else None
    return provider_cls(
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout_seconds=settings.llm_timeout_seconds,
        max_tokens=settings.llm_max_tokens,
        extra_body=extra_body,
    )


def build_provider(settings: Settings | None = None) -> LLMProvider:
    """Instancia el proveedor LLM (con respaldo failover si está configurado)."""
    settings = settings or get_settings()
    primary = _build_single(
        settings,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        reasoning_effort=settings.llm_reasoning_effort,
    )

    if not settings.llm_fallback_provider:
        return primary

    fallback = _build_single(
        settings,
        base_url=settings.llm_fallback_base_url,
        api_key=settings.llm_fallback_api_key,
        model=settings.llm_fallback_model,
        reasoning_effort=settings.llm_fallback_reasoning_effort,
    )
    return FallbackProvider(primary, fallback)


def build_memory(settings: Settings | None = None) -> MemoryManager:
    """Abre la memoria persistente de SIA en DATA_DIR/sia.db."""
    settings = settings or get_settings()
    return MemoryManager(settings.data_dir / "sia.db").open()


def build_tool_registry(
    settings: Settings | None = None,
    *,
    memory: MemoryManager | None = None,
    search_tool: WebSearchTool | None = None,
) -> ToolRegistry:
    """Registra las herramientas integradas de SIA (+ memoria si aplica)."""
    registry = ToolRegistry()
    for tool_cls in BUILTIN_TOOLS:
        registry.register(tool_cls())
    registry.register(GetWeatherTool())
    registry.register(SearchWikipediaTool())
    search = search_tool or WebSearchTool()
    registry.register(search)
    registry.register(OpenAppTool())
    registry.register(OpenFolderTool())
    registry.register(PlayMusicTool())
    registry.register(MediaControlTool())
    registry.register(CreateFolderTool())
    registry.register(CreateFileTool())
    registry.register(ListFilesTool())
    registry.register(ReadFileTool())
    registry.register(DeletePathTool())
    registry.register(RunCommandTool())
    registry.register(SystemUpdateTool())
    registry.register(PhoneCallTool())
    registry.register(WhatsAppTool())
    registry.register(SendEmailTool())
    registry.register(OpenPhoneAppTool())
    registry.register(PhoneContactsTool())
    registry.register(PhoneStatusTool())
    registry.register(PhoneTorchTool())
    registry.register(PhoneVibrateTool())
    registry.register(PhoneClipboardTool())
    registry.register(SendSmsTool())
    registry.register(SetAlarmTool())
    registry.register(NavigateTool())
    registry.register(PhoneVolumeTool())
    registry.register(NotifyPhoneTool())
    registry.register(GetPresenceTool())
    if memory is not None:
        registry.register(RememberFactTool(memory))
        registry.register(ForgetFactTool(memory))
    return registry


def build_tool_policy(settings: Settings | None = None) -> ToolPolicy:
    """Construye la política de permisos desde la configuración."""
    settings = settings or get_settings()
    return ToolPolicy(
        allowed=_parse_names(settings.tools_enabled),
        auto_confirm=_parse_names(settings.tools_auto_confirm),
    )


def _register_brain_tools(
    brain: NeuralBrain,
    registry: ToolRegistry,
) -> None:
    """Exponer al cerebro neural las herramientas que puede ejecutar.

    ``NeuralBrain.think()`` ejecuta los comandos que detecta en el mensaje
    usando ``register_tools``. Sin esto, el cerebro del orquestador (y por
    tanto el flujo por voz) no podría listar, leer, crear ni ejecutar nada.
    """
    brain.register_tools(
        {
            tool.name: tool
            for tool in registry.all()
            if tool.name
            in {
                "list_files",
                "read_file",
                "create_file",
                "create_folder",
                "delete_path",
                "run_command",
                "open_app",
                "get_time",
                "get_system_info",
                "system_update",
            }
        }
    )


def build_neural_brain(settings: Settings | None = None) -> NeuralBrain | None:
    """Construye e inicializa el cerebro neural si está habilitado."""
    settings = settings or get_settings()
    if not settings.neural_enabled:
        return None

    brain = NeuralBrain(settings.neural_data_dir)
    brain.initialize()

    # Solo se entrena si aún no hay un modelo entrenado. De lo contrario el
    # arranque (p. ej. al encender el PC) se vuelve lentísimo re-entrenando.
    if settings.neural_auto_train and not brain.is_trained():
        # Entrenar si hay datos de entrenamiento
        training_file = settings.neural_data_dir / "training_data.json"
        if training_file.exists():
            brain.train(epochs=20)

    return brain


def build_orchestrator(
    settings: Settings | None = None,
    *,
    confirm=None,
    memory: MemoryManager | None = None,
    max_history_messages: int | None = None,
) -> AssistantOrchestrator:
    """Construye el orquestador completo: proveedor + herramientas + permisos + memoria."""
    settings = settings or get_settings()

    # Construir neural brain si está habilitado
    neural_brain = build_neural_brain(settings) if settings.neural_enabled else None

    # Si el neural brain está activo y no hay API key, usar solo neural (sin LLM)
    provider = None
    if neural_brain is None or settings.llm_api_key:
        provider = build_provider(settings)

    if memory is None and settings.memory_enabled:
        memory = build_memory(settings)
    search_tool = WebSearchTool()
    registry = build_tool_registry(settings, memory=memory, search_tool=search_tool)
    if memory is not None and provider is not None:
        registry.register(DeepStudyTool(provider, memory, search_tool.search))
    if neural_brain is not None:
        # El cerebro neural necesita sus propias herramientas para poder
        # ejecutar comandos cuando no hay LLM (p. ej. el flujo por voz).
        _register_brain_tools(neural_brain, registry)

    kwargs: dict = {
        "registry": registry,
        "policy": build_tool_policy(settings),
        "neural_brain": neural_brain,
    }
    if provider is not None:
        kwargs["provider"] = provider
    if max_history_messages is not None:
        kwargs["max_history_messages"] = max_history_messages
    if settings.llm_system_prompt:
        kwargs["system_prompt"] = settings.llm_system_prompt
    else:
        kwargs["system_prompt"] = build_personality_prompt(settings.aria_creator_name)
    if confirm is not None:
        kwargs["confirm"] = confirm
    if memory is not None:
        kwargs["memory"] = memory
        if provider is not None:
            aprendices = []
            if settings.auto_learn_enabled:
                aprendices.append(SelfLearner(provider, memory).learn)
            if settings.auto_curiosity_enabled:
                curiosidad = CuriosityEngine(provider, memory, search_tool.search)
                aprendices.append(curiosidad.research)
            if len(aprendices) == 1:
                kwargs["auto_learner"] = aprendices[0]
            elif aprendices:

                async def _aprender_todo(user_text: str, assistant_text: str) -> None:
                    for aprender in aprendices:
                        try:
                            await aprender(user_text, assistant_text)
                        except Exception as exc:  # noqa: BLE001 — aprender nunca rompe la charla
                            logger.debug("Aprendizaje falló (%s): %s", aprender, exc)

                kwargs["auto_learner"] = _aprender_todo
    return AssistantOrchestrator(**kwargs)


def build_speaker_manager(settings: Settings | None = None):
    """Construye el gestor de identificación de hablante (o None si está desactivado).

    El import es perezoso para no cargar onnxruntime/speakeronnx a menos que la
    identificación de voz esté realmente en uso.
    """
    settings = settings or get_settings()
    if not settings.speaker_id_enabled:
        return None
    from app.voice.speaker_id import SpeakerIdManager  # import perezoso

    return SpeakerIdManager(
        storage_dir=settings.speaker_id_dir,
        threshold=settings.speaker_id_threshold,
        default_authority=settings.speaker_default_authority or settings.aria_creator_name,
    )