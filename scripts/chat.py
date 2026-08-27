#!/usr/bin/env python3
"""REPL de chat con SIA (FASES 2-4): chat + herramientas + memoria persistente.

Uso:
    source .venv/bin/activate
    python scripts/chat.py [--reset-memory]

Opciones:
    --reset-memory  Borra toda la memoria persistente de SIA antes de empezar.
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.factory import build_memory, build_orchestrator
from app.ai.providers.base import LLMError
from app.core.config import get_settings
from app.core.logging import setup_logging

_EXIT_COMMANDS = {"salir", "exit", "quit", "q"}
_YES = {"s", "si", "sí", "y", "yes"}


async def confirm_tool(call) -> bool:
    """Pide confirmación al usuario antes de ejecutar una herramienta."""
    args = ", ".join(f"{k}={v!r}" for k, v in call.arguments.items())
    answer = input(f"¿Permitir {call.name}({args})? (s/N) ").strip().lower()
    return answer in _YES


async def main() -> None:
    parser = argparse.ArgumentParser(description="Chat con SIA")
    parser.add_argument("--reset-memory", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)

    memory = build_memory(settings) if settings.memory_enabled else None
    if memory is not None:
        if args.reset_memory:
            memory.clear()
        memory.start_session("chat cli")
        print(f"Memoria: {memory.stats()}")

    orchestrator = build_orchestrator(settings, confirm=confirm_tool, memory=memory)

    if not settings.llm_api_key:
        print("ADVERTENCIA: LLM_API_KEY vacía (configura .env si el proveedor la exige).")

    print("SIA listo. Escribe 'salir' para terminar.")
    while True:
        try:
            user_input = input("Tú> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n¡Hasta luego!")
            break

        if user_input.lower() in _EXIT_COMMANDS:
            break
        if not user_input:
            continue

        try:
            response = await orchestrator.ask(user_input)
        except LLMError as exc:
            print(f"¡Error! {exc}")
            continue
        print(f"SIA> {response.content}")


if __name__ == "__main__":
    asyncio.run(main())