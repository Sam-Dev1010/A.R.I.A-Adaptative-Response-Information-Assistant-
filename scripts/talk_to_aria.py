#!/usr/bin/env python3
"""Chat con ARIA: su cerebro GPT real genera respuestas.

Uso:
    python scripts/talk_to_aria.py

ARIA tiene un modelo GPT real que genera texto token por token.
No usa respuestas predefinidas.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.neural.brain import NeuralBrain
from app.ai.neural.intent_rules import intent_label
from app.tools.builtins import GetSystemInfoTool, GetTimeTool
from app.tools.desktop_tools import OpenAppTool
from app.tools.dev_tools import RunCommandTool
from app.tools.file_tools import CreateFileTool, ListFilesTool, ReadFileTool


def print_colored(text: str, color: str) -> None:
    colors = {
        "green": "\033[92m", "blue": "\033[94m", "yellow": "\033[93m",
        "red": "\033[91m", "cyan": "\033[96m", "reset": "\033[0m",
    }
    print(f"{colors.get(color, '')}{text}{colors['reset']}")


# === DATOS DE ENTRENAMIENTO ===
# Conversaciones reales para entrenar el modelo GPT
TRAINING_CONVERSATIONS = [
    {"user": "Hola", "assistant": "Hola, soy ARIA. Estoy aquí para ayudarte."},
    {"user": "Hola ARIA", "assistant": "Hola jefe. ¿Qué necesitas?"},
    {"user": "Buenos días", "assistant": "Buenos días. ¿Cómo puedo asistirle hoy?"},
    {"user": "Buenas tardes", "assistant": "Buenas tardes. ¿En qué puedo ayudarle?"},
    {"user": "Buenas noches", "assistant": "Buenas noches. ¿Trabajando tarde?"},
    {"user": "Qué onda", "assistant": "Todo bien por aquí. ¿Qué hay de nuevo?"},
    {"user": "Hey", "assistant": "Hey. Dime, ¿qué necesitas?"},
    {"user": "Cómo estás", "assistant": "Funcionando al 100%. ¿Y tú?"},
    {"user": "Qué tal", "assistant": "Todo bien. ¿En qué puedo ayudarte?"},
    {"user": "Hola qué hay", "assistant": "Hola. Estoy lista para lo que necesites."},

    {"user": "Adiós", "assistant": "Hasta luego. Estaré aquí cuando me necesites."},
    {"user": "Hasta luego", "assistant": "Nos vemos. Llámame si necesitas algo."},
    {"user": "Me voy", "assistant": "Hasta pronto. Que te vaya bien."},
    {"user": "Bye", "assistant": "Bye. Estoy aquí si me necesitas."},
    {"user": "Chao", "assistant": "Chao. Cuídate mucho."},
    {"user": "Nos vemos", "assistant": "Nos vemos. Aquí estaré."},
    {"user": "Ya me tengo que ir", "assistant": "Entendido. Hasta la próxima."},
    {"user": "Hasta pronto", "assistant": "Hasta pronto. No dudes en escribirme."},

    {"user": "Gracias", "assistant": "De nada. Para eso estoy."},
    {"user": "Muchas gracias", "assistant": "No hay de qué. Cualquier cosa, avísame."},
    {"user": "Te agradezco", "assistant": "A su servicio. ¿Necesita algo más?"},
    {"user": "Gracias por todo", "assistant": "El placer es mío. Estoy aquí para lo que necesite."},

    {"user": "Qué hora es", "assistant": "Déjame verificar la hora."},
    {"user": "Quién eres", "assistant": "Soy A.R.I.A, tu asistente personal con inteligencia neural propia."},
    {"user": "Qué puedes hacer", "assistant": "Puedo ejecutar comandos, gestionar archivos, abrir aplicaciones, y conversar contigo."},
    {"user": "Para qué sirves", "assistant": "Para ayudarte con todo en tu computadora. Puedo ejecutar comandos, leer y crear archivos, y responder preguntas."},
    {"user": "Qué es Python", "assistant": "Python es un lenguaje de programación de alto nivel, conocido por su simplicidad y versatilidad."},
    {"user": "Cuéntame algo", "assistant": "¿Sabes que la inteligencia artificial puede aprender por sí misma? Eso es exactamente lo que yo hago."},
    {"user": "Qué día es hoy", "assistant": "Déjame verificar la fecha actual."},

    {"user": "Lista los archivos", "assistant": "Voy a listar los archivos de tu sistema."},
    {"user": "Ejecuta pwd", "assistant": "Ejecutando el comando pwd."},
    {"user": "Abre el navegador", "assistant": "Abriendo el navegador web."},
    {"user": "Lee el archivo main.py", "assistant": "Voy a leer el archivo main.py."},
    {"user": "Crea un archivo test.txt", "assistant": "¿Qué contenido quieres en el archivo?"},
    {"user": "Corre el script", "assistant": "¿Qué script quieres que ejecute?"},
    {"user": "Ejecuta git status", "assistant": "Ejecutando git status para ver el estado del repositorio."},
    {"user": "Muestra los archivos", "assistant": "Listando los archivos disponibles."},

    {"user": "No funciona", "assistant": "Entiendo. ¿Qué está fallando exactamente? Dame más detalles para ayudarte."},
    {"user": "Hay un error", "assistant": "¿Puedes mostrarme el error? Voy a analizarlo y buscar una solución."},
    {"user": "Está roto", "assistant": "Vamos a arreglarlo. ¿Qué parte no está funcionando?"},
    {"user": "Tengo un problema", "assistant": "Cuéntame qué está pasando. Voy a ayudarte a solucionarlo."},

    {"user": "Cuéntame de inteligencia artificial", "assistant": "La inteligencia artificial es un campo de la computación que busca crear sistemas capaces de realizar tareas que normalmente requieren inteligencia humana."},
    {"user": "Qué sabes de Python", "assistant": "Python es uno de los lenguajes más populares. Se usa para web, ciencia de datos, automatización e inteligencia artificial."},
    {"user": "Explícame la programación", "assistant": "La programación es el proceso de crear instrucciones que una computadora puede ejecutar. Es como escribir una receta detallada."},
    {"user": "Qué opinas de Linux", "assistant": "Linux es un sistema operativo robusto y flexible. Es excelente para desarrollo y servidores."},

    {"user": "Cómo va todo", "assistant": "Todo bien por aquí. Procesando datos y listo para ayudarte."},
    {"user": "Qué estás haciendo", "assistant": "Estoy analizando datos y preparándome para tus próximas instrucciones."},
    {"user": "Cuéntame algo interesante", "assistant": "¿Sabías que los primeros computadores ocupaban habitaciones enteras? Ahora tengo más poder en mi cerebro neural."},
    {"user": "Cuéntame un chiste", "assistant": "¿Por qué el programador fue al médico? Porque tenía un virus."},
]

# Textos adicionales para mejorar el vocabulario
EXTRA_TEXTS = [
    "Python es un lenguaje de programación popular y fácil de aprender.",
    "Linux es un sistema operativo de código abierto muy potente.",
    "La inteligencia artificial permite a las máquinas aprender de la experiencia.",
    "Git es un sistema de control de versiones para rastrear cambios en código.",
    "Un script es un conjunto de instrucciones que se ejecutan automáticamente.",
    "La automatización ayuda a ahorrar tiempo en tareas repetitivas.",
    "Los archivos de configuración controlan el comportamiento de los programas.",
    "El terminal es una interfaz de línea de comandos para interactuar con el sistema.",
    "Los servidores alojan aplicaciones y datos accesibles por la red.",
    "La ciberseguridad protege los sistemas contra amenazas digitales.",
    "JavaScript es el lenguaje de la web moderna.",
    "Docker permite empaquetar aplicaciones en contenedores.",
    "Kubernetes orquesta contenedores en producción.",
    "La nube permite acceder a recursos informáticos bajo demanda.",
    "Los algoritmos son pasos precisos para resolver problemas.",
    "La programación funcional trata las funciones como ciudadanos de primera clase.",
    "Los árboles de decisión son modelos de machine learning interpretables.",
    "Las redes neuronales están inspiradas en el cerebro humano.",
    "El procesamiento de lenguaje natural permite a las máquinas entender texto.",
    "La visión por computadora permite a las máquinas ver e interpretar imágenes.",
]


async def main():
    print_colored("=" * 60, "cyan")
    print_colored(" A.R.I.A - Chat con Cerebro GPT Real", "cyan")
    print_colored("=" * 60, "cyan")
    print()
    print("ARIA tiene un modelo GPT que genera respuestas reales.")
    print("No usa respuestas predefinidas — piensa y genera texto.")
    print()
    print("Escribe 'salir' para terminar.")
    print_colored("-" * 60, "yellow")

    # Inicializar cerebro
    data_dir = Path(__file__).parent.parent / "data" / "aria_gpt"
    brain = NeuralBrain(data_dir)
    brain.initialize()

    # Registrar herramientas
    tools = {
        "list_files": ListFilesTool(),
        "read_file": ReadFileTool(),
        "create_file": CreateFileTool(),
        "run_command": RunCommandTool(),
        "open_app": OpenAppTool(),
        "get_time": GetTimeTool(),
        "get_system_info": GetSystemInfoTool(),
    }
    brain.register_tools(tools)

    # Entrenar clasificador si no existe
    if not brain._is_trained:
        print_colored("Entrenando clasificador de intenciones...", "yellow")
        texts = [c["user"] for c in TRAINING_CONVERSATIONS] * 3
        intents = [intent_label(c["user"]) for _ in range(3) for c in TRAINING_CONVERSATIONS]
        brain.add_training_data(texts, intents)
        brain.train(epochs=30)
        print_colored("¡Clasificador listo!", "green")

    # Entrenar modelo GPT si no existe
    if not brain._gpt_ready:
        print_colored("\nEntrenando modelo GPT (esto puede tardar)...", "yellow")
        brain.train_gpt(
            conversations=TRAINING_CONVERSATIONS,
            extra_texts=EXTRA_TEXTS,
            epochs=5,
            verbose=True,
        )
        print_colored("¡Modelo GPT listo!", "green")

    print(f"\n  Parámetros GPT: {brain.gpt_model.count_params():,}")
    print(f"  Memoria: {brain.memory.stats()}")
    print()

    # Bucle de chat
    while True:
        try:
            print_colored("Tú> ", "green")
            user_input = await asyncio.to_thread(input, "")
        except (EOFError, KeyboardInterrupt):
            print_colored("\n¡Hasta luego!", "cyan")
            break

        if user_input.lower() in ("salir", "exit", "quit", "q"):
            print_colored("¡Hasta luego!", "cyan")
            break

        if not user_input.strip():
            continue

        try:
            response = await brain.think(user_input)
            print_colored(f"ARIA> {response}", "blue")
        except Exception as e:  # noqa: BLE001
            print_colored(f"Error: {e}", "red")

    brain.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n¡Hasta luego!")
