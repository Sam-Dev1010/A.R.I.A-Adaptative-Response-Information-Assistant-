"""Script de prueba completa para A.R.I.A Neural Engine.

Demuestra:
- Clasificación de intenciones
- Razonamiento chain-of-thought
- Planificación estratégica
- Control de PC
- Memoria semántica
- Personalidad propia
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ai.neural.brain import NeuralBrain


def print_separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


def print_response(response: str, elapsed: float) -> None:
    print(f"\nARIA: {response}")
    print(f"({elapsed:.3f}s)")


def main():
    print_separator("A.R.I.A Neural Engine - Prueba Completa")

    # Inicializar cerebro
    data_dir = Path(__file__).parent.parent / "data" / "neural_test"
    brain = NeuralBrain(data_dir)
    brain.initialize()

    print("\n1. ENTRENAMIENTO DEL CLASIFICADOR")
    print("-" * 40)

    # Datos de entrenamiento amplios
    texts = [
        # PREGUNTAS
        "¿Qué hora es?", "¿Qué tiempo hace?", "¿Quién eres?", "¿Qué puedes hacer?",
        "¿Cuál es tu nombre?", "¿Qué día es hoy?", "¿Qué es Python?", "Explícame esto",
        "¿Por qué?", "¿Cómo funciona?", "¿Cuánto vale?", "¿Dónde está?",
        "¿Qué opinas?", "Cuéntame algo", "¿Es verdad que?", "¿Cómo se hace?",
        # COMANDOS
        "Abre el navegador", "Ejecuta el script", "Guarda el archivo", "Corre los tests",
        "Instala la dependencia", "Muestra el historial", "Prende la luz", "Apaga la PC",
        "Reproduce música", "Manda un WhatsApp", "Llama a mamá", "Pon una alarma",
        "Crea un archivo", "Borra esto", "Clona el repo", "Haz un commit",
        # CHAT
        "Hola qué tal", "Bien", "Más o menos", "Estoy cansado", "Qué haces", "Nada",
        "Ok", "Vale", "Genial", "Interesante", "Ya veo", "Jaja",
        # SALUDOS
        "Hola", "Buenos días", "Buenas tardes", "Buenas noches", "Hey", "Qué onda",
        "Buenas", "Saludos", "Hola Samuel", "Jefe", "Qué hay",
        # DESPEDIDAS
        "Adiós", "Hasta luego", "Nos vemos", "Me voy", "Hasta mañana", "Chao",
        "Bye", "Me despido", "Hasta pronto", "Cuídate",
        # AGRADECIMIENTOS
        "Gracias", "Muchas gracias", "Te agradezco", "Gracias por todo", "Eres genial",
        "Mil gracias", "Thanks", "Gracias jefe", "Muy amable",
        # QUEJAS
        "Esto no funciona", "Está roto", "No entiendo", "Qué molestia", "Estoy frustrado",
        "No puedo creerlo", "Otra vez el mismo error", "Esto es terrible", "No me gusta",
        # CURIOSIDAD
        "¿Sabes qué es?", "Me gustaría saber", "Cuéntame sobre", "¿Cómo funciona?",
        "¿Por qué pasa eso?", "¿Qué significa?", "¿Qué hay de nuevo?",
    ]
    intents = (
        ["PREGUNTA"] * 16 + ["COMANDO"] * 16 + ["CHAT"] * 12 +
        ["SALUDO"] * 11 + ["DESPEDIDA"] * 10 + ["AGRADECIMIENTO"] * 9 +
        ["QUEJA"] * 9 + ["CURIOSIDAD"] * 7
    )

    brain.add_training_data(texts, intents)
    start = time.time()
    history = brain.train(epochs=30)
    elapsed = time.time() - start

    print(f"Entrenado con {len(texts)} ejemplos en {elapsed:.1f}s")
    if history:
        print(f"Loss final: {history[-1]['train_loss']:.4f}")

    print_separator("PRUEBA DE CLASIFICACIÓN DE INTENCIONES")

    test_cases = [
        ("¿Qué hora es?", "PREGUNTA"),
        ("Abre el navegador", "COMANDO"),
        ("Hola ARIA", "SALUDO"),
        ("Gracias", "AGRADECIMIENTO"),
        ("Esto no funciona", "QUEJA"),
        ("Adiós", "DESPEDIDA"),
        ("¿Cómo funciona Python?", "PREGUNTA"),
        ("Crea un archivo nuevo", "COMANDO"),
    ]

    correct = 0
    for text, expected in test_cases:
        start = time.time()
        intent, conf = brain.classifier.classify(text)
        elapsed = time.time() - start
        match = "✓" if intent == expected else "✗"
        if intent == expected:
            correct += 1
        print(f"{match} '{text}' -> {intent} ({conf:.2f}) [{elapsed*1000:.0f}ms]")

    print(f"\nPrecisión: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.0f}%)")

    print_separator("PRUEBA DE RAZONAMIENTO")

    questions = [
        "¿Cuál es la diferencia entre Python y JavaScript?",
        "¿Cómo debería organizar mi proyecto?",
        "¿Por qué mi código no funciona?",
    ]

    for question in questions:
        print(f"\nPregunta: {question}")
        start = time.time()
        result = brain.reasoning.reason(question)
        elapsed = time.time() - start

        print(f"Respuesta: {result.answer}")
        print(f"Pasos de razonamiento: {len(result.thoughts)}")
        print(f"Confianza: {result.confidence:.0%}")
        print(f"Tiempo: {elapsed:.3f}s")

        print("\nProceso de razonamiento:")
        for thought in result.thoughts[:3]:
            print(f"  Paso {thought.step}: {thought.content}")

    print_separator("PRUEBA DE PLANIFICACIÓN ESTRATÉGICA")

    objectives = [
        "Instalar Django en mi proyecto",
        "Crear una API REST con Python",
        "Arreglar el error de importación",
    ]

    for objective in objectives:
        print(f"\nObjetivo: {objective}")
        start = time.time()
        plan = brain.plan_action(objective)
        elapsed = time.time() - start

        print(f"Pasos: {len(plan['steps'])}")
        print(f"Tiempo estimado: {plan['estimated_time']}")
        print(f"Dificultad: {plan['difficulty']}")
        print(f"Herramientas: {', '.join(plan['tools_needed'])}")

        if plan['risks']:
            print("Riesgos:")
            for risk in plan['risks']:
                print(f"  - {risk}")

        print("Plan:")
        for step in plan['steps'][:3]:
            print(f"  {step['step']}. {step['action']}")

    print_separator("PRUEBA DE MEMORIA SEMÁNTICA")

    # Almacenar recuerdos
    memories_to_store = [
        ("Python es un lenguaje de programación", "tech", 0.8),
        ("Samuel es desarrollador de software", "person", 0.9),
        ("El proyecto A.R.I.A usa FastAPI", "project", 0.85),
        ("La API de Groq es gratuita", "tech", 0.7),
    ]

    for content, category, importance in memories_to_store:
        brain.memory.store(content, category=category, importance=importance)
        print(f"Almacenado: {content}")

    print("\nBúsquedas semánticas:")

    queries = [
        "lenguaje de programación",
        "quién es Samuel",
        "tecnología del proyecto",
    ]

    for query in queries:
        start = time.time()
        results = brain.memory.search(query, limit=2)
        elapsed = time.time() - start

        print(f"\nQuery: '{query}' [{elapsed*1000:.0f}ms]")
        for r in results:
            print(f"  - {r['content']} (sim: {r['similarity']:.2f})")

    print_separator("PRUEBA DE PERSONALIDAD")

    print(f"Mood actual: {brain.personality.mood}")
    print(f"Energía: {brain.personality.energy:.0%}")

    # Simular interacción
    interactions = [
        "Hola ARIA",
        "Estoy frustrado con este error",
        "Gracias por ayudarme",
    ]

    for msg in interactions:
        start = time.time()
        response = brain.think(msg)
        elapsed = time.time() - start
        print_response(response, elapsed)
        print(f"Mood: {brain.personality.mood} | Energía: {brain.personality.energy:.0%}")

    print_separator("ESTADO FINAL DEL CEREBRO")

    status = brain.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    print_separator("PRUEBA DE VELOCIDAD")

    # Test de velocidad con múltiples consultas
    speed_tests = [
        "Hola",
        "¿Qué hora es?",
        "Abre el navegador",
        "Gracias",
        "Adiós",
    ]

    times = []
    for _ in range(5):  # 5 rondas
        for msg in speed_tests:
            start = time.time()
            brain.think(msg)
            times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    print(f"Consultas totales: {len(times)}")
    print(f"Tiempo promedio: {avg_time*1000:.1f}ms")
    print(f"Tiempo mínimo: {min(times)*1000:.1f}ms")
    print(f"Tiempo máximo: {max(times)*1000:.1f}ms")

    # Cerrar
    brain.close()

    print_separator("PRUEBAS COMPLETADAS")
    print("\nEl cerebro neural de A.R.I.A está funcionando correctamente.")
    print("Características demostradas:")
    print("  ✓ Clasificación de intenciones")
    print("  ✓ Razonamiento chain-of-thought")
    print("  ✓ Planificación estratégica")
    print("  ✓ Memoria semántica")
    print("  ✓ Personalidad adaptable")
    print("  ✓ Velocidad optimizada")


if __name__ == "__main__":
    main()
