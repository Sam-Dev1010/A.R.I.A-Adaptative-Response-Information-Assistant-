"""Script de demostración para probar el motor neural de A.R.I.A.

Muestra cómo funciona el clasificador de intenciones y la base de conocimiento.
"""
import sys
from pathlib import Path

# Añadir el directorio del proyecto al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ai.neural.brain import NeuralBrain
from app.ai.neural.intent_classifier import IntentClassifier


def main():
    """Demostración del motor neural."""
    print("=" * 60)
    print("A.R.I.A Neural Engine - Demostración")
    print("=" * 60)

    # Crear directorio de datos
    data_dir = Path(__file__).parent.parent / "data" / "neural_demo"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Inicializar cerebro
    brain = NeuralBrain(data_dir)
    brain.initialize()

    # Configurar un vocabulario más pequeño para la demo
    brain.classifier = IntentClassifier(vocab_size=500, hidden_dim=32)

    # Añadir algunos datos de entrenamiento
    print("\n1. Añadiendo datos de entrenamiento...")
    texts = [
        # PREGUNTAS
        '¿Qué hora es?', '¿Qué tiempo hace?', '¿Quién eres?', '¿Qué puedes hacer?',
        '¿Cuál es tu nombre?', '¿Qué día es hoy?', '¿Qué es Python?', 'Explícame esto',
        '¿Por qué?', '¿Cómo funciona?', '¿Cuánto vale?', '¿Dónde está?',
        # COMANDOS
        'Abre el navegador', 'Ejecuta el script', 'Guarda el archivo', 'Corre los tests',
        'Instala la dependencia', 'Muestra el historial', 'Prende la luz', 'Apaga la PC',
        'Reproduce música', 'Manda un WhatsApp', 'Llama a mamá', 'Pon una alarma',
        # CHAT
        'Hola qué tal', 'Bien', 'Más o menos', 'Estoy cansado', 'Qué haces', 'Nada',
        'Ok', 'Vale', 'Genial', 'Interesante', 'Ya veo', 'Jaja',
        # SALUDOS
        'Hola', 'Buenos días', 'Buenas tardes', 'Buenas noches', 'Hey', 'Qué onda',
        'Buenas', 'Saludos', 'Hola Samuel', 'Jefe', 'Qué hay',
        # DESPEDIDAS
        'Adiós', 'Hasta luego', 'Nos vemos', 'Me voy', 'Hasta mañana', 'Chao',
        'Bye', 'Me despido', 'Hasta pronto', 'Cuídate',
        # AGRADECIMIENTOS
        'Gracias', 'Muchas gracias', 'Te agradezco', 'Gracias por todo', 'Eres genial',
        'Mil gracias', 'Thanks', 'Gracias jefe', 'Muy amable',
        # QUEJAS
        'Esto no funciona', 'Está roto', 'No entiendo', 'Qué molestia', 'Estoy frustrado',
        'No puedo creerlo', 'Otra vez el mismo error', 'Esto es terrible', 'No me gusta',
        # CURIOSIDAD
        '¿Sabes qué es?', 'Me gustaría saber', 'Cuéntame sobre', '¿Cómo funciona?',
        '¿Por qué pasa eso?', '¿Qué significa?', '¿Qué hay de nuevo?',
    ]
    intents = (
        ['PREGUNTA'] * 12 + ['COMANDO'] * 12 + ['CHAT'] * 12 +
        ['SALUDO'] * 11 + ['DESPEDIDA'] * 10 + ['AGRADECIMIENTO'] * 9 +
        ['QUEJA'] * 9 + ['CURIOSIDAD'] * 7
    )
    brain.add_training_data(texts, intents)

    # Entrenar
    print("2. Entrenando clasificador...")
    history = brain.train(epochs=50)
    if history:
        print(f"   Loss final: {history[-1]['train_loss']:.4f}")

    # Probar el clasificador
    print("\n3. Probando clasificador:")
    test_cases = [
        "¿Qué hora es?",
        "Abre el navegador",
        "Hola ARIA",
        "Gracias",
        "Esto no funciona",
        "Cuéntame sobre inteligencia artificial",
        "Buenos días",
        "Adiós",
        "¿Qué es Python?",
        "Ejecuta el script",
    ]

    for text in test_cases:
        intent, confidence = brain.classifier.classify(text)
        print(f"   '{text}' -> {intent} ({confidence:.2f})")

    # Añadir conocimiento
    print("\n4. Añadiendo conocimiento...")
    brain.learn("Python es un lenguaje de programación", category="tecnologia")
    brain.learn_entity("Samuel", "persona", {"rol": "desarrollador", "creador": "ARIA"})
    brain.learn_relation("Samuel", "creó", "ARIA")

    # Buscar conocimiento
    print("\n5. Buscando conocimiento:")
    facts = brain.search_knowledge("Python")
    for fact in facts:
        print(f"   - {fact['content']}")

    # Inferir sobre una entidad
    print("\n6. Inferencia sobre Samuel:")
    inferences = brain.infer_about("Samuel")
    for inf in inferences:
        print(f"   - {inf}")

    # Estadísticas
    print("\n7. Estadísticas de la base de conocimiento:")
    stats = brain.get_knowledge_stats()
    print(f"   - Hechos: {stats['facts']}")
    print(f"   - Entidades: {stats['entities']}")
    print(f"   - Relaciones: {stats['relations']}")

    # Probar think
    print("\n8. Probando think (generación de respuestas):")
    responses = [
        "Hola",
        "¿Qué hora es?",
        "Gracias",
        "Adiós",
    ]
    for text in responses:
        response = brain.think(text)
        print(f"   '{text}' -> {response}")

    brain.close()
    print("\n" + "=" * 60)
    print("¡Demostración completada!")
    print("=" * 60)


if __name__ == "__main__":
    main()
