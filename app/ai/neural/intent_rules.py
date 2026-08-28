"""Reglas deterministas de detección de intención para A.R.I.A.

Complementan a la red neuronal: las preguntas tienen prioridad porque la
presencia de "?" es un indicador determinista que la red (entrenada con pocos
ejemplos) no siempre captura.
"""


def intent_label(message: str) -> str:
    """Etiqueta la intención de un mensaje. Las preguntas tienen prioridad."""
    msg = message.lower()
    if "?" in msg:
        return "PREGUNTA"
    if any(w in msg for w in ["hola", "buenos", "buenas", "hey", "qué onda", "cómo estás", "qué tal"]):
        return "SALUDO"
    if any(w in msg for w in ["adiós", "hasta luego", "me voy", "bye", "chao", "nos vemos", "hasta pronto"]):
        return "DESPEDIDA"
    if any(w in msg for w in ["gracias", "thanks", "agradezco"]):
        return "AGRADECIMIENTO"
    if any(w in msg for w in ["lista", "ejecuta", "abre", "corre", "lee", "crea", "muestra", "instala", "guarda"]):
        return "COMANDO"
    if any(w in msg for w in ["no funciona", "no me funciona", "no sirve", "no arranca", "no enciende", "no responde", "error", "roto", "problema", "falla"]):
        return "QUEJA"
    if any(w in msg for w in ["cuéntame", "qué sabes", "explícame", "explica", "opinas", "cuéntame de", "cuéntame algo"]):
        return "CURIOSIDAD"
    return "CHAT"