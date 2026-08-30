"""Reglas deterministas de detección de intención para A.R.I.A.

Complementan a la red neuronal: las preguntas tienen prioridad porque la
presencia de "?" es un indicador determinista que la red (entrenada con pocos
ejemplos) no siempre captura. Las órdenes explícitas también tienen prioridad
sobre saludos/deseos, para que un contenido de archivo tipo "hola" no haga que
una instrucción de creación se confunda con un saludo.
"""
import re

# Verbos que marcan una orden/acción (matching por palabra completa, no substring)
VERBOS_COMANDO = {
    "lista", "listar", "listame", "muestra", "mostrar",
    "ejecuta", "ejecutar", "corre", "correr", "instala", "instalar",
    "abre", "abrir", "abrime", "cierra", "cerrar",
    "lee", "leer", "leeme", "crea", "crear", "creame", "escribe", "escribir",
    "guarda", "guardar", "borra", "borrar", "elimina", "eliminar", "renombra",
    "copia", "mueve", "actualiza", "actualizar",
    "compila", "compilar", "depura", "depurar", "formatea", "formatear",
    "reinicia", "reiniciar", "pausa", "pausar",
}
# Frases de estado que son órdenes a consultar (hora, fecha, sistema…)
FRASES_ESTADO = [
    "qué hora", "hora es", "que hora",
    "qué fecha", "que fecha", "cuál es la fecha",
    "qué día", "que día", "qué dia", "que dia", "qué día es", "día es",
    "clima", "temperatura", "sistema", "versión de python", "versión de os",
]

_WORDS_RE = re.compile(r"[a-záéíóúñü]+")


def intent_label(message: str) -> str:
    """Etiqueta la intención de un mensaje. Preguntas y órdenes tienen prioridad."""
    msg = message.lower()
    if "?" in msg:
        return "PREGUNTA"

    # Órdenes/acciones explícitas primero: el usuario quiere que hagamos algo.
    words = set(_WORDS_RE.findall(msg))
    if words & VERBOS_COMANDO:
        return "COMANDO"
    if any(fra in msg for fra in FRASES_ESTADO):
        return "COMANDO"

    if any(w in msg for w in ["hola", "buenos", "buenas", "hey", "qué onda", "cómo estás", "qué tal"]):
        return "SALUDO"
    if any(w in msg for w in ["adiós", "hasta luego", "me voy", "bye", "chao", "nos vemos", "hasta pronto"]):
        return "DESPEDIDA"
    if any(w in msg for w in ["gracias", "thanks", "agradezco"]):
        return "AGRADECIMIENTO"
    if any(w in msg for w in ["no funciona", "no me funciona", "no sirve", "no arranca", "no enciende", "no responde", "error", "roto", "problema", "falla"]):
        return "QUEJA"
    if any(w in msg for w in ["cuéntame", "qué sabes", "explícame", "explica", "opinas", "cuéntame de", "cuéntame algo"]):
        return "CURIOSIDAD"
    return "CHAT"
