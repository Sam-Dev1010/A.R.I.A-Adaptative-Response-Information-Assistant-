# SIA — backend en contenedor (multi-stage para una imagen ligera)
# Construir:  docker build -t sia-backend .
# Correr:     docker run -d --name sia -p 8000:8000 --env-file .env -v ./data:/app/data sia-backend
# O más fácil: docker compose up -d --build

# --- Etapa 1: compilar dependencias ---------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt requirements-server.txt ./
RUN pip install -r requirements-server.txt

# --- Etapa 2: imagen final mínima ------------------------------------------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# ffmpeg: decodifica el audio del navegador (webm/opus → PCM) para el STT.
# tzdata: respeta TZ para la charla espontánea (silencio nocturno por horas locales).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 sia

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY app ./app
RUN mkdir -p /app/data && chown -R sia:sia /app/data

USER sia

VOLUME ["/app/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; r=urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4); sys.exit(0 if r.status==200 else 1)" || exit 1

# HOST/PORT se leen del entorno (.env vía compose); un solo worker porque SIA
# guarda estado en memoria (conexiones de voz, celular conectado, charla espontánea).
CMD ["sh", "-c", "exec uvicorn app.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000}"]
