"""Logging estructurado para SIA.

Reglas:
- No se registran nunca API keys, contraseñas, tokens ni secretos.
- Campos adicionales se pasan con ``extra={"campo": valor}`` y se serializan a JSON.

Uso:
    from app.core.logging import get_logger

    log = get_logger("sia.tools")
    log.info("Herramienta ejecutada", extra={"tool": "get_time", "duration_ms": 12})
"""
import json
import logging
import sys
from datetime import UTC, datetime

LOG_FORMAT = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"
_TIME_FORMAT = "%Y-%m-%d %H:%M:%S %z"
_CONFIGURED = False


class StructuredFormatter(logging.Formatter):
    """Formatter que añade campos extra en JSON sin romper el formato estándar."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        dt = datetime.fromtimestamp(record.created, tz=UTC).astimezone()
        return dt.strftime(datefmt or _TIME_FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        extra = getattr(record, "_structured_fields", None)
        if extra:
            message = f"{message} {json.dumps(extra, ensure_ascii=False, default=str)}"
        return message


class SafeLogger(logging.Logger):
    """Logger que captura los campos extra de cada llamada log."""

    def makeRecord(
        self,
        name: str,
        level: int,
        fn: str,
        lno: int,
        msg: object,
        args: tuple,
        exc_info: tuple | None,
        func: str | None = None,
        extra: dict | None = None,
        sinfo: str | None = None,
    ) -> logging.LogRecord:
        record = super().makeRecord(name, level, fn, lno, msg, args, exc_info, func, extra, sinfo)
        if extra:
            record._structured_fields = extra  # type: ignore[attr-defined]
        return record


def setup_logging(level: str = "INFO") -> None:
    """Configura el logging del proceso. Es idempotente: solo aplica la primera vez."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    logging.setLoggerClass(SafeLogger)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter(fmt=LOG_FORMAT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Obtiene un logger con soporte de campos estructurados."""
    logging.setLoggerClass(SafeLogger)
    return logging.getLogger(name)
