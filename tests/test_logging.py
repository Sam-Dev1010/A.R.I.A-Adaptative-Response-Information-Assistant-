"""Tests del logging estructurado."""
import logging

from app.core.logging import get_logger, setup_logging


def test_logger_includes_structured_fields(caplog):
    setup_logging("DEBUG")
    with caplog.at_level(logging.INFO):
        log = get_logger("sia.test")
        log.info("herramienta ejecutada", extra={"tool": "get_time", "duration_ms": 5})
    record = caplog.records[-1]
    assert record.message == "herramienta ejecutada"
    assert record._structured_fields == {"tool": "get_time", "duration_ms": 5}  # type: ignore[attr-defined]


def test_logger_works_without_extra_fields(caplog):
    setup_logging("DEBUG")
    with caplog.at_level(logging.INFO):
        get_logger("sia.test").info("mensaje simple")
    assert caplog.records[-1].message == "mensaje simple"


def test_setup_logging_is_idempotent():
    root = logging.getLogger()
    handlers_before = list(root.handlers)
    setup_logging("INFO")
    assert root.handlers == handlers_before
