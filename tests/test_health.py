"""Tests del endpoint /health."""
from fastapi.testclient import TestClient

from app.main import create_app


def test_health_returns_ok():
    client = TestClient(create_app())
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "SIA"
    assert "version" in body
    assert "environment" in body
    assert body["uptime_seconds"] >= 0
    assert "server_time" in body


def test_health_is_reachable_without_dependencies():
    """El endpoint no debe requerir configuraciones ni servicios externos."""
    client = TestClient(create_app())
    response = client.get("/health")

    assert response.status_code == 200
    assert set(response.json().keys()) == {
        "status",
        "service",
        "version",
        "environment",
        "uptime_seconds",
        "server_time",
        "data_dir",
    }
