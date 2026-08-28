"""Tests del endpoint HTTP de eventos del nodo ESP32 (/api/esp32/event)."""
import pytest
from fastapi.testclient import TestClient

from app.api.esp32 import (
    esp32_handlers,
    register_esp32_handler,
    unregister_esp32_handler,
)
from app.main import create_app


@pytest.fixture
def cliente():
    return TestClient(create_app())


def test_evento_valido_responde_200(cliente):
    resp = cliente.post(
        "/api/esp32/event",
        json={"node_id": "esp32_ia_local", "prediction": 0.87},
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "node_id": "esp32_ia_local",
        "prediction": 0.87,
    }


def test_prediccion_fuera_de_rango_es_rechazada(cliente):
    resp = cliente.post(
        "/api/esp32/event",
        json={"node_id": "esp32_ia_local", "prediction": 1.5},
    )
    assert resp.status_code == 422


def test_payload_incompleto_es_rechazado(cliente):
    assert (
        cliente.post("/api/esp32/event", json={}).status_code == 422
    )
    assert (
        cliente.post(
            "/api/esp32/event", json={"prediction": 0.5}
        ).status_code
        == 422
    )


def test_hooks_por_defecto_no_rompen_sin_orquestador(cliente):
    """Sin orquestador cargado, los hooks de memoria/voz son no-op."""
    resp = cliente.post(
        "/api/esp32/event",
        json={"node_id": "esp32", "prediction": 0.9},
    )
    assert resp.status_code == 200


def test_hook_registrado_es_invocado():
    app = create_app()
    eventos = []

    async def capturar(node_id, prediction):
        eventos.append((node_id, prediction))

    register_esp32_handler(app, capturar)
    TestClient(app).post(
        "/api/esp32/event", json={"node_id": "n1", "prediction": 0.5}
    )
    assert eventos == [("n1", 0.5)]
    unregister_esp32_handler(app, capturar)


def test_hooks_son_aislados_por_aplicacion():
    app_a = create_app()
    app_b = create_app()
    eventos = []

    async def capturar(node_id, prediction):
        eventos.append(node_id)

    register_esp32_handler(app_a, capturar)

    TestClient(app_a).post(
        "/api/esp32/event", json={"node_id": "a", "prediction": 0.1}
    )
    TestClient(app_b).post(
        "/api/esp32/event", json={"node_id": "b", "prediction": 0.1}
    )
    TestClient(app_a).post(
        "/api/esp32/event", json={"node_id": "a2", "prediction": 0.1}
    )

    assert eventos == ["a", "a2"]
    assert capturar in esp32_handlers(app_a)
    assert capturar not in esp32_handlers(app_b)


def test_hook_que_falla_no_rompe_la_ruta_ni_a_los_demas():
    app = create_app()
    eventos = []

    async def malo(node_id, prediction):
        raise RuntimeError("boom")

    async def bueno(node_id, prediction):
        eventos.append(node_id)

    register_esp32_handler(app, malo)
    register_esp32_handler(app, bueno)
    resp = TestClient(app).post(
        "/api/esp32/event", json={"node_id": "n", "prediction": 0.2}
    )
    assert resp.status_code == 200
    assert eventos == ["n"]