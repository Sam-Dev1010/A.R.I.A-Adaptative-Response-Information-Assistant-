"""Configuración compartida de pytest (conftest a nivel raíz).

Permite que los tests importen el paquete ``app`` sin instalación previa.
"""
import pytest

from app.core.config import get_settings


@pytest.fixture(autouse=True)
def _sin_token_del_env_real(monkeypatch):
    """Aísla los tests del ACCESS_TOKEN definido en el .env local.

    Los tests que prueban la autenticación fijan su propio token con
    ``monkeypatch.setattr(get_settings(), "access_token", ...)``.
    """
    monkeypatch.setattr(get_settings(), "access_token", "")
