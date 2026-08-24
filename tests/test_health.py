from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_health_check_reports_healthy_when_all_services_are_up() -> None:
    with (
        patch("src.main._check_database", AsyncMock(return_value=True)),
        patch("src.main._check_opensearch", AsyncMock(return_value=True)),
        patch("src.main._check_ollama", AsyncMock(return_value=True)),
    ):
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "environment": "development",
        "version": "0.1.0",
        "services": {
            "database": "healthy",
            "opensearch": "healthy",
            "ollama": "healthy",
        },
    }


def test_health_check_reports_degraded_when_a_service_is_down() -> None:
    with (
        patch("src.main._check_database", AsyncMock(return_value=True)),
        patch("src.main._check_opensearch", AsyncMock(return_value=False)),
        patch("src.main._check_ollama", AsyncMock(return_value=True)),
    ):
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["services"] == {
        "database": "healthy",
        "opensearch": "unhealthy",
        "ollama": "healthy",
    }
