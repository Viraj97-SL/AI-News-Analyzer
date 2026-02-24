"""Unit tests for FastAPI endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app

settings = get_settings()


@pytest.fixture
def client():
    """Test client with API key auth header."""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    return {"X-API-Key": settings.api_key}


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/healthz/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_health_includes_environment(self, client):
        resp = client.get("/healthz/")
        assert "environment" in resp.json()


class TestRootEndpoint:
    def test_root_returns_service_info(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json()["service"] == "AI News Summarizer"


class TestRunsEndpoint:
    def test_trigger_requires_api_key(self, client):
        resp = client.post("/api/v1/runs/trigger")
        assert resp.status_code == 403

    def test_trigger_with_valid_key_returns_run_id(self, client, auth_headers):
        resp = client.post("/api/v1/runs/trigger", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "run_id" in data
        assert data["status"] == "started"

    def test_get_unknown_run_returns_404(self, client, auth_headers):
        resp = client.get("/api/v1/runs/nonexistent-id", headers=auth_headers)
        assert resp.status_code == 404
