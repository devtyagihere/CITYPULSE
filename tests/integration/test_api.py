"""Integration tests for FastAPI endpoints in api.py."""

from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "snapshot_available" in body


def test_dashboard_served():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "CITYPULSE" in resp.text


def test_refresh_validation_rejects_negative_lambda():
    resp = client.post("/api/refresh", json={"lambda_penalty": -1.0})
    assert resp.status_code == 422
