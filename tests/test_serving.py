"""Tests for the FastAPI serving layer (schemas + endpoints)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class _DummyModel:
    """Minimal model: y = pH + density. For easy verification."""

    def predict(self, X):
        return (X["ore_pulp_ph"] + X["ore_pulp_density"]).to_numpy()


@pytest.fixture
def client(monkeypatch):
    """Spin up the FastAPI app with a dummy model preloaded for both targets."""
    from frothiq.serving import api

    # Preload both targets with the dummy model so requests succeed without MLflow.
    api.store._models = {
        "pct_iron_concentrate": _DummyModel(),
        "pct_silica_concentrate": _DummyModel(),
    }
    api.store._versions = {
        "pct_iron_concentrate": "test:dummy@v0",
        "pct_silica_concentrate": "test:dummy@v0",
    }
    return TestClient(api.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "pct_iron_concentrate" in body["models_loaded"]


def test_predict_single_row(client):
    body = {
        "rows": [
            {"features": {"ore_pulp_ph": 9.5, "ore_pulp_density": 1.7}},
        ],
    }
    r = client.post("/predict", json=body)
    assert r.status_code == 200
    payload = r.json()
    assert "predictions" in payload
    assert len(payload["predictions"]) == 1
    pred = payload["predictions"][0]
    # Dummy model: 9.5 + 1.7 = 11.2 for both targets.
    assert pred["pct_iron_concentrate"] == pytest.approx(11.2, rel=1e-6)
    assert pred["pct_silica_concentrate"] == pytest.approx(11.2, rel=1e-6)


def test_predict_with_target_filter(client):
    body = {
        "rows": [{"features": {"ore_pulp_ph": 1.0, "ore_pulp_density": 1.0}}],
        "target": "pct_iron_concentrate",
    }
    r = client.post("/predict", json=body)
    assert r.status_code == 200
    pred = r.json()["predictions"][0]
    assert "pct_iron_concentrate" in pred
    assert "pct_silica_concentrate" not in pred


def test_predict_empty_rows(client):
    r = client.post("/predict", json={"rows": []})
    assert r.status_code == 400


def test_whatif(client):
    body = {
        "current_features": {"ore_pulp_ph": 9.4, "ore_pulp_density": 1.7},
        "overrides": {"ore_pulp_ph": 9.8},
        "target": "pct_iron_concentrate",
    }
    r = client.post("/whatif", json=body)
    assert r.status_code == 200
    payload = r.json()
    # baseline = 9.4 + 1.7 = 11.1; counterfactual = 9.8 + 1.7 = 11.5; delta = 0.4.
    assert payload["baseline"] == pytest.approx(11.1, rel=1e-6)
    assert payload["counterfactual"] == pytest.approx(11.5, rel=1e-6)
    assert payload["delta"] == pytest.approx(0.4, rel=1e-6)
    assert payload["target"] == "pct_iron_concentrate"
