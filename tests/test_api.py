from fastapi.testclient import TestClient
from ml_ops_assignment.api import app
import pytest


def test_read_root():
    """Test health check endpoint."""
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "OK", "status-code": 200}


def test_predict_endpoint_mock():
    """Test prediction endpoint (mock mode if model missing)."""
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "This is a test sentence."})
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "label" in data
        assert data["class_name"] in ["Elementary", "Intermediate", "Advance"]
        assert data["status_code"] == 200
        assert data["text"] == "This is a test sentence."


@pytest.mark.parametrize(
    "text", ["I love this MLops course!", "This is absolutely terrible.", "", "This is a repeat sentence. " * 120]
)
def test_predict_various_inputs(text):
    """Test prediction endpoint with various text inputs."""
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": text})
        assert response.status_code == 200
        assert response.json()["text"] == text


def test_predict_invalid_input():
    """Test prediction endpoint with invalid JSON."""
    with TestClient(app) as client:
        # Missing 'text' key
        response = client.post("/predict", json={"wrong_key": "some text"})
        assert response.status_code == 422  # Unprocessable Entity
