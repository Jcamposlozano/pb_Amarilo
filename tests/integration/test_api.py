import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from amarilo.infrastructure.bootstrap import create_app
from amarilo.infrastructure.settings import Settings


@pytest.fixture
def client(tmp_path):
    pipeline = make_pipeline(StandardScaler(), LinearRegression())
    pipeline.fit([[0, 0], [1, 0], [0, 1], [1, 1]], [0, 1, 2, 3])
    path = tmp_path / "model.joblib"
    joblib.dump(
        {"pipeline": pipeline, "model_version": "test-1", "feature_names": ["x1", "x2"]},
        path,
    )
    with TestClient(create_app(Settings(path))) as client:
        yield client


def test_inference_loads_once_and_returns_version(client, monkeypatch):
    def unexpected_load(*args, **kwargs):
        pytest.fail("El modelo no debe cargarse por petición")

    monkeypatch.setattr(joblib, "load", unexpected_load)
    assert client.get("/health").json() == {"status": "ok"}
    for _ in range(2):
        response = client.post("/v1/predict", json={"instances": [[2, 3], [1, 1]]})
        assert response.status_code == 200
        assert response.json()["predictions"] == pytest.approx([8, 3])
        assert response.json()["model_version"] == "test-1"


@pytest.mark.parametrize("instances", [[], [[]], [[1]], [[1, 2], [1]], [["text", 2]], [["NaN", 2]]])
def test_bad_requests_return_422(client, instances):
    assert client.post("/v1/predict", json={"instances": instances}).status_code == 422


def test_missing_model_prevents_startup(tmp_path):
    with pytest.raises(FileNotFoundError):
        with TestClient(create_app(Settings(tmp_path / "missing.joblib"))):
            pass
