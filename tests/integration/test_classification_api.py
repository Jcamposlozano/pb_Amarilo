from fastapi.testclient import TestClient

from deployment.api.main import create_app


def test_health_starts_without_model(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "/nonexistent/model.joblib")
    with TestClient(create_app()) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        assert not hasattr(client.app.state, "classify")


def test_only_health_is_enabled():
    app = create_app()
    with TestClient(app) as client:
        assert client.get("/docs").status_code == 200
        schema = client.get("/openapi.json")
        assert schema.status_code == 200
        assert list(schema.json()["paths"]) == ["/health"]
        assert client.post("/v1/classify", json={"features": [1, 2]}).status_code == 404
        for path in ["/", "/redoc"]:
            assert client.get(path).status_code == 404
