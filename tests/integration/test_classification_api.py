import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline

from amarilo.modeling.dataset_preparin import build_preprocessor, prepare_dataset
from deployment.api.main import create_app


@pytest.fixture
def payload():
    return {
        "ETAPA_DE_VENTA": "Contacto",
        "FECHA_DE_CREACION_OP": "2026-01-15",
        "FECHA_ULTIMA_ACTUALIZACION_OP": "2026-01-16",
        "REFERIDO": "No",
        "TIPO_DE_PROYECTO": "Vivienda",
        "METODO_DE_VENTAS": "Directa",
        "CIUDAD": "Bogotá",
        "IMPORTE_TOTAL_OP_PRIMERA_VERSION": 250000000.0,
        "DESTINO": "Vivienda",
        "TIPO_VENTA": "Directa",
        "SALA_DE_VENTAS_ALTERNATIVA": "Principal",
        "ESTADO_PROYECTO": "En ventas",
        "FECHA_INICIO_VENTAS_TRANSACCIONAL": "2025-01-01",
        "BK_VENDEDOR": 123,
        "OBSERVACIONES": "Contacto inicial",
        "OCUPACION_COMPRADOR1": "Empleado",
        "CIUDAD_COMPRADOR1": "Bogotá",
    }


@pytest.fixture
def model_path(tmp_path, payload):
    rows = []
    for index in range(40):
        rows.append(
            {**payload, "IMPORTE_TOTAL_OP_PRIMERA_VERSION": float(index), "COMPRA": index % 2}
        )
    X, y = prepare_dataset(pd.DataFrame(rows))
    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    ).fit(X, y)
    path = tmp_path / "model.joblib"
    joblib.dump(
        {"pipeline": pipeline, "feature_names": list(X.columns), "model_version": "test-1"}, path
    )
    return path


def test_health_docs_and_missing_model(tmp_path, payload):
    with TestClient(create_app(tmp_path / "absent.joblib")) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/docs").status_code == 200
        paths = client.get("/openapi.json").json()["paths"]
        assert set(paths) == {"/health", "/v1/classify"}
        assert client.post("/v1/classify", json=payload).status_code == 503


def test_classification_loads_once_and_ignores_unused_fields(model_path, payload, monkeypatch):
    with TestClient(create_app(model_path)) as client:

        def unexpected_load(*args, **kwargs):
            pytest.fail("No recargar el artefacto por petición")

        monkeypatch.setattr(joblib, "load", unexpected_load)
        response = client.post("/v1/classify", json=payload)
        assert response.status_code == 200
        assert response.json()["classification"] in [0, 1]
        assert isinstance(response.json()["classification"], int)
        changed = {
            **payload,
            "BK_VENDEDOR": 999,
            "OBSERVACIONES": "Otro texto",
            "ETAPA_DE_VENTA": "Otra etapa",
        }
        assert client.post("/v1/classify", json=changed).json() == response.json()


def test_target_is_not_an_input(model_path, payload):
    with TestClient(create_app(model_path)) as client:
        schema = client.get("/openapi.json").json()
        assert (
            "COMPRA" not in schema["components"]["schemas"]["ClassificationRequest"]["properties"]
        )
        assert client.post("/v1/classify", json=payload).status_code == 200
        assert client.post("/v1/classify", json={**payload, "COMPRA": 1}).status_code == 422


def test_utc_dates_and_unknown_project_start_date(model_path, payload):
    body = {
        **payload,
        "FECHA_DE_CREACION_OP": "2026-08-09 17:14:18.000000 UTC",
        "FECHA_ULTIMA_ACTUALIZACION_OP": "2026-10-18 16:03:55.000000 UTC",
        "FECHA_INICIO_VENTAS_TRANSACCIONAL": "Sin información",
        "IMPORTE_TOTAL_OP_PRIMERA_VERSION": 0,
        "BK_VENDEDOR": 300000701904251,
    }
    with TestClient(create_app(model_path)) as client:
        response = client.post("/v1/classify", json=body)
        assert response.status_code == 200
        assert response.json()["classification"] in (0, 1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("BK_VENDEDOR", 1.5),
        ("BK_VENDEDOR", 2**63),
        ("BK_VENDEDOR", True),
        ("COMPRA", 2),
        ("IMPORTE_TOTAL_OP_PRIMERA_VERSION", "NaN"),
        ("CIUDAD", 1),
    ],
)
def test_invalid_types_return_422(tmp_path, payload, field, value):
    with TestClient(create_app(tmp_path / "absent.joblib")) as client:
        assert client.post("/v1/classify", json={**payload, field: value}).status_code == 422


def test_required_field_and_unknown_field(tmp_path, payload):
    with TestClient(create_app(tmp_path / "absent.joblib")) as client:
        without_city = {key: val for key, val in payload.items() if key != "CIUDAD"}
        assert client.post("/v1/classify", json=without_city).status_code == 422
        assert client.post("/v1/classify", json={**payload, "unknown": 0}).status_code == 422


def test_incompatible_model_leaves_health_available(tmp_path, payload):
    path = tmp_path / "wrong.joblib"
    joblib.dump(LinearRegression().fit([[0], [1]], [0, 1]), path)
    with TestClient(create_app(path)) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/v1/classify", json=payload).status_code == 503


def test_model_path_environment_and_bare_pipeline(model_path, payload, monkeypatch):
    pipeline = joblib.load(model_path)["pipeline"]
    joblib.dump(pipeline, model_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    with TestClient(create_app()) as client:
        assert client.post("/v1/classify", json=payload).status_code == 200
