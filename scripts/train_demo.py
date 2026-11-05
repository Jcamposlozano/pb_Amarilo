"""Artefacto sintético para verificar el servicio; no es un modelo de negocio."""

from pathlib import Path

import joblib
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    pipeline = make_pipeline(StandardScaler(), LinearRegression())
    pipeline.fit([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [0, 1, 2, 3])
    path = Path("models/model.joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": pipeline, "model_version": "demo-0.1.0", "feature_names": ["x1", "x2"]},
        path,
    )
    print(f"Modelo de demostración guardado en {path}")


if __name__ == "__main__":
    main()
