from pathlib import Path

import joblib
from sklearn.base import is_classifier

from amarilo.domain.prediction import InvalidPredictionInput, PredictionResult


class SklearnPredictor:
    """Carga un único pipeline; el preprocesamiento debe estar incluido en él."""

    def __init__(self, model_path: Path, *, require_classifier: bool = False) -> None:
        # joblib solo debe cargar artefactos de confianza, generados por el equipo.
        artifact = joblib.load(model_path)
        self._model = artifact["pipeline"]
        self._version = artifact["model_version"]
        self._feature_names = artifact["feature_names"]
        if (
            not isinstance(self._version, str)
            or not self._version
            or not isinstance(self._feature_names, list)
            or not self._feature_names
            or not all(isinstance(name, str) for name in self._feature_names)
            or not callable(getattr(self._model, "predict", None))
            or getattr(self._model, "n_features_in_", None) != len(self._feature_names)
        ):
            raise ValueError("El artefacto no cumple el contrato documentado del modelo.")
        if require_classifier and not is_classifier(self._model):
            raise ValueError("La API de clasificación requiere un clasificador entrenado.")

    def predict(self, rows: list[list[float]]) -> PredictionResult:
        if any(len(row) != len(self._feature_names) for row in rows):
            raise InvalidPredictionInput(
                f"Se esperan {len(self._feature_names)} columnas en este orden: "
                + ", ".join(self._feature_names)
            )
        return PredictionResult(
            predictions=self._model.predict(rows).tolist(),
            model_version=self._version,
        )
