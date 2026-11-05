"""Inferencia del clasificador entrenado con las características de Amarilo."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import is_classifier

from amarilo.adapters.outbound.opportunity_features import (
    categoricas,
    columnas_numericas,
    prepare_features,
)


class OpportunityClassifier:
    def __init__(self, path: Path):
        # Solo cargar artefactos de confianza generados por el equipo.
        artifact = joblib.load(path)
        self.pipeline = artifact["pipeline"] if isinstance(artifact, dict) else artifact
        expected = categoricas + columnas_numericas
        names = list(getattr(self.pipeline, "feature_names_in_", []))
        if (
            not is_classifier(self.pipeline)
            or names != expected
            or set(getattr(self.pipeline, "classes_", [])) != {0, 1}
        ):
            raise ValueError(
                "Modelo incompatible: se requiere el pipeline binario de train_model.py "
                "con su preprocesamiento y columnas de Amarilo."
            )
        if isinstance(artifact, dict) and artifact.get("feature_names", expected) != expected:
            raise ValueError("Los metadatos de columnas no coinciden con el pipeline.")

    def classify(self, opportunity: dict) -> int:
        features = prepare_features(pd.DataFrame([opportunity]))
        return int(self.pipeline.predict(features)[0])
