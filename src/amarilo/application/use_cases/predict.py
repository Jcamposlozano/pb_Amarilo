from math import isfinite

from amarilo.application.ports.predictor import Predictor
from amarilo.domain.prediction import InvalidPredictionInput, PredictionResult


class Predict:
    def __init__(self, predictor: Predictor) -> None:
        self._predictor = predictor

    def execute(self, rows: list[list[float]]) -> PredictionResult:
        if not rows or not rows[0]:
            raise InvalidPredictionInput("Se requiere al menos una fila con características.")
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            raise InvalidPredictionInput("Todas las filas deben tener el mismo número de columnas.")
        if any(not isfinite(value) for row in rows for value in row):
            raise InvalidPredictionInput("Las características deben ser números finitos.")
        return self._predictor.predict(rows)
