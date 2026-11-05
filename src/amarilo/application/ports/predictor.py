from typing import Protocol

from amarilo.domain.prediction import PredictionResult


class Predictor(Protocol):
    def predict(self, rows: list[list[float]]) -> PredictionResult: ...
