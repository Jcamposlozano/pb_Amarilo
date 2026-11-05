import pytest

from amarilo.application.use_cases.predict import Predict
from amarilo.domain.prediction import InvalidPredictionInput, PredictionResult


class FakePredictor:
    def predict(self, rows):
        return PredictionResult([sum(row) for row in rows], "test")


def test_predict_through_port_without_framework():
    assert Predict(FakePredictor()).execute([[1, 2]]) == PredictionResult([3], "test")


@pytest.mark.parametrize("rows", [[], [[]], [[1], [1, 2]], [[float("nan")]], [[float("inf")]]])
def test_invalid_input_does_not_reach_model(rows):
    class UnreachablePredictor:
        def predict(self, rows):
            pytest.fail("El modelo no debe recibir entradas inválidas")

    with pytest.raises(InvalidPredictionInput):
        Predict(UnreachablePredictor()).execute(rows)
