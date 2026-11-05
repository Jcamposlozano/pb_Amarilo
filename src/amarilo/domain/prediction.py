"""Contratos de negocio independientes de HTTP y del framework de ML."""

from dataclasses import dataclass


class InvalidPredictionInput(ValueError):
    """La entrada no satisface el contrato del modelo."""


@dataclass(frozen=True)
class PredictionResult:
    predictions: list[float | int | str]
    model_version: str
