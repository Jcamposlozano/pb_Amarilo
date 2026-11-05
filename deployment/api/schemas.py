"""Contrato provisional: reemplazar ClassificationRequest al definir las variables."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


class ClassificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # TODO: reemplazar por campos de negocio tipados cuando se definan las variables.
    features: Annotated[list[FiniteFloat], Field(min_length=1, max_length=1000)]

    def to_model_row(self) -> list[float]:
        """Adaptar aquí el orden de los futuros campos al feature_names del artefacto."""
        return list(self.features)


class ClassificationResponse(BaseModel):
    classification: int | str | float | bool
