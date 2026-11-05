from typing import Annotated

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from amarilo.domain.prediction import InvalidPredictionInput

router = APIRouter()


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instances: Annotated[
        list[Annotated[list[FiniteFloat], Field(min_length=1, max_length=1000)]],
        Field(min_length=1, max_length=1000),
    ]


class PredictionResponse(BaseModel):
    predictions: list[float | int | str]
    model_version: str


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/v1/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    try:
        result = request.app.state.predict.execute(payload.instances)
    except InvalidPredictionInput as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PredictionResponse(predictions=result.predictions, model_version=result.model_version)
