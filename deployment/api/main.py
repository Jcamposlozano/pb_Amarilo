"""API REST: health y clasificación de una oportunidad con un pipeline entrenado."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from amarilo.adapters.outbound.opportunity_classifier import OpportunityClassifier
from deployment.api.schemas import ClassificationRequest, ClassificationResponse

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "model.joblib"
logger = logging.getLogger(__name__)


def create_app(model_path: Path | None = None) -> FastAPI:
    path = (
        model_path if model_path is not None else Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    )

    @asynccontextmanager
    async def lifespan(api: FastAPI):
        # Sin artefacto se permite comprobar health/docs; clasificación devuelve 503.
        api.state.classifier = None
        if path.is_file():
            try:
                api.state.classifier = OpportunityClassifier(path)
            except (ValueError, KeyError, TypeError) as exc:
                logger.error("No se pudo habilitar clasificación con %s: %s", path, exc)
        try:
            yield
        finally:
            del api.state.classifier

    api = FastAPI(
        title="Amarilo · Clasificación",
        version="0.1.0",
        redoc_url=None,
        lifespan=lifespan,
    )

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.post("/v1/classify", response_model=ClassificationResponse)
    def classify(body: ClassificationRequest, request: Request) -> ClassificationResponse:
        classifier = request.app.state.classifier
        if classifier is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Modelo no disponible o incompatible. Exporta el pipeline con "
                    "scripts/train_model.py a deployment/model/model.joblib y reinicia."
                ),
            )
        return ClassificationResponse(classification=classifier.classify(body.to_opportunity()))

    return api


app = create_app()
