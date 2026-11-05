from contextlib import asynccontextmanager

from fastapi import FastAPI

from amarilo.adapters.inbound.api.routes import router
from amarilo.adapters.outbound.sklearn_predictor import SklearnPredictor
from amarilo.application.use_cases.predict import Predict
from amarilo.infrastructure.settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predict = Predict(SklearnPredictor(config.model_path))
        yield
        del app.state.predict

    app = FastAPI(title="Amarilo · Inferencia", version="0.1.0", lifespan=lifespan)
    app.include_router(router)
    return app
