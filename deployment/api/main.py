"""API mínima para verificar el arranque: solo GET /health."""

from fastapi import FastAPI


def create_app() -> FastAPI:
    api = FastAPI(
        title="Amarilo · Health",
        version="0.1.0",
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return api


app = create_app()
