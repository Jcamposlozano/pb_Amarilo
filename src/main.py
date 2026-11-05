"""Punto de entrada compatible: poetry run uvicorn main:app --app-dir src."""

from amarilo.infrastructure.bootstrap import create_app

app = create_app()
