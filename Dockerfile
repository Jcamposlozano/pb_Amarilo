FROM python:3.13-slim AS builder

ARG POETRY_VERSION=2.3.2
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN pip install "poetry==${POETRY_VERSION}"
COPY pyproject.toml poetry.lock README.md ./
RUN poetry sync --only main --no-root
COPY src/amarilo ./src/amarilo
RUN poetry install --only-root

FROM python:3.13-slim AS runtime
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/model.joblib
WORKDIR /app
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --no-create-home app \
    && mkdir /app/models
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=2)"
CMD ["uvicorn", "amarilo.infrastructure.bootstrap:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
