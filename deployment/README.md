# Despliegue — prueba de arranque

Por ahora la API habilita únicamente `GET /health`. No carga ningún modelo ni
requiere un artefacto entrenado. Swagger está disponible en
http://localhost:8000/docs y su esquema en `/openapi.json` para probar `/health`.
Clasificación y ReDoc siguen deshabilitados. La raíz `/` devuelve 404.

Desde la raíz del proyecto:

```bash
poetry run uvicorn deployment.api.main:app --reload
```

Para verificar:

```bash
curl http://localhost:8000/health
```

Respuesta HTTP 200:

```json
{"status": "ok"}
```

Con Docker iniciado:

```bash
docker compose -f deployment/compose.yaml up --build
```

`api/main.py` contiene la aplicación FastAPI y `api/schemas.py` conserva el
contrato provisional para cuando se habilite la clasificación. En ese momento se
conectará el pipeline entrenado y se definirán las variables de entrada del body.
El proyecto reutiliza Poetry y el lockfile de la raíz.
