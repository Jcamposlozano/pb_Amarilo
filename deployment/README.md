# Despliegue REST del clasificador

La API recibe una oportunidad en `POST /v1/classify` y devuelve su clasificación:

```json
{"classification": 1}
```

El esquema de entrada está en `api/schemas.py`. Acepta las 17 variables predictoras
solicitadas. `COMPRA` no se acepta en el body: es la etiqueta que queremos
predecir y su valor se devuelve mediante `classification`.
`BK_VENDEDOR` valida un entero de 64 bits y el importe es un número finito.
Las fechas se reciben como cadenas y se transforman igual que en entrenamiento.

## Modelo

Coloca el pipeline completo entrenado en `deployment/model/model.joblib`.
También puedes generarlo con el entrenamiento integrado, desde la raíz:

```bash
poetry install --with analytics
poetry run python scripts/train_model.py
```

Este comando guarda las métricas y exporta el ganador seleccionado por average
precision de validación, reajustado en el 80%. Reemplaza el artefacto de destino.
El archivo contiene el clasificador, encoder, imputación y escalado aprendidos.
Un estimador aislado o un modelo de regresión no es compatible.

La limpieza de categorías y derivación de fechas se comparte entre entrenamiento
e inferencia mediante `amarilo.adapters.outbound.opportunity_features`.
Se usan las nueve categóricas, el importe y las variables derivadas de fecha que
se definieron al entrenar. Se aceptan `ETAPA_DE_VENTA`,
`FECHA_ULTIMA_ACTUALIZACION_OP`, `ESTADO_PROYECTO`, `BK_VENDEDOR` y
`OBSERVACIONES`, pero el modelo actual no las usa. Incorporarlas requiere
reentrenar cambiando la selección de características.

## Iniciar y probar

Desde la raíz del repositorio:

```bash
poetry run uvicorn deployment.api.main:app --reload
```

Swagger: http://localhost:8000/docs. `GET /health` verifica que el proceso responde.
Ejemplo de petición ficticia en `deployment/example_request.json`:

```bash
curl -X POST http://localhost:8000/v1/classify \
  -H 'Content-Type: application/json' \
  --data-binary @deployment/example_request.json
```

Se carga el modelo una vez al arrancar; no se entrena dentro de la API. Sin modelo
o con un formato incompatible, `/health` y `/docs` permanecen disponibles y
clasificación devuelve HTTP 503. El motivo de incompatibilidad se registra en el
log. Un body inválido devuelve HTTP 422. Reinicia la API al colocar o sustituir
el archivo. `MODEL_PATH` permite configurar una ruta alternativa; `.env` no se
carga automáticamente.

Usa solo artefactos de confianza y las mismas versiones de Python y librerías
que en entrenamiento. El modelo no se incluye en Git ni en la imagen.

## Docker

Con Docker iniciado y la carpeta `deployment/model` existente:

```bash
docker compose -f deployment/compose.yaml up --build
```

Compose monta esa carpeta en lectura. Poetry instala pandas en producción porque
la inferencia usa el mismo preprocesamiento tabular que el entrenamiento.
La imagen ejecuta un único worker y un usuario sin privilegios. Configura TLS y
autenticación en la infraestructura de destino antes de exponerla públicamente.
