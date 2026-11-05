# Amarilo — analítica e inferencia

Base en Python 3.13 y Poetry para desarrollar analítica y desplegar **un único
modelo de machine learning** mediante FastAPI. El entrenamiento se ejecuta fuera
del servicio; el contenedor solo carga el artefacto y atiende inferencias.

## Estructura

```text
.
├── pyproject.toml                 # Paquete y grupos de dependencias de Poetry
├── poetry.lock                    # Versiones reproducibles
├── Dockerfile                     # Imagen de inferencia en dos etapas
├── compose.yaml                   # API + un artefacto montado en lectura
├── .env.example
├── src/
│   ├── amarilo/
│   │   ├── domain/
│   │   │   └── prediction.py       # Resultado y errores de negocio
│   │   ├── application/
│   │   │   ├── ports/predictor.py  # Puerto de salida: contrato de inferencia
│   │   │   └── use_cases/predict.py
│   │   ├── adapters/
│   │   │   ├── inbound/api/routes.py
│   │   │   └── outbound/sklearn_predictor.py
│   │   └── infrastructure/
│   │       ├── settings.py         # Configuración por variables de entorno
│   │       └── bootstrap.py        # Conecta adaptadores y caso de uso
│   ├── main.py                    # Entrada alternativa de la API
│   ├── notebook/                  # Notebook existente, conservado
│   └── data/raw/                  # Datos existentes, conservados
├── data/processed/                # Datos transformados para análisis
├── models/                        # Un pipeline entrenado + su versión
├── scripts/train_demo.py          # Ejemplo sintético reemplazable
├── tests/
│   ├── unit/                      # Caso de uso sin HTTP ni scikit-learn
│   └── integration/               # HTTP → caso de uso → pipeline real
└── docs/architecture.md
```

## Arquitectura hexagonal

Flujo: HTTP → adaptador de entrada → caso de uso → puerto `Predictor` →
adaptador scikit-learn → pipeline entrenado.

El dominio no importa FastAPI, pandas ni scikit-learn. La aplicación depende del
dominio y define el puerto; los adaptadores implementan las interacciones externas.
`bootstrap.py` conecta todo. Puedes reemplazar scikit-learn por otro motor
implementando el mismo puerto.

## Desarrollo local

Requisitos: Python 3.13 y Poetry 2.3.2 (versión utilizada por el Dockerfile).
Ejecuta desde la raíz del proyecto:

```bash
poetry install --with analytics
poetry run python scripts/train_demo.py
poetry run uvicorn amarilo.infrastructure.bootstrap:create_app --factory --reload
```

`analytics` es opcional e incluye pandas, ipykernel y openpyxl para los notebooks
y los archivos Excel existentes. Para trabajar solo en la API: `poetry install`.
La dependencia de pandas existente se conserva en ese grupo. Los notebooks pueden
seguir usando las rutas originales; no se movieron los datos.

La API está en http://localhost:8000 y Swagger en http://localhost:8000/docs.

```bash
curl -X POST http://localhost:8000/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"instances": [[2.0, 3.0], [1.0, 1.0]]}'
```

Respuesta del ejemplo (con posibles diferencias de redondeo):

```json
{"predictions": [8.0, 3.0], "model_version": "demo-0.1.0"}
```

El contrato inicial admite matrices numéricas y una predicción escalar por fila.
El pipeline debe incluir el preprocesamiento y las columnas deben enviarse en el
orden de `feature_names`. Ver [contrato del artefacto](models/README.md).
El modelo sintético solo comprueba el circuito de inferencia; no está entrenado
con los datos de clientes ni representa una solución de negocio.

`MODEL_PATH` configura la ruta del artefacto. La configuración lee el entorno del
proceso; `.env` no se carga automáticamente. Ejemplo de ruta alternativa:

```bash
MODEL_PATH=/ruta/model.joblib poetry run uvicorn amarilo.infrastructure.bootstrap:create_app --factory
```

## Docker

Primero genera el ejemplo anterior o coloca tu artefacto real en
`models/model.joblib`. Después:

```bash
docker compose up --build
```

La imagen instala únicamente las dependencias de producción y se ejecuta con un
usuario sin privilegios. Datos, notebooks, secretos y artefactos quedan fuera del
contexto de construcción. Compose monta un único modelo en lectura; no entrena al
arrancar. Si el artefacto falta o es incompatible, la API no inicia.

Se usa un worker: el modelo se carga una vez durante el arranque de ese proceso.
Más workers implicarían una copia adicional del modelo en memoria por proceso.
`GET /health` devuelve 200 una vez terminado el arranque. Cambiar el artefacto
requiere reiniciar el servicio. El servicio incluye límites de filas y columnas
por petición; límites de bytes, autenticación y TLS dependen del entorno de
despliegue y deben configurarse antes de exponerlo públicamente.

## Comparación de modelos

El módulo offline `amarilo.modeling.classification` compara 11 clasificadores
con particiones 64/16/20, selecciona en validación y evalúa en el test reservado.
Exporta solo el ganador seleccionado. Consulta los ejemplos de notebook y CLI en
[comparación de modelos](docs/model_comparison.md).

## Verificación

```bash
poetry check --lock
poetry run ruff check src/amarilo src/main.py scripts tests
poetry run ruff format --check src/amarilo src/main.py scripts tests
poetry run pytest
```

Incluye pruebas del caso de uso, validación de entradas, inferencia HTTP con un
pipeline real, carga única y fallo al arrancar sin modelo.

## Referencias

- [Poetry: instalación y archivo lock](https://python-poetry.org/docs/basic-usage/).
- [FastAPI: lifespan para cargar recursos al arrancar](https://fastapi.tiangolo.com/advanced/events/).
