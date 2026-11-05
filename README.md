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

`analytics` es opcional e incluye ipykernel y openpyxl para los notebooks
y los archivos Excel existentes. Para trabajar solo en la API: `poetry install`.
pandas se instala también en producción para el preprocesamiento de inferencia. Los notebooks pueden
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

Para desplegar el clasificador con el body de oportunidad, usa el proyecto
[deployment](deployment/README.md), con entrada FastAPI en `deployment/api/main.py`
y `POST /v1/classify`, `GET /health` y Swagger. Coloca el pipeline entrenado en
`deployment/model/model.joblib`; `scripts/train_model.py` exporta allí el ganador.

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

Para ejecutar el entrenamiento con la preparación de datos de Amarilo:

```bash
poetry install --with analytics
poetry run python scripts/train_model.py
```

Selecciona por `average_precision` y guarda `resultados_validacion.csv` y
`resultados_finales.csv`, y el pipeline ganador en `deployment/model/model.joblib`.
Usa `--data` para indicar la ubicación real del CSV.

El módulo offline `amarilo.modeling.classification` compara 11 clasificadores
con particiones 64/16/20, selecciona en validación y evalúa en el test reservado.
Exporta solo el ganador seleccionado. Consulta los ejemplos de notebook y CLI en
[comparación de modelos](docs/model_comparison.md).

## Verificación

```bash
poetry check --lock
poetry run ruff check src/amarilo src/main.py scripts tests deployment
poetry run ruff format --check src/amarilo src/main.py scripts tests deployment
poetry run pytest
```

Incluye pruebas del caso de uso, validación de entradas, inferencia HTTP con un
pipeline real, carga única y fallo al arrancar sin modelo.

## Referencias

- [Poetry: instalación y archivo lock](https://python-poetry.org/docs/basic-usage/).
- [FastAPI: lifespan para cargar recursos al arrancar](https://fastapi.tiangolo.com/advanced/events/).



```bash
[1/3] Dividiendo dataset...
Entrenamiento: (1780, 14)
Validación: (446, 14)
Prueba: (557, 14)

[2/3] Entrenando y validando modelos...
/Users/jonathancamposlozano/Documents/Amarilo/.venv/lib/python3.13/site-packages/sklearn/svm/_base.py:236: FutureWarning: The `probability` parameter was deprecated in 1.9 and will be removed in version 1.11. Use `CalibratedClassifierCV(SVC(), ensemble=False)` instead of `SVC(probability=True)`
  warnings.warn(

RESULTADOS DE VALIDACIÓN
              model  accuracy  precision   recall       f1  roc_auc  average_precision
        Extra Trees  0.838565   0.830579 0.866379 0.848101 0.911628           0.916804
      Random Forest  0.822870   0.802372 0.875000 0.837113 0.910631           0.906927
  Gradient Boosting  0.784753   0.793103 0.793103 0.793103 0.864426           0.870812
                SVM  0.771300   0.762097 0.814655 0.787500 0.859219           0.868647
   Ridge Classifier  0.775785   0.789474 0.775862 0.782609 0.851001           0.857385
Logistic Regression  0.784753   0.798246 0.784483 0.791304 0.852310           0.856725
     SGD Classifier  0.780269   0.810185 0.754310 0.781250 0.849511           0.847341
           AdaBoost  0.730942   0.739316 0.745690 0.742489 0.822017           0.818815
                KNN  0.744395   0.743802 0.775862 0.759494 0.816931           0.775211
      Decision Tree  0.784753   0.798246 0.784483 0.791304 0.788290           0.742279
        Naive Bayes  0.594170   0.567639 0.922414 0.702791 0.744320           0.705748

Mejor modelo: Extra Trees

[3/3] Evaluando modelos en holdout...
/Users/jonathancamposlozano/Documents/Amarilo/.venv/lib/python3.13/site-packages/sklearn/svm/_base.py:236: FutureWarning: The `probability` parameter was deprecated in 1.9 and will be removed in version 1.11. Use `CalibratedClassifierCV(SVC(), ensemble=False)` instead of `SVC(probability=True)`
  warnings.warn(

RESULTADOS FINALES
              model  accuracy  precision   recall       f1  roc_auc  average_precision
      Random Forest  0.842011   0.821656 0.889655 0.854305 0.926508           0.935369
        Extra Trees  0.840215   0.813084 0.900000 0.854337 0.920083           0.928459
  Gradient Boosting  0.779174   0.783051 0.796552 0.789744 0.865840           0.863996
                SVM  0.788151   0.779221 0.827586 0.802676 0.862198           0.854170
Logistic Regression  0.761221   0.773519 0.765517 0.769497 0.845150           0.845593
   Ridge Classifier  0.748654   0.758621 0.758621 0.758621 0.838486           0.842488
           AdaBoost  0.755835   0.758389 0.779310 0.768707 0.827289           0.828392
     SGD Classifier  0.707361   0.850829 0.531034 0.653928 0.823763           0.821720
                KNN  0.734291   0.738255 0.758621 0.748299 0.823770           0.790054
      Decision Tree  0.804309   0.787302 0.855172 0.819835 0.806335           0.753066
        Naive Bayes  0.592460   0.568330 0.903448 0.697736 0.708266           0.680340

Modelo seleccionado: Extra Trees
```



1. Si el equipo comercial solo puede gestionar 100 clientes al día, ¿cómo utilizarías las probabilidades para priorizar los contactos?
Las probabilidades permitirían mejorar la priorización de los contactos mediante la creación de un modelo de perfilamiento, con el objetivo de invertir más tiempo en aquellos leads que presentan una mayor probabilidad de compra. Sin embargo, el modelo desarrollado todavía no cuenta con esta capacidad. Aunque es posible identificar algunas variables relevantes, estas aún no son lo suficientemente contundentes para realizar dicha afirmación.

2. ¿Qué diferencia existe entre predecir una compra y demostrar que una gestión comercial causó esa compra?
La predicción es una inferencia basada en patrones y relaciones identificadas en los datos. Sin embargo, demostrar causalidad requiere un estudio de mayor profundidad que permita comprender cómo la gestión comercial influyó directamente en la conversión de una oportunidad en una compra.

3. ¿Qué riesgos de sesgo o calidad podrían afectar las predicciones?
Todos los modelos deben pasar por dos fases fundamentales: la validación en el mundo real y el ajuste posterior. Podemos desarrollar y optimizar un modelo, pero esto no garantiza que estemos solucionando el problema de negocio. Los riesgos de sesgo y calidad de los datos son considerables, especialmente en esta primera fase, que puede entenderse como exploratoria.

4. ¿Qué indicadores propondrías para medir el valor del modelo una vez implementado?
Se pueden establecer indicadores clave de desempeño (KPIs), como el incremento en la tasa de conversión de los contactos comerciales. Para validar estos resultados, sería recomendable implementar una prueba A/B que permita determinar si el uso del modelo realmente genera valor para la organización.

5. ¿Cómo detectarías que el modelo está perdiendo desempeño y cuándo considerarías reentrenarlo?
Podríamos monitorear las métricas de evaluación de los modelos, como F1, F2 y F3, para identificar posibles disminuciones en su desempeño. Sin embargo, más allá de la evaluación de las métricas, buscaría comprender los nuevos escenarios que justificarían un reentrenamiento, considerando que cada caso puede presentar diferencias significativas según las características de las muestras.

Para ello, sería importante establecer entornos de experimentación independientes que permitan evaluar diferentes escenarios y garantizar la mayor diversidad posible en los datos, con el propósito de reducir los sesgos y mejorar la capacidad de generalización del modelo.