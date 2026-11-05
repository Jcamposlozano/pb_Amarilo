# Comparación de clasificadores binarios

Se comparan Logistic Regression, Ridge Classifier, SGD Classifier, KNN, SVM,
Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost y Naive Bayes.

## Dos momentos

1. Se reserva el 20% externo como test final. El otro 80% se divide nuevamente
   80/20: aproximadamente 64% del total para entrenamiento y 16% para validación.
   Las dos divisiones son estratificadas, reproducibles con semilla 42 y comunes
   a todos los modelos. Los porcentajes exactos dependen del redondeo por filas.
2. Cada candidato se ajusta únicamente en el 64%. Se calculan sus seis métricas
   sobre el 16%. Se elige el mayor F1 por defecto; la métrica es configurable.
   Los empates conservan el orden del catálogo. Se clona y reajusta cada candidato
   sobre el 80% completo y se calculan las seis métricas definitivas en el 20%.
   Solo se exporta el ganador elegido previamente en validación.

El test final se usa una vez por candidato, como evaluación descriptiva solicitada.
No se vuelve a elegir el ganador usando esa tabla ni se ajustan hiperparámetros
después de verla. El preprocesamiento también se reajusta al entrenar con el 80%.
El modelo exportado conserva ese ajuste: no se reentrena con el test.

## Métricas

| Nombre | Entrada |
| --- | --- |
| `accuracy` | Clases predichas |
| `precision` | Clases predichas; clase positiva configurable |
| `recall` | Clases predichas; clase positiva configurable |
| `f1` | Clases predichas; clase positiva configurable |
| `roc_auc` | Probabilidad positiva o `decision_function` |
| `average_precision` | La misma puntuación continua usada para ROC AUC |

Se corrige `accurracy` a `accuracy` y `evalute_model` a `evaluate_model`.
Ridge usa `decision_function`; se invierte su signo si la clase positiva es la
primera de `classes_`. Las probabilidades se toman de la columna de la clase
positiva, no siempre de la columna 1. Sin scores, AUC y AP son `NaN`.
Si una evaluación aislada contiene una sola clase, ambas se reportan como `NaN`;
el procedimiento de particiones exige ambas clases en los tres subconjuntos.
Las divisiones por cero en precision/recall/F1 devuelven 0.

## Uso en un notebook

Instala el entorno: `poetry install --with analytics`. Usa su kernel Python 3.13.

```python
import pandas as pd
import joblib

from amarilo.modeling.classification import (
    split_data,
    validate_models,
    evaluate_final,
)

# df es tu DataFrame. Define explícitamente las características y el objetivo.
features = ["variable_1", "variable_2"]  # Reemplazar por columnas reales
target = "objetivo"  # Reemplazar por columna real
X = df[features].to_numpy(dtype=float)
y = df[target].to_numpy()

splits = split_data(X, y, positive_label=1)
validation = validate_models(splits, selection_metric="f1", positive_label=1)
display(pd.DataFrame(validation.metrics).sort_values("f1", ascending=False))
print("Seleccionado por validación:", validation.best_model_name)

# Ejecutar al cerrar la selección, sin más ajustes posteriores al ver el test.
final = evaluate_final(validation, splits)
display(pd.DataFrame(final.metrics))

joblib.dump(
    {
        "pipeline": final.best_pipeline,
        "model_version": "classification-0.1.0",
        "feature_names": features,
    },
    "models/selected_model.joblib",
)
```

Para un catálogo propio, pasa `models=tu_diccionario` a `validate_models`.
Los estimadores originales se clonan y no se modifican.
`evaluate_model(model, X_data, y_true, positive_label=1)` también puede usarse
de forma independiente con un modelo ya entrenado.

## Ejecución desde CSV

```bash
poetry run python scripts/compare_models.py \
  --csv ruta/dataset.csv \
  --target objetivo \
  --features variable_1 variable_2 \
  --selection-metric f1 \
  --model-version classification-0.1.0
```

El CSV debe tener encabezado, separador coma, características numéricas y objetivo
codificado como 0/1. Se imputan celdas vacías mediante la mediana y se aplica
`StandardScaler` dentro de cada pipeline. No se permiten objetivos vacíos ni
características infinitas. `--positive-label 0` permite cambiar la clase positiva.

Salidas predeterminadas:

- `reports/model_comparison/validation_metrics.csv`: los 11 modelos en el 16%.
- `reports/model_comparison/test_metrics.csv`: los 11 modelos reajustados, en el 20%.
- `reports/model_comparison/experiment.json`: semilla, selección, columnas y tamaños.
- `models/selected_model.joblib`: un único pipeline, compatible con la API.

Repetir el comando reemplaza esas salidas. Para conservar experimentos, usa
`--output-dir reports/experimento_02 --model-output models/experimento_02.joblib`.
La API puede servir el resultado con:

```bash
MODEL_PATH=models/selected_model.joblib poetry run uvicorn amarilo.infrastructure.bootstrap:create_app --factory
```

## Alcance

El preprocesador predeterminado sirve para datos tabulares numéricos. Excluye
identificadores y columnas que anticipen el resultado. Para variables categóricas,
la función admite `preprocessor=...` con un `ColumnTransformer` propio; el contrato
de entrada numérico de la API necesitaría adaptarse si se usan nombres de columnas
o texto. GaussianNB necesita una salida densa.

Se asume que las filas son observaciones independientes. Si hay varias filas por
cliente o una secuencia temporal, hay que adaptar las particiones por cliente o
fecha antes de usar estas métricas como evaluación final. No se ejecuta sobre los
datos existentes hasta definir la columna objetivo y las características.
KNN requiere al menos cinco filas de entrenamiento. Un fallo de entrenamiento
detiene el experimento con el nombre del candidato; no se ocultan errores.

Se conserva el `SVC(probability=True)` solicitado. scikit-learn 1.9 avisa que ese
parámetro se retirará en 1.11; antes de actualizar a esa versión habrá que migrar
la calibración o usar `decision_function`. El lockfile fija la versión probada.

Referencias: [prevención de fuga de datos con pipelines](https://scikit-learn.org/stable/common_pitfalls.html)
y [scores para average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
