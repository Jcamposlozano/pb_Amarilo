Aquí se almacena un único artefacto de inferencia: `model.joblib`.

Contrato: diccionario serializado con joblib con las claves `pipeline` (pipeline
scikit-learn entrenado, con preprocesamiento y `predict`), `model_version` (texto)
y `feature_names` (lista ordenada de nombres de columnas).

La API recibe una matriz numérica en ese orden y devuelve una predicción escalar
por fila (regresión o clasificación de una sola salida). Para otro formato de
entrada o framework, cambia el adaptador de salida y el esquema HTTP.

Entrena y sirve con las mismas versiones de Python y librerías del lockfile.
Carga únicamente artefactos de confianza: joblib puede ejecutar código al cargar.
Los modelos no se incluyen en Git ni en la imagen; se montan en modo lectura.
El script `scripts/train_demo.py` crea un ejemplo sintético, sin usar datos de clientes.
