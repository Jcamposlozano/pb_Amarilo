Coloca aquí `model.joblib`, el clasificador exportado por `scripts/train_model.py`.
Debe incluir el pipeline completo con el preprocesador aprendido y conservar los
nombres de las columnas. No sirve un estimador aislado ni un modelo de regresión.
La API también acepta el pipeline completo serializado directamente con joblib.

El entrenamiento guarda el ganador por average precision de validación, reajustado
en el 80% de desarrollo. Reinicia la API cuando reemplaces el archivo. Usa las
mismas versiones de librerías del lockfile y carga solo artefactos de confianza.
El archivo no se incluye en Git ni en la imagen Docker; Compose monta esta carpeta.
