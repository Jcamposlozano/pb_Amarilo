# Decisiones de arquitectura

- Un paquete `src/amarilo`, instalable con Poetry. Python limitado a 3.13 para
  alinear desarrollo y contenedor; ampliar el rango exige validar el stack de ML.
- `domain`: contratos de negocio sin dependencias de frameworks.
- `application`: orquesta inferencia y define el puerto de salida `Predictor`.
  El método `Predict.execute` es la entrada del caso de uso.
- `adapters/inbound`: traduce HTTP al caso de uso y sus errores a HTTP 422.
- `adapters/outbound`: conoce joblib y scikit-learn. Valida número de columnas y
  ejecuta el pipeline completo, incluido su preprocesamiento.
- `infrastructure`: configuración y composición. Carga el único modelo en el
  lifespan de FastAPI; un error de carga impide que el servicio acepte tráfico.
- `scripts` y notebooks: exploración y entrenamiento fuera del ciclo de peticiones.
  El ejemplo sintético no usa los datos existentes. Cuando se defina el objetivo,
  agregar aquí el entrenamiento, evaluación y exportación del pipeline real.
- `modeling`: utilidades offline de comparación de clasificadores binarios con
  scikit-learn. Se consumen desde scripts o notebooks; no se importan desde el
  dominio, la aplicación de inferencia ni el arranque de la API.
- `models`: artefacto externo a la imagen, versionado con metadatos dentro del
  archivo. Sin catálogo multimodelo, selección dinámica ni servidor de entrenamiento.
- Sin base de datos, cola ni repositorios vacíos: se agregan como puertos y
  adaptadores cuando exista una necesidad concreta del producto.

Para nuevas funciones de analítica reutilizables, ubicar reglas en dominio y
casos de uso en aplicación. Mantener lectura de CSV/Excel, conexiones y pandas
en adaptadores o scripts; evitar importar notebooks desde el servicio.

Los datos existentes se conservan en `src/data/raw` y el notebook en
`src/notebook`. `.gitignore` evita agregar datos y modelos nuevos accidentalmente;
no elimina archivos que ya estuvieran versionados. La lista permitida de
`.dockerignore` restringe la imagen al paquete de inferencia.
