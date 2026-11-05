"""Transformaciones sin ajuste, compartidas por entrenamiento e inferencia."""

import numpy as np
import pandas as pd

categoricas = [
    "REFERIDO",
    "TIPO_DE_PROYECTO",
    "METODO_DE_VENTAS",
    "CIUDAD",
    "DESTINO",
    "TIPO_VENTA",
    "SALA_DE_VENTAS_ALTERNATIVA",
    "OCUPACION_COMPRADOR1",
    "CIUDAD_COMPRADOR1",
]
numericas = ["IMPORTE_TOTAL_OP_PRIMERA_VERSION"]
fechas = ["FECHA_DE_CREACION_OP", "FECHA_INICIO_VENTAS_TRANSACCIONAL"]
target = "COMPRA"
columnas_modelo = categoricas + numericas + fechas + [target]
columnas_numericas = numericas + [
    "MES_CREACION",
    "DIA_SEMANA_CREACION",
    "ANIO_CREACION",
    "ANTIGUEDAD_PROYECTO_DIAS",
]

columnas_entrada = categoricas + numericas + fechas


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    missing = set(columnas_entrada) - set(data.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
    # Selección explícita: no incorporar identificadores ni columnas ajenas al modelo.
    frame = data[columnas_entrada].copy()
    for col in categoricas:
        frame[col] = (
            frame[col]
            .astype("string")
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NULL": pd.NA, "NaN": pd.NA})
            .fillna("SIN_INFORMACION")
            .astype(str)
        )
    for col in fechas:
        frame[col] = pd.to_datetime(frame[col], errors="coerce", utc=True)
    created = frame["FECHA_DE_CREACION_OP"]
    frame["MES_CREACION"] = created.dt.month
    frame["DIA_SEMANA_CREACION"] = created.dt.day_of_week
    frame["ANIO_CREACION"] = created.dt.year
    age = (created - frame["FECHA_INICIO_VENTAS_TRANSACCIONAL"]).dt.total_seconds() / 86400
    frame["ANTIGUEDAD_PROYECTO_DIAS"] = age.where(age >= 0)
    for col in columnas_numericas:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame[columnas_numericas] = frame[columnas_numericas].replace([np.inf, -np.inf], np.nan)
    return frame[categoricas + columnas_numericas].copy()
