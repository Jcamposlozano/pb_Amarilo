"""Preparación determinista del dataset y preprocesamiento ajustable por partición."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_DATASET = Path("src/notebook/data/bronce/df_bronce.csv")

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


def prepare_dataset(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Limpia sin aprender estadísticas ni categorías del conjunto completo."""
    missing = set(columnas_modelo) - set(data.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
    # Selección explícita: no incorporar identificadores ni columnas ajenas al modelo.
    frame = data[columnas_modelo].copy()
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
    frame["DIA_SEMANA_CREACION"] = created.dt.dayofweek
    frame["ANIO_CREACION"] = created.dt.year
    age = (created - frame["FECHA_INICIO_VENTAS_TRANSACCIONAL"]).dt.total_seconds() / 86400
    frame["ANTIGUEDAD_PROYECTO_DIAS"] = age.where(age >= 0)
    for col in columnas_numericas:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame[columnas_numericas] = frame[columnas_numericas].replace([np.inf, -np.inf], np.nan)
    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    frame = frame.loc[frame[target].isin([0, 1])].copy()
    if frame.empty:
        raise ValueError("No hay registros con COMPRA válida (0/1).")
    X = frame[categoricas + columnas_numericas].copy()
    y = frame[target].astype(int)
    return X, y


def load_dataset(path: Path | str = DEFAULT_DATASET):
    """La lectura ocurre al lanzar el entrenamiento, nunca al importar el módulo."""
    return prepare_dataset(pd.read_csv(path))


def build_preprocessor() -> ColumnTransformer:
    """Cada candidato aprende mediana, escalado y categorías solo de su train."""
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                columnas_numericas,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False),
                categoricas,
            ),
        ],
        remainder="drop",
        sparse_threshold=0,
    )
