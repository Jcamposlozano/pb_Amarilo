"""Preparación determinista del dataset y preprocesamiento ajustable por partición."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from amarilo.adapters.outbound.opportunity_features import (
    categoricas,
    columnas_modelo,  # noqa: F401 -- compatibilidad para consumidores existentes
    columnas_numericas,
    prepare_features,
    target,
)

DEFAULT_DATASET = Path("src/notebook/data/bronce/df_bronce.csv")


def prepare_dataset(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Limpia sin aprender estadísticas ni categorías del conjunto completo."""
    if target not in data.columns:
        raise ValueError(f"Falta la variable objetivo: {target}")
    frame = prepare_features(data)
    frame[target] = data[target]
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
