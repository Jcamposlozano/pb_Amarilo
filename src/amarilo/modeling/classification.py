"""Comparación binaria con holdout externo y selección en validación interna."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "average_precision")


def build_models(random_state: int = 42) -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Ridge Classifier": RidgeClassifier(),
        "SGD Classifier": SGDClassifier(loss="log_loss", max_iter=1000, random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(probability=True, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "Extra Trees": ExtraTreesClassifier(random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "AdaBoost": AdaBoostClassifier(random_state=random_state),
        "Naive Bayes": GaussianNB(),
    }


@dataclass
class DataSplits:
    X_train: Any
    X_validation: Any
    X_test: Any
    y_train: Any
    y_validation: Any
    y_test: Any
    X_development: Any
    y_development: Any


def split_data(X, y, *, positive_label=1, random_state: int = 42) -> DataSplits:
    """80/20 externo; sobre el 80%, otro 80/20: aproximadamente 64/16/20."""
    labels = np.unique(y)
    if len(labels) != 2 or positive_label not in labels:
        raise ValueError("Se requieren dos clases y una positive_label presente en y.")
    try:
        X_dev, X_test, y_dev, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=random_state
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_dev, y_dev, test_size=0.2, stratify=y_dev, random_state=random_state
        )
    except ValueError as exc:
        raise ValueError(f"No fue posible crear las particiones estratificadas: {exc}") from exc
    for labels_in_split in (y_train, y_val, y_test):
        if len(np.unique(labels_in_split)) != 2:
            raise ValueError("Cada partición necesita ambas clases; aumenta los datos por clase.")
    return DataSplits(X_train, X_val, X_test, y_train, y_val, y_test, X_dev, y_dev)


def evaluate_model(model, X_data, y_true, *, positive_label=1) -> dict[str, float]:
    """AUC y AP usan scores de la clase positiva, nunca clases predichas."""
    classes = np.asarray(model.classes_)
    if len(classes) != 2 or positive_label not in classes:
        raise ValueError("El modelo debe ser binario y contener positive_label.")
    if not np.isin(y_true, classes).all():
        raise ValueError("y_true contiene clases desconocidas para el modelo.")
    y_binary = np.asarray(y_true) == positive_label
    predicted_binary = np.asarray(model.predict(X_data)) == positive_label
    metrics = {
        "accuracy": float(accuracy_score(y_binary, predicted_binary)),
        "precision": float(precision_score(y_binary, predicted_binary, zero_division=0)),
        "recall": float(recall_score(y_binary, predicted_binary, zero_division=0)),
        "f1": float(f1_score(y_binary, predicted_binary, zero_division=0)),
        "roc_auc": float("nan"),
        "average_precision": float("nan"),
    }
    if hasattr(model, "predict_proba"):
        positive_index = int(np.flatnonzero(classes == positive_label)[0])
        scores = model.predict_proba(X_data)[:, positive_index]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_data)
        if classes[1] != positive_label:
            scores = -scores
    else:
        return metrics
    if len(np.unique(y_binary)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_binary, scores))
        metrics["average_precision"] = float(average_precision_score(y_binary, scores))
    return metrics


@dataclass
class ValidationResult:
    metrics: list[dict]
    pipelines: dict[str, Pipeline]
    best_model_name: str
    selection_metric: str
    positive_label: Any


def validate_models(
    splits: DataSplits,
    *,
    models: dict[str, Any] | None = None,
    selection_metric: str = "f1",
    positive_label=1,
    random_state: int = 42,
    preprocessor=None,
) -> ValidationResult:
    """Entrena en 64%, compara en 16%; no accede al conjunto final."""
    if selection_metric not in METRICS:
        raise ValueError(f"selection_metric debe ser una de {METRICS}.")
    candidates = build_models(random_state) if models is None else models
    if not candidates:
        raise ValueError("Debes proporcionar al menos un modelo.")
    preprocessing = (
        preprocessor
        if preprocessor is not None
        else Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
            ]
        )
    )
    rows, pipelines = [], {}
    for name, estimator in candidates.items():
        pipeline = Pipeline(
            [("preprocessor", clone(preprocessing)), ("classifier", clone(estimator))]
        )
        try:
            pipeline.fit(splits.X_train, splits.y_train)
            metrics = evaluate_model(
                pipeline, splits.X_validation, splits.y_validation, positive_label=positive_label
            )
        except ValueError as exc:
            raise ValueError(f"Falló la validación de {name}: {exc}") from exc
        pipelines[name] = pipeline
        rows.append({"model": name, **metrics})
    eligible = [row for row in rows if np.isfinite(row[selection_metric])]
    if not eligible:
        raise ValueError("Ningún modelo tiene un valor finito para la métrica de selección.")
    # En empate se conserva el orden del catálogo; no se consulta el test.
    best = max(eligible, key=lambda row: row[selection_metric])
    return ValidationResult(rows, pipelines, best["model"], selection_metric, positive_label)


@dataclass
class FinalResult:
    metrics: list[dict]
    best_model_name: str
    best_pipeline: Pipeline


def evaluate_final(validation: ValidationResult, splits: DataSplits) -> FinalResult:
    """Reajusta todos en 80%, evalúa en 20%; conserva la selección de validación."""
    rows = []
    best_pipeline = None
    for name, validated_pipeline in validation.pipelines.items():
        pipeline = clone(validated_pipeline)
        pipeline.fit(splits.X_development, splits.y_development)
        metrics = evaluate_model(
            pipeline, splits.X_test, splits.y_test, positive_label=validation.positive_label
        )
        rows.append({"model": name, **metrics})
        if name == validation.best_model_name:
            best_pipeline = pipeline
    if best_pipeline is None:
        raise ValueError("El modelo seleccionado no está en los pipelines de validación.")
    return FinalResult(rows, validation.best_model_name, best_pipeline)
