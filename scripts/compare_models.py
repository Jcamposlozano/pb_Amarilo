"""CLI offline: CSV numérico -> validación -> evaluación final -> un artefacto."""

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np

from amarilo.modeling.classification import (
    METRICS,
    evaluate_final,
    split_data,
    validate_models,
)


def read_dataset(path: Path, target: str, features: list[str]):
    if target in features or len(set(features)) != len(features):
        raise ValueError("Las características deben ser únicas y no incluir la variable objetivo.")
    X, y = [], []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set([target, *features]) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Columnas ausentes: {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            try:
                values = [float(row[name]) if row[name].strip() else np.nan for name in features]
                label = float(row[target])
                if label not in (0, 1):
                    raise ValueError("La variable objetivo debe ser 0 o 1.")
                if np.isinf(values).any():
                    raise ValueError("No se admiten infinitos en las características.")
                X.append(values)
                y.append(int(label))
            except (TypeError, AttributeError, ValueError) as exc:
                raise ValueError(f"CSV inválido en línea {line}: {exc}") from exc
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def save_metrics(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", *METRICS])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--target", required=True, help="Columna binaria codificada como 0/1")
    parser.add_argument("--features", nargs="+", required=True, help="Columnas numéricas en orden")
    parser.add_argument("--positive-label", type=int, choices=[0, 1], default=1)
    parser.add_argument("--selection-metric", choices=METRICS, default="f1")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/model_comparison"))
    parser.add_argument("--model-output", type=Path, default=Path("models/selected_model.joblib"))
    parser.add_argument("--model-version", default="comparison-0.1.0")
    args = parser.parse_args()

    if not args.model_version.strip():
        parser.error("--model-version no puede estar vacío.")
    try:
        X, y = read_dataset(args.csv, args.target, args.features)
        splits = split_data(
            X, y, positive_label=args.positive_label, random_state=args.random_state
        )
        validation = validate_models(
            splits,
            positive_label=args.positive_label,
            selection_metric=args.selection_metric,
            random_state=args.random_state,
        )
        print(f"Seleccionado en validación: {validation.best_model_name}", flush=True)
        final = evaluate_final(validation, splits)
    except ValueError as exc:
        parser.error(str(exc))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(args.output_dir / "validation_metrics.csv", validation.metrics)
    save_metrics(args.output_dir / "test_metrics.csv", final.metrics)
    metadata = {
        "best_model_name": final.best_model_name,
        "selection_metric": validation.selection_metric,
        "positive_label": args.positive_label,
        "random_state": args.random_state,
        "feature_names": args.features,
        "target": args.target,
        "model_version": args.model_version,
        "split_sizes": {
            "train": len(splits.y_train),
            "validation": len(splits.y_validation),
            "test": len(splits.y_test),
            "development": len(splits.y_development),
        },
    }
    (args.output_dir / "experiment.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": final.best_pipeline,
            "model_version": args.model_version,
            "feature_names": args.features,
            "positive_label": args.positive_label,
            "model_name": final.best_model_name,
        },
        args.model_output,
    )
    print(f"Métricas: {args.output_dir}")
    print(f"Único modelo exportado: {args.model_output}")


if __name__ == "__main__":
    main()
