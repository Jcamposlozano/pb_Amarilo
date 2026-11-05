"""Entrenamiento sobre el dataset preparado de Amarilo (ejecutar desde la raíz)."""

import argparse
from pathlib import Path

import pandas as pd

from amarilo.modeling.classification import evaluate_final, split_data, validate_models
from amarilo.modeling.dataset_preparin import DEFAULT_DATASET, build_preprocessor, load_dataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    if not args.data.is_file():
        parser.error(f"No existe el dataset: {args.data}. Indica la ruta con --data.")
    X, y = load_dataset(args.data)

    print("\n[1/3] Dividiendo dataset...", flush=True)
    splits = split_data(X, y, positive_label=1, random_state=42)
    print(f"Entrenamiento: {splits.X_train.shape}")
    print(f"Validación: {splits.X_validation.shape}")
    print(f"Prueba: {splits.X_test.shape}")

    print("\n[2/3] Entrenando y validando modelos...", flush=True)
    validation = validate_models(
        splits,
        selection_metric="average_precision",
        positive_label=1,
        random_state=42,
        preprocessor=build_preprocessor(),
    )
    resultados_validacion = pd.DataFrame(validation.metrics).sort_values(
        "average_precision", ascending=False, kind="stable"
    )
    print("\nRESULTADOS DE VALIDACIÓN")
    print(resultados_validacion.to_string(index=False))
    print(f"\nMejor modelo: {validation.best_model_name}")

    print("\n[3/3] Evaluando modelos en holdout...", flush=True)
    final = evaluate_final(validation, splits)
    resultados_finales = pd.DataFrame(final.metrics).sort_values(
        "average_precision", ascending=False, kind="stable"
    )
    print("\nRESULTADOS FINALES")
    print(resultados_finales.to_string(index=False))
    print(f"\nModelo seleccionado: {final.best_model_name}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    resultados_validacion.to_csv(args.output_dir / "resultados_validacion.csv", index=False)
    resultados_finales.to_csv(args.output_dir / "resultados_finales.csv", index=False)
    print(f"\nMétricas guardadas en: {args.output_dir.resolve()}")
    print("\nEntrenamiento finalizado correctamente.")


if __name__ == "__main__":
    main()
