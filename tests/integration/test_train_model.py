import numpy as np
import pytest

from amarilo.adapters.outbound.opportunity_classifier import OpportunityClassifier

pd = pytest.importorskip("pandas", reason="El entrenamiento requiere el grupo analytics")

from amarilo.modeling.dataset_preparin import (  # noqa: E402
    build_preprocessor,
    categoricas,
    columnas_numericas,
    prepare_dataset,
)
from scripts.train_model import main  # noqa: E402


@pytest.fixture
def raw_dataset():
    count = 100
    frame = pd.DataFrame({name: [" A ", "B"] * (count // 2) for name in categoricas})
    frame["IMPORTE_TOTAL_OP_PRIMERA_VERSION"] = np.arange(count, dtype=float)
    frame["FECHA_DE_CREACION_OP"] = "2026-01-15"
    frame["FECHA_INICIO_VENTAS_TRANSACCIONAL"] = "2025-01-01"
    frame["COMPRA"] = [0, 1] * (count // 2)
    frame["ID_NO_USAR"] = np.arange(count)
    return frame


def test_preparation_selects_only_features_and_handles_invalid_values(raw_dataset):
    raw_dataset.loc[0, "REFERIDO"] = "NULL"
    raw_dataset.loc[0, "IMPORTE_TOTAL_OP_PRIMERA_VERSION"] = np.inf
    raw_dataset.loc[0, "FECHA_DE_CREACION_OP"] = "2020-01-01"
    raw_dataset.loc[1, "COMPRA"] = 3
    X, y = prepare_dataset(raw_dataset)
    assert len(X) == len(y) == 99
    assert list(X.columns) == categoricas + columnas_numericas
    assert X.loc[0, "REFERIDO"] == "SIN_INFORMACION"
    assert np.isnan(X.loc[0, "IMPORTE_TOTAL_OP_PRIMERA_VERSION"])
    assert np.isnan(X.loc[0, "ANTIGUEDAD_PROYECTO_DIAS"])
    assert raw_dataset.loc[1, "COMPRA"] == 3


def test_encoder_is_fitted_only_on_training_categories(raw_dataset):
    X, _ = prepare_dataset(raw_dataset)
    preprocessor = build_preprocessor()
    assert not hasattr(preprocessor, "transformers_")
    preprocessor.fit(X.iloc[:80])
    unseen = X.iloc[80:].copy()
    unseen[categoricas] = "NEW_CATEGORY"
    encoded = preprocessor.transform(unseen)
    encoder = preprocessor.named_transformers_["categorical"]
    assert all("NEW_CATEGORY" not in categories for categories in encoder.categories_)
    assert np.isfinite(encoded).all()


def test_training_entrypoint_runs_all_models_and_writes_both_reports(raw_dataset, tmp_path, capsys):
    data_path = tmp_path / "dataset.csv"
    raw_dataset.to_csv(data_path, index=False)
    output = tmp_path / "reports"
    artifact_path = tmp_path / "model.joblib"
    main(
        [
            "--data",
            str(data_path),
            "--output-dir",
            str(output),
            "--model-output",
            str(artifact_path),
        ]
    )
    deployed = OpportunityClassifier(artifact_path)
    assert deployed.classify(raw_dataset.iloc[0].drop("COMPRA").to_dict()) in (0, 1)
    for filename in ["resultados_validacion.csv", "resultados_finales.csv"]:
        metrics = pd.read_csv(output / filename)
        assert len(metrics) == 11
        assert metrics["average_precision"].is_monotonic_decreasing
        assert metrics["roc_auc"].between(0, 1).all()
    console = capsys.readouterr().out
    assert "Entrenamiento finalizado correctamente." in console
    assert "Entrenamiento: (64, 14)" in console


def test_missing_csv_has_clear_cli_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--data", str(tmp_path / "absent.csv")])
    assert exc.value.code == 2
    assert "No existe el dataset" in capsys.readouterr().err
