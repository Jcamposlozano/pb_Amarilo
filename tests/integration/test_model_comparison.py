import csv
import json
import subprocess
import sys
from pathlib import Path

from sklearn.datasets import make_classification

from amarilo.adapters.outbound.sklearn_predictor import SklearnPredictor


def test_csv_cli_exports_reports_and_one_model_compatible_with_serving(tmp_path):
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    dataset_path = tmp_path / "input.csv"
    features = ["a", "b", "c", "d"]
    with dataset_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([*features, "target"])
        writer.writerows([*row, label] for row, label in zip(X, y, strict=True))
    artifact = tmp_path / "winner.joblib"
    reports = tmp_path / "reports"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/compare_models.py",
            "--csv",
            str(dataset_path),
            "--target",
            "target",
            "--features",
            *features,
            "--output-dir",
            str(reports),
            "--model-output",
            str(artifact),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    metadata = json.loads((reports / "experiment.json").read_text())
    assert metadata["split_sizes"] == {"train": 64, "validation": 16, "test": 20, "development": 80}
    for name in ["validation_metrics.csv", "test_metrics.csv"]:
        with (reports / name).open() as stream:
            assert len(list(csv.DictReader(stream))) == 11
    assert list(tmp_path.glob("*.joblib")) == [artifact]
    predictions = SklearnPredictor(artifact).predict(X[:3].tolist())
    assert len(predictions.predictions) == 3
    assert predictions.model_version == "comparison-0.1.0"
