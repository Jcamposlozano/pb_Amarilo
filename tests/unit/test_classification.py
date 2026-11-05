import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from amarilo.modeling.classification import (
    METRICS,
    build_models,
    evaluate_final,
    evaluate_model,
    split_data,
    validate_models,
)


def dataset():
    return make_classification(n_samples=200, n_features=6, random_state=12)


def test_split_sizes_stratification_disjoint_and_reproducible():
    X = np.arange(100).reshape(-1, 1)
    y = np.tile([0, 1], 50)
    splits = split_data(X, y)
    assert [len(splits.y_train), len(splits.y_validation), len(splits.y_test)] == [64, 16, 20]
    parts = [set(part[:, 0]) for part in [splits.X_train, splits.X_validation, splits.X_test]]
    assert not parts[0] & parts[1]
    assert not parts[0] & parts[2]
    assert not parts[1] & parts[2]
    assert len(set.union(*parts)) == 100
    for labels in (splits.y_train, splits.y_validation, splits.y_test):
        assert labels.mean() == 0.5
    np.testing.assert_array_equal(splits.X_train, split_data(X, y).X_train)


class ScoredClassifier:
    classes_ = np.array([0, 1])

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        return np.column_stack([1 - scores, scores])


def test_auc_and_ap_use_continuous_scores_despite_wrong_threshold_predictions():
    result = evaluate_model(ScoredClassifier(), [[0]] * 4, [0, 0, 1, 1])
    assert result["f1"] == 0
    assert result["roc_auc"] == 1
    assert result["average_precision"] == 1
    assert set(result) == set(METRICS)


@pytest.mark.parametrize("positive_label", [0, 1])
def test_decision_function_scores_are_oriented_to_positive_class(positive_label):
    X, y = dataset()
    model = RidgeClassifier().fit(X, y)
    scores = model.decision_function(X) * (1 if positive_label == 1 else -1)
    result = evaluate_model(model, X, y, positive_label=positive_label)
    assert result["roc_auc"] == pytest.approx(roc_auc_score(y == positive_label, scores))
    assert result["average_precision"] == pytest.approx(
        average_precision_score(y == positive_label, scores)
    )


def test_predict_only_model_has_nan_ranking_metrics():
    class PredictOnly:
        classes_ = np.array([0, 1])

        def predict(self, X):
            return [0, 1]

    result = evaluate_model(PredictOnly(), [[0], [1]], [0, 1])
    assert result["accuracy"] == 1
    assert np.isnan(result["roc_auc"])
    assert np.isnan(result["average_precision"])


def test_validation_preprocessing_uses_only_train_and_final_refits_on_development():
    X, y = dataset()
    X[0, 0] = np.nan
    splits = split_data(X, y)
    # Deliberately different means make fitting on the wrong partition visible.
    splits.X_validation = splits.X_validation + 1000
    estimator = LogisticRegression(max_iter=1000)
    result = validate_models(splits, models={"logistic": estimator})
    pipeline = result.pipelines["logistic"]
    np.testing.assert_allclose(
        pipeline["preprocessor"]["imputer"].statistics_, np.nanmedian(splits.X_train, axis=0)
    )
    assert not hasattr(estimator, "classes_")
    final = evaluate_final(result, splits)
    assert final.best_model_name == result.best_model_name
    np.testing.assert_allclose(
        final.best_pipeline["preprocessor"]["imputer"].statistics_,
        np.nanmedian(splits.X_development, axis=0),
    )
    # The validated pipeline remains fitted on the original 64%.
    np.testing.assert_allclose(
        pipeline["preprocessor"]["imputer"].statistics_, np.nanmedian(splits.X_train, axis=0)
    )


def test_all_eleven_models_complete_both_stages():
    X, y = dataset()
    splits = split_data(X, y)
    test_X, test_y = splits.X_test, splits.y_test
    splits.X_test = splits.y_test = None
    validation = validate_models(splits)
    best_name = max(validation.metrics, key=lambda row: row["f1"])["model"]
    # The final holdout must not be accessed by validation/selection.
    splits.X_test, splits.y_test = -test_X, test_y
    final = evaluate_final(validation, splits)
    assert final.best_model_name == best_name
    assert len(validation.metrics) == len(final.metrics) == len(build_models()) == 11
    for row in [*validation.metrics, *final.metrics]:
        assert all(0 <= row[metric] <= 1 for metric in METRICS)


@pytest.mark.parametrize("y", [[0] * 20, [0, 1, 2] * 10, [0] * 19 + [1]])
def test_invalid_targets_fail_with_clear_error(y):
    with pytest.raises(ValueError):
        split_data(np.zeros((len(y), 2)), y)


def test_string_labels_and_nondefault_positive_class():
    X, y = dataset()
    y = np.where(y == 1, "yes", "no")
    splits = split_data(X, y, positive_label="no")
    result = validate_models(
        splits, models={"ridge": RidgeClassifier()}, positive_label="no", selection_metric="roc_auc"
    )
    assert np.isfinite(result.metrics[0]["roc_auc"])
