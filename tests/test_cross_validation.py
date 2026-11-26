"""Verify fold-local preprocessing, independent thresholds and candidate metrics."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from src.data.preprocess import (
    PreprocessingConfig, build_preprocessor, encode_target, split_features_target,
)
from src.models import cross_validation as cv
from src.models.tasks import binary_classification as binary


@pytest.fixture
def training_data(sample_adult_df):
    """Create distinct synthetic groups with nonconsecutive row indices."""
    batches = []
    for number in range(12):
        batch = sample_adult_df.copy()
        batch["fnlwgt"] += number * 1_000_000
        batches.append(batch)
    raw = pd.concat(batches, ignore_index=True)
    X, y = split_features_target(encode_target(raw))
    X.index = y.index = pd.Index(np.arange(len(X)) * 2 + 10)
    return X, y


@pytest.fixture
def prepared(training_data):
    """Prepare reusable folds with the baseline feature configuration."""
    return cv.prepare_cv_folds(*training_data, PreprocessingConfig())


@pytest.mark.parametrize("config", [
    PreprocessingConfig(),
    PreprocessingConfig(numerical_scaling="robust", capital_transform="both"),
    PreprocessingConfig(interaction_features=True),
])
def test_fold_local_preprocessing(training_data, monkeypatch, config):
    """Check each preprocessor fits only its own fit rows and keeps feature names."""
    X, y = training_data
    fit_indices = []
    instances = []
    original_builder = cv.build_preprocessor

    def tracked_builder(config):
        """Record preprocessing fit inputs while delegating to the real pipeline."""
        pipeline = original_builder(config)
        instances.append(pipeline)
        original_fit_transform = pipeline.fit_transform

        def tracked_fit_transform(frame, *args, **kwargs):
            """Record raw fit row labels before fitting the actual transformer."""
            fit_indices.append(frame.index.copy())
            return original_fit_transform(frame, *args, **kwargs)

        monkeypatch.setattr(pipeline, "fit_transform", tracked_fit_transform)
        return pipeline

    monkeypatch.setattr(cv, "build_preprocessor", tracked_builder)
    folds = cv.prepare_cv_folds(X, y, config)
    assert len(instances) == len(fit_indices) == len(folds) == 3
    assert len({id(instance) for instance in instances}) == 3
    for fold, fitted_rows in zip(folds, fit_indices):
        assert fitted_rows.equals(fold.X_fit.index)
        assert fitted_rows.intersection(fold.X_calibration.index).empty
        assert fitted_rows.intersection(fold.X_scoring.index).empty
        assert fold.X_fit.columns.equals(fold.X_calibration.columns)
        assert fold.X_fit.columns.equals(fold.X_scoring.columns)
        assert fold.X_fit.index.equals(fold.y_fit.index)
        assert fold.X_calibration.index.equals(fold.y_calibration.index)
        assert fold.X_scoring.index.equals(fold.y_scoring.index)
        # Independently reproduce one fold's transformations from its raw rows.
        expected = build_preprocessor(config)
        expected.fit(X.loc[fitted_rows])
        np.testing.assert_allclose(
            fold.X_scoring.to_numpy(), expected.transform(X.loc[fold.X_scoring.index]),
        )


def test_prepare_does_not_modify_raw_inputs(training_data):
    """Preserve the raw training predictors and targets during preparation."""
    X, y = training_data
    before_X, before_y = X.copy(deep=True), y.copy(deep=True)
    cv.prepare_cv_folds(X, y, PreprocessingConfig())
    pd.testing.assert_frame_equal(X, before_X)
    pd.testing.assert_series_equal(y, before_y)


def test_candidate_matches_independent_fold_calculations(prepared):
    """Recompute each fold threshold and score using a fresh reference estimator."""
    prototype = LogisticRegression(max_iter=5000, random_state=42)
    result = cv.evaluate_cv_candidate(prototype, prepared)
    assert not hasattr(prototype, "classes_")
    for fold, actual in zip(prepared, result.folds):
        model = clone(prototype).fit(fold.X_fit, fold.y_fit)
        threshold = binary.optimal_threshold(
            fold.y_calibration, binary.positive_probabilities(model, fold.X_calibration),
        )
        probabilities = binary.positive_probabilities(model, fold.X_scoring)
        expected = binary.evaluate(fold.y_scoring, probabilities, threshold)
        expected["f1_default_050"] = binary.default_f1(fold.y_scoring, probabilities)
        assert actual.threshold == pytest.approx(threshold)
        assert actual.metrics == expected
        assert actual.scoring_rows == len(fold.y_scoring)
    for key, mean in result.mean_metrics.items():
        values = [fold.metrics[key] for fold in result.folds]
        assert mean == pytest.approx(np.mean(values))
        assert result.std_metrics[key] == pytest.approx(np.std(values, ddof=0))
    assert "confusion_matrix" not in result.mean_metrics


def test_scoring_labels_cannot_change_threshold(prepared):
    """Changing scoring labels must not influence fit or threshold calibration."""
    model = LogisticRegression(max_iter=5000, random_state=42)
    original = cv.evaluate_cv_candidate(model, prepared)
    changed_folds = tuple(replace(fold, y_scoring=1 - fold.y_scoring) for fold in prepared)
    changed = cv.evaluate_cv_candidate(model, changed_folds)
    assert [fold.threshold for fold in original.folds] == [fold.threshold for fold in changed.folds]
    assert original.mean_metrics["f1"] != changed.mean_metrics["f1"]


def test_repeated_candidate_evaluation_is_reproducible(prepared):
    """Reuse fold matrices without carrying model state into later evaluations."""
    model = LogisticRegression(max_iter=5000, random_state=42)
    snapshots = [fold.X_fit.copy(deep=True) for fold in prepared]
    assert cv.evaluate_cv_candidate(model, prepared) == cv.evaluate_cv_candidate(model, prepared)
    for fold, snapshot in zip(prepared, snapshots):
        pd.testing.assert_frame_equal(fold.X_fit, snapshot)


def test_missing_folds_rejected():
    """Reject a candidate evaluation without multiple prepared folds."""
    with pytest.raises(ValueError, match="at least two"):
        cv.evaluate_cv_candidate(LogisticRegression(), ())


def test_invalid_estimator_fails_entire_candidate(prepared):
    """Propagate fitting errors rather than scoring a partial CV result."""
    with pytest.raises(ValueError):
        cv.evaluate_cv_candidate(LogisticRegression(C=-1), prepared)
