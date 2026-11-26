"""Prepare leakage-aware folds and evaluate binary classifier candidates."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.data.preprocess import (
    PreprocessingConfig,
    build_preprocessor,
    create_cv_folds,
)
from src.models.tasks import binary_classification as binary


@dataclass(frozen=True)
class PreparedFold:
    """Hold one fold's independently transformed fit, calibration and scoring data."""

    number: int
    X_fit: pd.DataFrame
    y_fit: pd.Series
    X_calibration: pd.DataFrame
    y_calibration: pd.Series
    X_scoring: pd.DataFrame
    y_scoring: pd.Series
    preprocessor: object


@dataclass(frozen=True)
class FoldResult:
    """Store one fold's fixed-threshold scores and calibration threshold."""

    number: int
    fit_rows: int
    calibration_rows: int
    scoring_rows: int
    threshold: float
    metrics: dict


@dataclass(frozen=True)
class CVResult:
    """Store fold results and unweighted means and population standard deviations."""

    folds: tuple[FoldResult, ...]
    mean_metrics: dict[str, float]
    std_metrics: dict[str, float]


def _named_frame(preprocessor, X: pd.DataFrame, values) -> pd.DataFrame:
    """Preserve the fold-local feature names and raw row index after transformation."""
    return pd.DataFrame(
        values, columns=preprocessor.get_feature_names_out(), index=X.index,
    )


def prepare_cv_folds(
    X: pd.DataFrame,
    y: pd.Series,
    config: PreprocessingConfig,
    *,
    n_splits: int = 3,
    calibration_size: float = 0.20,
    random_state: int = 42,
) -> tuple[PreparedFold, ...]:
    """Fit fresh preprocessors on fold fit rows and transform all three subsets.

    Pass only the original raw training partition. The resulting matrices may
    be reused across candidates with identical preprocessing and fold settings.
    Neither the original validation partition nor the test partition is needed.
    """
    indices = create_cv_folds(
        X, y, n_splits=n_splits, calibration_size=calibration_size,
        random_state=random_state,
    )
    folds = []
    for number, (fit, calibration, scoring) in enumerate(indices, start=1):
        # Use positional indexing, even when raw row labels are nonconsecutive.
        X_fit, X_cal, X_score = (X.iloc[part] for part in (fit, calibration, scoring))
        preprocessor = build_preprocessor(config)
        fit_values = preprocessor.fit_transform(X_fit)
        # No learned transformation may be fitted on calibration or scoring rows.
        folds.append(PreparedFold(
            number=number,
            X_fit=_named_frame(preprocessor, X_fit, fit_values),
            y_fit=y.iloc[fit].copy(),
            X_calibration=_named_frame(preprocessor, X_cal, preprocessor.transform(X_cal)),
            y_calibration=y.iloc[calibration].copy(),
            X_scoring=_named_frame(preprocessor, X_score, preprocessor.transform(X_score)),
            y_scoring=y.iloc[scoring].copy(),
            preprocessor=preprocessor,
        ))
    return tuple(folds)


def evaluate_cv_candidate(
    estimator, folds: tuple[PreparedFold, ...], *, on_fold=None,
) -> CVResult:
    """Clone and fit a candidate per fold, calibrate thresholds and score independently.

    Each threshold is specific to its fold. Do not use their mean as a deployment
    threshold: the eventual refitted model needs its own threshold calibration.
    Failed folds raise an error rather than returning a partial objective.
    An optional callback receives the prepared fold, fitted model and result
    for artifact logging without refitting the estimator.
    """
    if len(folds) < 2:
        raise ValueError("Cross-validation requires at least two prepared folds.")

    results = []
    for fold in folds:
        # A fresh estimator prevents learned state from carrying between folds.
        model = clone(estimator)
        model.fit(fold.X_fit, fold.y_fit)
        calibration_probabilities = binary.positive_probabilities(model, fold.X_calibration)
        threshold = binary.optimal_threshold(fold.y_calibration, calibration_probabilities)

        # Freeze the threshold before looking at scoring labels.
        scoring_probabilities = binary.positive_probabilities(model, fold.X_scoring)
        metrics = binary.evaluate(fold.y_scoring, scoring_probabilities, threshold)
        metrics["f1_default_050"] = binary.default_f1(fold.y_scoring, scoring_probabilities)
        result = FoldResult(
            number=fold.number, fit_rows=len(fold.y_fit),
            calibration_rows=len(fold.y_calibration), scoring_rows=len(fold.y_scoring),
            threshold=threshold, metrics=metrics,
        )
        if on_fold is not None:
            on_fold(fold, model, result)
        results.append(result)

    # Aggregate scalar metrics only; confusion matrices remain fold-level evidence.
    keys = [key for key in results[0].metrics if key != "confusion_matrix"]
    means, deviations = {}, {}
    for key in keys:
        values = np.asarray([result.metrics[key] for result in results], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Invalid cross-validation metric: {key}")
        means[key] = float(np.mean(values))
        deviations[key] = float(np.std(values, ddof=0))

    return CVResult(tuple(results), means, deviations)
