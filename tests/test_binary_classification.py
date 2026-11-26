"""Unit tests for binary classification validation and evaluation."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from src.models.tasks import binary_classification as task

# ============================================================
# 1. Target validation
# ============================================================

def test_valid_target_is_converted_without_mutation():
    """Verify integer conversion without modifying the original target."""

    y = pd.Series([0.0, 1.0, 0.0])
    original = y.copy()

    result = task.validate_target(y, "baseline/train")

    assert result.tolist() == [0, 1, 0]
    assert pd.api.types.is_integer_dtype(result)

    # Confirm that the original Series remains unchanged.
    pd.testing.assert_series_equal(y, original)


@pytest.mark.parametrize("values", [
    [0, 2],
    [0, np.nan],
    ["0", "1"],
])
def test_invalid_target_is_rejected(values):
    """Reject unsupported labels, missing values and string targets."""

    with pytest.raises(
        ValueError, match="Target must contain only 0/1"
    ):
        task.validate_target(
            pd.Series(values), "baseline/train"
        )


def test_both_classes_required():
    """Verify that training and validation require both binary classes."""

    # A valid binary target must pass without raising an exception.
    task.require_both_classes(
        pd.Series([0, 1]), "baseline/val"
    )

    # Reject a partition containing only one class.
    with pytest.raises(
        ValueError, match="Both target classes required"
    ):
        task.require_both_classes(
            pd.Series([0, 0]), "baseline/val"
        )


# ============================================================
# 2. Probability extraction
# ============================================================

@pytest.mark.parametrize("classes,expected", [
    ([0, 1], [0.8, 0.3]),
    ([1, 0], [0.2, 0.7]),
])
def test_positive_probability_respects_class_order(
    classes, expected
):
    """Verify positive-class extraction regardless of class ordering."""

    # Simulate a fitted classifier with configurable class ordering.
    model = Mock()
    model.classes_ = np.array(classes)
    model.predict_proba.return_value = np.array([
        [0.2, 0.8],
        [0.7, 0.3],
    ])

    np.testing.assert_allclose(
        task.positive_probabilities(model, [[0], [1]]),
        expected,
    )


@pytest.mark.parametrize("classes,probabilities", [
    ([0, 2], [[0.2, 0.8]]),
    ([0, 1], [[1.0]]),
    ([0, 1], [[np.nan, 0.8]]),
    ([0, 1], [[-0.1, 1.1]]),
    ([0, 1], [[0.2, 0.3]]),
])
def test_invalid_probabilities_are_rejected(
    classes, probabilities
):
    """Reject invalid classes, probability shapes and probability values."""

    # Simulate malformed predictions from a classifier.
    model = Mock()
    model.classes_ = np.array(classes)
    model.predict_proba.return_value = np.array(probabilities)

    with pytest.raises(ValueError):
        task.positive_probabilities(model, [[0]])


# ============================================================
# 3. Decision threshold optimization
# ============================================================

def test_threshold_maximizes_f1():
    """Verify that the selected threshold maximizes positive-class F1."""

    y = pd.Series([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.4, 0.35, 0.7, 0.9])

    threshold = task.optimal_threshold(y, p)

    # Independently calculate F1 for every possible threshold.
    scores = [
        f1_score(y, p >= t)
        for t in np.unique(p)
    ]

    assert threshold == pytest.approx(0.35)

    assert f1_score(y, p >= threshold) == pytest.approx(
        max(scores)
    )


# ============================================================
# 4. Model evaluation
# ============================================================

def test_metrics_and_default_threshold():
    """Verify classification metrics against known expected values."""

    y = pd.Series([0, 0, 1, 1])
    p = np.array([0.1, 0.7, 0.8, 0.3])

    metrics = task.evaluate(y, p, 0.5)

    # Expected predictions contain one TP, TN, FP and FN.
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]

    for key in ("f1", "precision", "recall", "accuracy"):
        assert metrics[key] == pytest.approx(0.5)

    # Default-threshold F1 must match the evaluation result.
    assert task.default_f1(y, p) == pytest.approx(metrics["f1"])


def test_single_class_metrics_preserve_legacy_behavior():
    """Verify metric behavior when only one target class is present."""

    metrics = task.evaluate(
        pd.Series([0, 0]),
        np.array([0.1, 0.2]),
        0.5,
    )

    # Preserve the existing single-class evaluation policy.
    assert metrics["pr_auc"] is None
    assert metrics["roc_auc"] is None
    assert metrics["confusion_matrix"] == [[2, 0], [0, 0]]


# ============================================================
# 5. Model ranking
# ============================================================

def test_ranking_preserves_metric_priority():
    """Verify F1, average precision and ROC-AUC ranking priorities."""

    def row(f1, ap, auc):
        """Construct a candidate row containing validation metrics."""

        return {
            "val_f1": f1,
            "val_pr_auc": ap,
            "val_roc_auc": auc,
        }

    # F1 takes priority over all other metrics.
    assert task.rank(row(0.8, 0.6, 0.6)) > task.rank(
        row(0.7, 0.9, 0.9)
    )

    # Average precision resolves F1 ties.
    assert task.rank(row(0.8, 0.9, 0.6)) > task.rank(
        row(0.8, 0.8, 0.9)
    )

    # ROC-AUC resolves remaining ties.
    assert task.rank(row(0.8, 0.9, 0.8)) > task.rank(
        row(0.8, 0.9, 0.7)
    )