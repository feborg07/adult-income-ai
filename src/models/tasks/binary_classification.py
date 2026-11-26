"""Target validation, prediction interpretation and binary model selection."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Binary classification defaults.
POSITIVE_CLASS = 1
DEFAULT_THRESHOLD = 0.5


# ============================================================
# 1. Target validation
# ============================================================

def validate_target(y: pd.Series, context: str) -> pd.Series:
    """Validate binary labels and return integer targets."""

    # Reject missing values and unsupported target labels.
    if y.isna().any() or not set(y.unique()).issubset({0, 1}):
        raise ValueError(f"Target must contain only 0/1: {context}")

    return y.astype(int)


def require_both_classes(y: pd.Series, context: str) -> None:
    """Require both classes in training and validation partitions."""

    if set(y.unique()) != {0, 1}:
        raise ValueError(f"Both target classes required: {context}")


# ============================================================
# 2. Probability extraction
# ============================================================

def positive_probabilities(model, X) -> np.ndarray:
    """Extract positive-class probabilities respecting estimator class order."""

    classes = np.asarray(model.classes_)

    # Verify that the fitted estimator supports binary classification.
    if classes.shape != (2,) or set(classes) != {0, 1}:
        raise ValueError("Estimator must expose binary classes 0 and 1.")

    probabilities = np.asarray(model.predict_proba(X))

    # Require one probability per class for each observation.
    if probabilities.shape != (len(X), 2):
        raise ValueError("Invalid prediction probability shape")

    # Validate probability values and normalization.
    if (
        not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
        or not np.allclose(probabilities.sum(axis=1), 1.0)
    ):
        raise ValueError("Invalid predicted probabilities")

    # Locate the positive class instead of assuming its column position.
    index = int(np.flatnonzero(classes == POSITIVE_CLASS)[0])

    return probabilities[:, index]


# ============================================================
# 3. Decision threshold optimization
# ============================================================

def optimal_threshold(
    y: pd.Series,
    probabilities: np.ndarray,
) -> float:
    """Find the threshold maximizing positive-class validation F1."""

    precision, recall, thresholds = precision_recall_curve(
        y, probabilities
    )

    # Fall back to the default threshold when none are available.
    if not len(thresholds):
        return DEFAULT_THRESHOLD

    # Calculate F1 for every available threshold.
    # Exclude the final precision-recall point, which has no threshold.
    f1 = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    )

    return float(thresholds[int(np.argmax(f1))])


# ============================================================
# 4. Model evaluation
# ============================================================

def evaluate(
    y: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    """Calculate binary classification metrics using a fixed threshold."""

    # Convert probabilities into binary predictions.
    predictions = (probabilities >= threshold).astype(int)

    # Threshold-dependent and probability-based metrics.
    return {
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "accuracy": float(accuracy_score(y, predictions)),

        # Preserve legacy behavior for single-class evaluation sets.
        "pr_auc": (
            float(average_precision_score(y, probabilities))
            if y.nunique() == 2 else None
        ),
        "roc_auc": (
            float(roc_auc_score(y, probabilities))
            if y.nunique() == 2 else None
        ),

        # Preserve a consistent 2x2 confusion matrix.
        "confusion_matrix": confusion_matrix(
            y, predictions, labels=[0, 1]
        ).tolist(),
    }


def default_f1(
    y: pd.Series,
    probabilities: np.ndarray,
) -> float:
    """Calculate positive-class F1 using the default decision threshold."""

    return float(
        f1_score(
            y,
            probabilities >= DEFAULT_THRESHOLD,
            zero_division=0,
        )
    )


# ============================================================
# 5. Model ranking
# ============================================================

def rank(row: dict) -> tuple[float, float, float]:
    """Rank candidates by validation F1, average precision and ROC-AUC."""

    # Tuple comparison preserves the established metric priority.
    return (
        row["val_f1"],
        row["val_pr_auc"],
        row["val_roc_auc"],
    )