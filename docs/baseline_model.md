# Baseline Model — Adult Income AI

**Established:** November 25, 2026

## 1. Purpose

This document records the initial model performance established during development of Adult Income AI. It serves as a permanent benchmark for subsequent feature engineering, algorithm selection, and hyperparameter optimization experiments.

The original reference model is preserved as a historical release; it is not the active reference.

## 2. Model Configuration

| Property | Value |
|---|---|
| Dataset | UCI Adult Income |
| Observations | 48,842 |
| Task | Binary classification |
| Positive class | Income >50K (1) |
| Algorithm | HistGradientBoostingClassifier |
| Preprocessing variant | baseline |
| Learning rate | 0.1 |
| Maximum leaf nodes | 15 |
| Maximum iterations | 150 |
| Random seed | 42 |
| Decision threshold | 0.3713238849891957 |
| Selection metric | Positive-class validation F1 |
| Training configurations | 54 |
| Successful configurations | 54 |

## 3. Baseline Performance

| Metric | Validation | Test |
|---|---:|---:|
| F1 | 0.733737 | 0.715003 |
| Precision | 0.694179 | 0.673477 |
| Recall | 0.778075 | 0.761986 |
| Accuracy | 0.864875 | 0.854670 |
| ROC-AUC | 0.930372 | 0.924719 |
| Average precision (PR-AUC) | 0.829579 | 0.820385 |

Validation F1 at the default 0.50 threshold: 0.706158.

The optimized threshold was selected exclusively using validation data.

### Test Confusion Matrix

| Actual / Predicted | <=50K | >50K |
|---|---:|---:|
| <=50K | 6,565 | 863 |
| >50K | 556 | 1,780 |

## 4. Preprocessing Variant Comparison

| Variant | Validation F1 | Test F1 |
|---|---:|---:|
| baseline | 0.733737 | 0.715003 |
| robust | 0.733737 | 0.715003 |
| interactions | 0.731417 | 0.719204 |

The baseline variant was selected through the existing deterministic validation-based ranking policy. Test performance was not used for selection.

## 5. Algorithm Comparison

Best validation F1 by algorithm family:

| Algorithm | Validation F1 |
|---|---:|
| HistGradientBoosting | 0.733737 |
| Random Forest | 0.722735 |
| Logistic Regression | 0.697598 |
| Extra Trees | 0.692910 |

## 6. Model Provenance

### Original Baseline Reference

- Execution ID: `ad7990374a3f453b9d879498349f714f`
- MLflow run ID: `fb19332ef80a4af49081c2c8434f16f3`
- Git commit: `e21b88715ddfd8d18968f0f4aab8cb76dcb07a8d`
- Promotion date: November 16, 2026
- Promotion type: explicit

### Reproduced Baseline Execution

- Execution ID: `15fb25b789db4fecb3d0c16992b83200`
- MLflow run ID: `3ff7fd7dceee4edc9701aed8442368ae`
- Git commit: `383f13cc998aebc4be83479214559a0fe48f6747`
- Execution date: November 25, 2026
- Training mode: full

## 7. Reproducibility Fingerprints

**Model SHA-256:**

`d52547575336d39e6aa3522d80aa3f7fc0a0e2d4617e24557de5de878657985b`

**Preprocessor SHA-256:**

`cb91a1953c516799f8b5a911469147b779b9d5d28fcc01af0513b5aa981f4ee7`

**Training configuration SHA-256:**

`c2f369cf6c978fd9176270632d068299ff028678f22f933e4d6bb3a1dc2ae1e7`

### Original Reference Dataset Fingerprints

| File | SHA-256 |
|---|---|
| X_train.csv | `bb6940896b6245623365bcf02fc01f967b4d800bc572850