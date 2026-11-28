"""Exercise the actual DVC reference without fitting or selecting any model."""

from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from src.data.drift_generator import DriftConfig, load_reference_test_data
from src.inference.schema import InputValidationError, validate_input
from src.inference.service import DEFAULT_REFERENCE, load_reference, predict
from src.models.promote import evidence_row, sha256_file

ROOT = Path(__file__).resolve().parents[1]
# Software regression tolerance, not a new model selection or promotion gate.
F1_REGRESSION_TOLERANCE = 0.005
V2_SHA256 = "e3d91ddbef536965481f3b614ee44c55f1f7960290849688e9d178a7f35a2711"


@pytest.fixture(scope="module")
def approved_reference():
    """Load the actual approved release; missing artifacts must fail, not skip."""
    return load_reference(DEFAULT_REFERENCE)


@pytest.fixture
def example_payload():
    """Return the synthetic observation verified during Task 5.1."""
    return {
        "age": 42, "education": "Bachelors", "capital-gain": 0,
        "capital-loss": 0, "hours-per-week": 45, "workclass": "Private",
        "marital-status": "Married-civ-spouse", "occupation": "Exec-managerial",
        "relationship": "Husband", "race": "White", "sex": "Male",
        "native-country": "United-States",
    }


@pytest.mark.parametrize("changes", [
    {},
    {"age": 22, "education": "HS-grad", "occupation": "Other-service",
     "marital-status": "Never-married", "relationship": "Own-child", "hours-per-week": 20},
    {"age": 60, "education": "Masters", "capital-gain": 5000,
     "workclass": "Self-emp-inc", "hours-per-week": 55},
    {"workclass": "Unknown", "occupation": "Unknown", "native-country": "Unknown"},
])
def test_reference_prediction_contract(approved_reference, example_payload, changes):
    """Check actual probabilities, reduced-input parity and structured output."""
    reference = approved_reference
    payload = {**example_payload, **changes}
    original = payload.copy()
    validated = validate_input(payload, reference.schema)
    reduced = pd.DataFrame([validated], columns=list(reference.schema.required_features))
    full = reduced.assign(fnlwgt=180000)
    probabilities = reference.model.predict_proba(reduced)
    np.testing.assert_allclose(probabilities, reference.model.predict_proba(full), rtol=0, atol=1e-7)
    assert probabilities.shape == (1, 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    score = float(probabilities[0, list(reference.model.classes_).index(1)])
    result = predict(reference, payload)
    assert type(result.predicted_class) is int
    assert type(result.positive_class_score) is float
    assert result.positive_class_score == pytest.approx(score, abs=1e-7, rel=0)
    assert result.predicted_class == int(score >= reference.threshold)
    assert result.predicted_income == (">50K" if result.predicted_class else "<=50K")
    assert result.threshold == reference.threshold
    assert result.model_sha256 == reference.model_sha256
    assert result.approval_sha256 == reference.approval_sha256
    assert result == predict(reference, payload)
    assert payload == original
    assert json.loads(json.dumps(asdict(result), allow_nan=False))["predicted_class"] == result.predicted_class


def test_reference_education_parity(approved_reference, example_payload):
    """Both equivalent education representations must reach the same prediction."""
    numeric = example_payload.copy()
    numeric.pop("education")
    numeric["education-num"] = 13
    assert predict(approved_reference, numeric) == predict(approved_reference, example_payload)


def test_reference_v2_known_score(approved_reference, example_payload):
    """Anchor the verified v2 example without pinning future approved releases to v2."""
    if approved_reference.model_sha256 != V2_SHA256:
        pytest.skip("The recorded example score belongs specifically to model-v2.0.0.")
    result = predict(approved_reference, example_payload)
    assert result.positive_class_score == pytest.approx(0.8036101460456848, abs=1e-6, rel=0)
    assert result.threshold == 0.3635956048965454
    assert result.predicted_class == 1


@pytest.mark.parametrize("field,value", [("race", None), ("age", 91), ("occupation", "unsupported")])
def test_reference_invalid_input(approved_reference, example_payload, field, value, monkeypatch):
    """Invalid input must be rejected before invoking the actual model."""
    def forbidden(*args, **kwargs):
        """Fail if a malformed request reaches the prediction pipeline."""
        pytest.fail("Invalid input reached predict_proba")

    monkeypatch.setattr(approved_reference.model, "predict_proba", forbidden)
    with pytest.raises(InputValidationError) as exc:
        predict(approved_reference, {**example_payload, field: value})
    assert field in exc.value.errors


def test_reference_historical_f1_regression(approved_reference):
    """Reproduce approved historical test F1 using fixed model, split and threshold."""
    heldout = json.loads((DEFAULT_REFERENCE / "evaluation/heldout.json").read_text(encoding="utf-8"))
    selected = evidence_row(heldout, heldout["selected_candidate"])
    assert selected["model_sha256"] == approved_reference.model_sha256
    paths = [ROOT / "data/raw" / name for name in ("adult.data", "adult.test")]
    # Check the raw dataset against the approved release before reconstructing rows.
    assert {path.name: sha256_file(path) for path in paths} == heldout["raw_source_sha256"]
    X, y = load_reference_test_data(paths, DriftConfig())
    assert len(X) == len(y) == selected["test_rows"] == heldout["test_rows"]
    assert set(y.unique()) == {0, 1}
    # Historical records contain real missing values; the fitted preprocessor owns
    # their handling. Do not apply the stricter conversational input validator.
    probabilities = approved_reference.model.predict_proba(X[list(approved_reference.schema.required_features)])
    assert probabilities.shape == (len(y), 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    positive = list(approved_reference.model.classes_).index(approved_reference.positive_class)
    labels = (probabilities[:, positive] >= approved_reference.threshold).astype(int)
    score = f1_score(y, labels, pos_label=1)
    expected = selected["test_f1"]
    assert 0 <= expected <= 1
    assert score >= expected - F1_REGRESSION_TOLERANCE, (
        f"Approved reference F1 regression: {score:.6f} vs {expected:.6f}"
    )
    assert score == pytest.approx(expected, abs=F1_REGRESSION_TOLERANCE, rel=0)


def test_reference_standalone_runtime(approved_reference, example_payload, tmp_path):
    """Run the real service in a data-free workspace without MLflow or Optuna imports."""
    # Copy code and the approved reference only; do not copy data or experiment files.
    for name in ("src", "scripts"):
        shutil.copytree(ROOT / name, tmp_path / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    isolated_reference = tmp_path / "models/reference"
    shutil.copytree(DEFAULT_REFERENCE, isolated_reference)
    before = {p.relative_to(isolated_reference): sha256_file(p) for p in isolated_reference.rglob("*") if p.is_file()}
    code = '''
import importlib.abc
import json
import sys
from dataclasses import asdict
from pathlib import Path

class BlockTrainingDependencies(importlib.abc.MetaPathFinder):
    """Reject imports of training-only experiment dependencies."""
    def find_spec(self, fullname, path=None, target=None):
        """Prevent the isolated inference process from importing MLflow or Optuna."""
        if fullname.split('.')[0] in {'mlflow', 'optuna'}:
            raise ImportError(f'Training dependency forbidden: {fullname}')
        return None

sys.meta_path.insert(0, BlockTrainingDependencies())
sys.path.insert(0, str(Path.cwd()))
from src.inference.service import load_reference, predict
reference = load_reference()
print(json.dumps(asdict(predict(reference, json.loads(sys.stdin.read()))), allow_nan=False))
'''
    completed = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path,
                               input=json.dumps(example_payload), capture_output=True,
                               text=True, timeout=90, check=False)
    assert completed.returncode == 0, completed.stderr
    actual = json.loads(completed.stdout)
    expected = json.loads(json.dumps(asdict(predict(approved_reference, example_payload))))
    assert actual == expected
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "mlflow.db").exists()
    assert not (tmp_path / "models/optimization").exists()
    after = {p.relative_to(isolated_reference): sha256_file(p) for p in isolated_reference.rglob("*") if p.is_file()}
    assert before == after
