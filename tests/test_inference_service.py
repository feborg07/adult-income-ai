"""Exercise reference loading using a synthetic fitted pipeline and real bundle checks."""

from dataclasses import FrozenInstanceError
import json

import joblib
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.exceptions import InconsistentVersionWarning
from xgboost import XGBClassifier

from src.data.preprocess import FEATURE_COLUMNS, PreprocessingConfig, build_preprocessor
from src.inference import service
from src.models import promote


@pytest.fixture
def release(tmp_path, sample_adult_df):
    """Create a disposable release using actual preprocessing and synthetic XGBoost."""
    raw = sample_adult_df[FEATURE_COLUMNS].copy()
    prep = build_preprocessor(PreprocessingConfig(excluded_features=("fnlwgt",)))
    matrix = prep.fit_transform(raw)
    names = list(prep.get_feature_names_out())
    adapter = FunctionTransformer(pd.DataFrame, kw_args={"columns": names},
                                  validate=False, feature_names_out="one-to-one")
    named_matrix = pd.DataFrame(matrix, columns=names)
    adapter.fit(named_matrix)
    classifier = XGBClassifier(n_estimators=2, max_depth=2, n_jobs=1, random_state=42)
    classifier.fit(named_matrix, (sample_adult_df["income"] == ">50K").astype(int))
    model = Pipeline([("preprocess", prep), ("feature_names", adapter), ("classifier", classifier)])
    directory = tmp_path / "reference"
    directory.mkdir()
    joblib.dump(model, directory / promote.MODEL_FILENAME)
    metadata = {
        "model_sha256": promote.sha256_file(directory / promote.MODEL_FILENAME),
        "decision_threshold": 0.3635956048965454, "positive_class": 1,
        "model_family": "xgboost", "preprocessing_variant": "baseline",
        "representation": "no_fnlwgt", "educational_use_only": True,
        "required_features": [c for c in FEATURE_COLUMNS if c != "fnlwgt"],
    }
    evidence = {key: {} for key in promote.EVALUATION_FILES.values()}
    evidence["heldout"] = {"selected_candidate": "xgboost", "performance_gate": {"passed": True},
                          "candidates": [{"candidate": "xgboost", "family": "xgboost", "variant": "baseline",
                                          "representation": "no_fnlwgt", "threshold": metadata["decision_threshold"],
                                          "model_sha256": metadata["model_sha256"], "required_features": metadata["required_features"]}]}
    metadata.update(evidence=evidence, approval_sha256=promote.canonical_sha(evidence))
    _write_metadata(directory, metadata)
    promote.write_evaluation_bundle(directory / "evaluation", metadata)
    return directory, metadata


def _write_metadata(directory, metadata):
    """Write the synthetic metadata for controlled failure tests."""
    (directory / promote.METADATA_FILENAME).write_text(json.dumps(metadata), encoding="utf-8")


def _forbid_deserialization(monkeypatch):
    """Make an integrity failure test fail if joblib is called prematurely."""
    def forbidden(*args, **kwargs):
        """Signal unexpected deserialization."""
        pytest.fail("Unverified bytes reached joblib.load")
    monkeypatch.setattr(service.joblib, "load", forbidden)


def test_load_verified_release(release, monkeypatch, sample_adult_df):
    """Load a real fitted pipeline exactly once and preserve release bytes."""
    directory, metadata = release
    before = {p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    original = joblib.load
    calls = []

    def counted(*args, **kwargs):
        """Count actual deserializations."""
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(service.joblib, "load", counted)
    loaded = service.load_reference(directory)
    assert calls == [1]
    assert loaded.model_sha256 == metadata["model_sha256"]
    assert loaded.approval_sha256 == metadata["approval_sha256"]
    assert loaded.threshold == metadata["decision_threshold"]
    assert loaded.positive_class == 1
    assert loaded.schema.required_features == tuple(metadata["required_features"])
    assert loaded.model_family == "xgboost" and loaded.preprocessing_variant == "baseline"
    assert loaded.representation == "no_fnlwgt"
    # Verify the deserialized adapter and classifier can consume a raw row.
    raw = sample_adult_df[list(loaded.schema.required_features)].iloc[:1]
    assert loaded.model.predict_proba(raw).shape == (1, 2)
    assert calls == [1]
    with pytest.raises(FrozenInstanceError):
        loaded.threshold = 0.5
    assert before == {p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()}


@pytest.mark.parametrize("filename", [promote.MODEL_FILENAME, promote.METADATA_FILENAME,
                                     "evaluation/manifest.json", "evaluation/heldout.json"])
def test_missing_file(release, monkeypatch, filename):
    """Reject missing model, metadata or required evidence before unpickling."""
    directory, _ = release
    (directory / filename).unlink()
    _forbid_deserialization(monkeypatch)
    with pytest.raises(service.ReferenceLoadError, match="dvc pull"):
        service.load_reference(directory)


def test_corrupted_model(release, monkeypatch):
    """Reject a model whose bytes differ from approved metadata."""
    directory, _ = release
    (directory / promote.MODEL_FILENAME).write_bytes(b"corrupted")
    _forbid_deserialization(monkeypatch)
    with pytest.raises(service.ReferenceLoadError, match="SHA-256 mismatch"):
        service.load_reference(directory)


def test_corrupted_evidence(release, monkeypatch):
    """Reject a tampered evaluation bundle even when model bytes are unchanged."""
    directory, _ = release
    (directory / "evaluation/heldout.json").write_text("{}")
    _forbid_deserialization(monkeypatch)
    with pytest.raises(service.ReferenceLoadError, match="manifest mismatch"):
        service.load_reference(directory)


@pytest.mark.parametrize("field,value", [
    ("decision_threshold", .5), ("decision_threshold", "0.36"),
    ("decision_threshold", -0.1), ("decision_threshold", float("nan")),
    ("decision_threshold", True), ("positive_class", 0), ("positive_class", True),
    ("representation", "control"), ("model_family", "random_forest"),
    ("preprocessing_variant", "robust"), ("required_features", ["age"]),
    ("educational_use_only", False),
])
def test_changed_metadata(release, monkeypatch, field, value):
    """Reject unsafe or inconsistent inference metadata before deserialization."""
    directory, metadata = release
    metadata[field] = value
    _write_metadata(directory, metadata)
    _forbid_deserialization(monkeypatch)
    with pytest.raises(service.ReferenceLoadError):
        service.load_reference(directory)


def test_model_changed_after_verification(release, monkeypatch):
    """Hash the exact bytes being loaded, even after earlier bundle verification."""
    directory, _ = release
    original = service.reference_metadata

    def changed(*args, **kwargs):
        """Simulate a changed model after the first integrity check."""
        metadata = original(*args, **kwargs)
        (directory / promote.MODEL_FILENAME).write_bytes(b"changed")
        return metadata

    monkeypatch.setattr(service, "reference_metadata", changed)
    _forbid_deserialization(monkeypatch)
    with pytest.raises(service.ReferenceLoadError, match="SHA-256 mismatch"):
        service.load_reference(directory)


@pytest.mark.parametrize("problem", ["wrong_type", "wrong_steps", "classes", "schema", "adapter", "count"])
def test_incompatible_pipeline(release, monkeypatch, problem):
    """Reject incompatible deserialized pipeline structures and fitted contracts."""
    directory, _ = release
    model = joblib.load(directory / promote.MODEL_FILENAME)
    if problem == "wrong_type":
        model = object()
    elif problem == "wrong_steps":
        model.steps.pop(1)
    elif problem == "classes":
        model.named_steps["classifier"].n_classes_ = 3
    elif problem == "schema":
        model.named_steps["preprocess"].named_steps["feature_engineering"].excluded_features = ()
    elif problem == "adapter":
        model.named_steps["feature_names"].kw_args["columns"] = ["bad"]
    else:
        # Alter reported booster feature count without fitting a replacement.
        def wrong_count(self):
            """Report an incompatible feature count."""
            return 1
        monkeypatch.setattr(XGBClassifier, "n_features_in_", property(wrong_count))

    def fake_load(*args, **kwargs):
        """Return the deliberately incompatible model."""
        return model

    monkeypatch.setattr(service.joblib, "load", fake_load)
    with pytest.raises(service.ReferenceLoadError):
        service.load_reference(directory)


def test_version_warning(release, monkeypatch):
    """Treat sklearn serialization version mismatches as a failed startup."""
    import warnings
    directory, _ = release

    def incompatible(*args, **kwargs):
        """Simulate sklearn detecting a different serialization version."""
        warnings.warn(InconsistentVersionWarning(estimator_name="Pipeline",
                      current_sklearn_version="2", original_sklearn_version="1"))

    monkeypatch.setattr(service.joblib, "load", incompatible)
    with pytest.raises(service.ReferenceLoadError) as caught:
        service.load_reference(directory)
    assert isinstance(caught.value.__cause__, InconsistentVersionWarning)


def test_deserialization_error(release, monkeypatch):
    """Report a failed deserialization with its original diagnostic attached."""
    directory, _ = release

    def broken(*args, **kwargs):
        """Simulate an incompatible serialized artifact."""
        raise ImportError("missing module")

    monkeypatch.setattr(service.joblib, "load", broken)
    with pytest.raises(service.ReferenceLoadError, match="missing module") as caught:
        service.load_reference(directory)
    assert isinstance(caught.value.__cause__, ImportError)


def test_missing_reference(tmp_path, monkeypatch):
    """Fail clearly when DVC has not restored the release directory."""
    _forbid_deserialization(monkeypatch)
    with pytest.raises(service.ReferenceLoadError, match="dvc pull"):
        service.load_reference(tmp_path / "missing")


def test_invalid_metadata_json(release, monkeypatch):
    """Reject unreadable metadata before unpickling the artifact."""
    directory, _ = release
    (directory / promote.METADATA_FILENAME).write_text("{invalid", encoding="utf-8")
    _forbid_deserialization(monkeypatch)
    with pytest.raises(service.ReferenceLoadError) as caught:
        service.load_reference(directory)
    assert isinstance(caught.value.__cause__, json.JSONDecodeError)
