"""Load a trusted, approved inference reference without training or data access."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import warnings

import joblib
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from scripts.manage_reference import reference_metadata
from src.models.promote import MODEL_FILENAME, rate, evidence_row
from src.inference.schema import InputSchema, build_input_schema

DEFAULT_REFERENCE = Path(__file__).resolve().parents[2] / "models" / "reference"


class ReferenceLoadError(RuntimeError):
    """Indicate that the approved reference cannot be safely used for inference."""


@dataclass(frozen=True)
class LoadedReference:
    """Hold one loaded pipeline and its verified inference contract.

    Keep this object for the service lifetime; treat its fitted model as read-only.
    The model SHA-256 identifies the release without guessing a Git tag.
    """

    model: Pipeline
    schema: InputSchema
    threshold: float
    positive_class: int
    model_family: str
    preprocessing_variant: str
    representation: str
    model_sha256: str
    approval_sha256: str


def _verify_contract_metadata(metadata: dict) -> None:
    """Bind inference-critical metadata to the approved held-out candidate."""
    rate(metadata["decision_threshold"])
    if type(metadata["positive_class"]) is not int or metadata["positive_class"] != 1:
        raise ValueError("Reference positive class must be integer 1.")
    if metadata["representation"] != "no_fnlwgt":
        raise ValueError("Unsupported reference representation.")
    heldout = metadata["evidence"]["heldout"]
    selected = evidence_row(heldout, heldout["selected_candidate"])
    # The manifest verifies evidence bytes, not arbitrary top-level metadata.
    for key, evidence_key in (
        ("model_sha256", "model_sha256"),
        ("decision_threshold", "threshold"),
        ("model_family", "family"),
        ("preprocessing_variant", "variant"),
        ("representation", "representation"),
        ("required_features", "required_features"),
    ):
        if metadata[key] != selected[evidence_key]:
            raise ValueError(f"Reference metadata disagrees with approved evidence: {key}.")
    if heldout["performance_gate"]["passed"] is not True:
        raise ValueError("Reference held-out performance gate did not pass.")
    if metadata["educational_use_only"] is not True:
        raise ValueError("Reference educational-use restriction is missing.")


def _verify_pipeline(model: Pipeline, metadata: dict) -> InputSchema:
    """Check fitted pipeline structure, class order, feature adapter and raw schema."""
    if not isinstance(model, Pipeline) or not callable(getattr(model, "predict_proba", None)):
        raise ValueError("Reference must be a pipeline with predict_proba().")
    if tuple(model.named_steps) != ("preprocess", "feature_names", "classifier"):
        raise ValueError("Unexpected reference pipeline steps.")
    classifier = model.named_steps["classifier"]
    check_is_fitted(classifier)
    if list(classifier.classes_) != [0, 1] or list(model.classes_) != [0, 1]:
        raise ValueError("Reference class order must be [0, 1].")
    schema = build_input_schema(model, metadata)
    names = list(model.named_steps["preprocess"].get_feature_names_out())
    adapter = model.named_steps["feature_names"]
    if not names or len(names) != len(set(names)) or list(adapter.kw_args["columns"]) != names:
        raise ValueError("Reference feature adapter does not match preprocessing output.")
    if classifier.n_features_in_ != len(names):
        raise ValueError("Reference classifier feature count does not match preprocessing output.")
    classifier_names = getattr(classifier, "feature_names_in_", None)
    if classifier_names is not None and list(classifier_names) != names:
        raise ValueError("Reference classifier feature names do not match preprocessing output.")
    return schema


def load_reference(reference_dir: Path = DEFAULT_REFERENCE) -> LoadedReference:
    """Verify and load the DVC-restored reference once per explicit initialization.

    Only use trusted local releases. Checksums establish consistency, not publisher
    authenticity. Restart the service after an approved promotion; do not run
    promotion or DVC checkout concurrently with service initialization.
    """
    directory = Path(reference_dir)
    try:
        metadata = reference_metadata(directory, require_bundle=True)
        _verify_contract_metadata(metadata)
        # Deserialize the exact bytes checked, not a path reopened by joblib.
        payload = (directory / MODEL_FILENAME).read_bytes()
        if sha256(payload).hexdigest() != metadata["model_sha256"]:
            raise ValueError("Reference model SHA-256 mismatch before deserialization.")
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            model = joblib.load(BytesIO(payload))
        schema = _verify_pipeline(model, metadata)
        return LoadedReference(
            model=model, schema=schema,
            threshold=float(metadata["decision_threshold"]),
            positive_class=metadata["positive_class"],
            model_family=metadata["model_family"],
            preprocessing_variant=metadata["preprocessing_variant"],
            representation=metadata["representation"],
            model_sha256=metadata["model_sha256"],
            approval_sha256=metadata["approval_sha256"],
        )
    except Exception as exc:
        # Fail closed; preserve the underlying diagnostic for the operator.
        raise ReferenceLoadError(
            f"Cannot load approved reference at {directory}: {exc}. "
            "Verify the trusted DVC release (dvc pull models/reference.dvc) "
            "and its pinned runtime dependencies."
        ) from exc
