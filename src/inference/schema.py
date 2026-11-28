"""Define the Adult Income application's raw input contract.

The schema combines approved domain constraints with categorical vocabularies
from the fitted reference pipeline. It validates and normalizes structured
user payloads, but does not load models, preprocess observations, or invoke
the classifier.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Literal


# Observed boundaries from the Adult Income EDA; application policy, not
# mathematical limits of the estimator.
NUMERIC_RANGES = MappingProxyType({
    "age": (17, 90),
    "education-num": (1, 16),
    "capital-gain": (0, 99999),
    "capital-loss": (0, 4356),
    "hours-per-week": (1, 99),
})

# Education is redundant in the pipeline, but both raw columns are required.
EDUCATION_TO_NUMBER = MappingProxyType({
    "Preschool": 1,
    "1st-4th": 2,
    "5th-6th": 3,
    "7th-8th": 4,
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-voc": 11,
    "Assoc-acdm": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Prof-school": 15,
    "Doctorate": 16,
})
NUMBER_TO_EDUCATION = MappingProxyType({
    number: category for category, number in EDUCATION_TO_NUMBER.items()
})
EDUCATION_FIELDS = ("education", "education-num")

# These columns are encoded by the promoted no_fnlwgt pipeline.
ENCODED_CATEGORICAL_FIELDS = (
    "workclass", "marital-status", "occupation", "relationship",
    "race", "sex", "native-country",
)
UNKNOWN_FIELDS = frozenset({"workclass", "occupation", "native-country"})
SENSITIVE_FIELDS = frozenset({"race", "sex"})
EXPECTED_RAW_FIELDS = frozenset(NUMERIC_RANGES) | frozenset(
    (*ENCODED_CATEGORICAL_FIELDS, "education")
)


@dataclass(frozen=True)
class FeatureSpec:
    """Describe a single raw field and its accepted application domain."""

    name: str
    kind: Literal["integer", "categorical"]
    minimum: int | None = None
    maximum: int | None = None
    categories: tuple[str, ...] = ()
    sensitive: bool = False

    @property
    def allows_unknown(self) -> bool:
        """Return whether an explicit Unknown category is supported."""
        return self.name in UNKNOWN_FIELDS and "Unknown" in self.categories


@dataclass(frozen=True)
class InputSchema:
    """Describe ordered pipeline fields and the education alternative."""

    fields: tuple[FeatureSpec, ...]
    education_alternatives: tuple[str, str] = EDUCATION_FIELDS

    @property
    def required_features(self) -> tuple[str, ...]:
        """Return all raw columns in the verified pipeline input order."""
        return tuple(field.name for field in self.fields)

    @property
    def required_user_fields(self) -> tuple[str, ...]:
        """Return mandatory user fields apart from the education pair.

        At least one education alternative must also be supplied.
        """
        return tuple(
            name for name in self.required_features
            if name not in self.education_alternatives
        )

    def feature(self, name: str) -> FeatureSpec:
        """Return the specification for a known field or raise KeyError."""
        for field in self.fields:
            if field.name == name:
                return field
        raise KeyError(name)


def _fitted_categories(transformer) -> dict[str, tuple[str, ...]]:
    """Extract and check categorical vocabularies from the fitted encoder."""
    categorical = [
        columns for name, _, columns in transformer.transformers_
        if name == "categorical"
    ]
    if len(categorical) != 1:
        raise ValueError("Expected one fitted categorical transformer.")
    columns = tuple(categorical[0])
    encoder = transformer.named_transformers_["categorical"].named_steps["encoder"]

    # Imputation passes an array to the encoder; column names belong to
    # the ColumnTransformer, not encoder.feature_names_in_.
    if (len(columns) != len(ENCODED_CATEGORICAL_FIELDS)
            or set(columns) != set(ENCODED_CATEGORICAL_FIELDS)
            or len(encoder.categories_) != len(columns)):
        raise ValueError("Fitted categorical fields do not match the application schema.")

    vocabularies = {}
    for name, values in zip(columns, encoder.categories_):
        categories = tuple(values)
        if not categories or any(
            not isinstance(value, str) or not value.strip() for value in categories
        ):
            raise ValueError(f"Invalid fitted vocabulary for {name}.")
        # Future case-insensitive normalization must be unambiguous.
        if len({value.strip().casefold() for value in categories}) != len(categories):
            raise ValueError(f"Ambiguous fitted vocabulary for {name}.")
        if name not in UNKNOWN_FIELDS and any(
            value.strip().casefold() == "unknown" for value in categories
        ):
            raise ValueError(f"Unsupported Unknown category for {name}.")
        vocabularies[name] = categories
    return vocabularies


def build_input_schema(model, metadata: Mapping[str, object]) -> InputSchema:
    """Build a schema from an already trusted, fitted pipeline and metadata.

    Reject incompatible releases rather than silently changing the application's
    input contract. Artifact integrity and deserialization belong to the loader.
    """
    raw = metadata.get("required_features")
    if not isinstance(raw, (list, tuple)) or any(
        not isinstance(name, str) for name in raw
    ):
        raise ValueError("Metadata must declare required_features as a sequence of names.")
    required = tuple(raw)
    if len(required) != len(EXPECTED_RAW_FIELDS) or set(required) != EXPECTED_RAW_FIELDS:
        raise ValueError("Metadata raw fields do not match the application schema.")

    try:
        preprocessor = model.named_steps["preprocess"]
        engineer = preprocessor.named_steps["feature_engineering"]
        transformer = preprocessor.named_steps["column_transformer"]
        fitted_required = tuple(engineer.required_features_)
        excluded = tuple(engineer.excluded_features)
        education_representation = engineer.education_representation
        vocabularies = _fitted_categories(transformer)
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError("Reference preprocessing is missing or not fitted.") from exc

    if fitted_required != required:
        raise ValueError("Metadata and fitted raw feature order do not match.")
    if excluded != ("fnlwgt",) or education_representation != "education-num":
        raise ValueError("Fitted feature representation is incompatible with the application schema.")

    # Education categories are absent from the encoder by design.
    vocabularies["education"] = tuple(EDUCATION_TO_NUMBER)
    fields = []
    for name in required:
        if name in NUMERIC_RANGES:
            minimum, maximum = NUMERIC_RANGES[name]
            fields.append(FeatureSpec(name, "integer", minimum, maximum))
        else:
            fields.append(FeatureSpec(
                name, "categorical", categories=vocabularies[name],
                sensitive=name in SENSITIVE_FIELDS,
            ))
    return InputSchema(tuple(fields))


class InputValidationError(ValueError):
    """Report invalid structured input with field-specific clarification messages."""

    def __init__(self, errors: Mapping[str, str]) -> None:
        """Preserve validation issues without echoing submitted field values."""
        self.errors = MappingProxyType(dict(errors))
        super().__init__("; ".join(
            f"{name}: {message}" for name, message in self.errors.items()
        ))


def _normalize_value(value: object, field: FeatureSpec) -> int | str:
    """Validate a field and return its canonical integer or category."""
    if field.kind == "integer":
        # Booleans are integer subclasses but are not numerical feature values.
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError("Expected an integer.")
        number = int(value)
        if field.minimum is None or field.maximum is None:
            raise ValueError("Integer field has no configured range.")
        if not field.minimum <= number <= field.maximum:
            raise ValueError(f"Expected an integer from {field.minimum} to {field.maximum}.")
        return number

    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected a nonempty category string.")
    # Only spelling case and surrounding whitespace are normalized here.
    normalized = value.strip().casefold()
    if normalized == "unknown" and not field.allows_unknown:
        raise ValueError("Unknown is not supported for this field.")
    for category in field.categories:
        if normalized == category.strip().casefold():
            return category
    raise ValueError("Unsupported category.")


def validate_input(
    payload: Mapping[str, object], schema: InputSchema
) -> dict[str, int | str]:
    """Return validated raw features in model order without mutating the input.

    Require all independent fields and at least one education representation.
    Reject extra fields, nulls, unsupported categories and conflicting education
    values. Only the redundant education field may be derived automatically.
    Raise InputValidationError with field-specific issues if validation fails.
    """
    if not isinstance(payload, Mapping):
        raise InputValidationError({"__input__": "Expected a mapping of feature names to values."})
    if any(not isinstance(name, str) for name in payload):
        raise InputValidationError({"__input__": "Feature names must be strings."})

    errors: dict[str, str] = {}
    normalized: dict[str, int | str] = {}
    required = schema.required_features

    # Report all missing independent fields and validate every supplied field.
    for name in schema.required_user_fields:
        if name not in payload:
            errors[name] = "Required field is missing."
    for name in required:
        if name in payload:
            try:
                normalized[name] = _normalize_value(payload[name], schema.feature(name))
            except ValueError as exc:
                errors[name] = str(exc)
    for name in payload:
        if name not in required:
            errors[name] = "Unexpected field."

    # Education is a known one-to-one mapping, not an imputed user attribute.
    category_name, number_name = schema.education_alternatives
    if category_name not in payload and number_name not in payload:
        errors[category_name] = "Provide education or education-num."
    elif category_name not in errors and number_name not in errors:
        if category_name in normalized and number_name in normalized:
            if EDUCATION_TO_NUMBER[normalized[category_name]] != normalized[number_name]:
                errors[number_name] = "Conflicts with education category."
        elif category_name in normalized:
            normalized[number_name] = EDUCATION_TO_NUMBER[normalized[category_name]]
        elif number_name in normalized:
            normalized[category_name] = NUMBER_TO_EDUCATION[normalized[number_name]]

    if errors:
        raise InputValidationError(errors)

    # Stable raw column order is required by the serialized pipeline contract.
    return {name: normalized[name] for name in required}
