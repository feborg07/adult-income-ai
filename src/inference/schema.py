"""Define the Adult Income application's raw input contract.

The schema combines approved domain constraints with categorical vocabularies
from the fitted reference pipeline. It does not load models, validate user
payloads, transform observations, or invoke the classifier.
"""

from collections.abc import Mapping
from dataclasses import dataclass
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
