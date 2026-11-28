"""Test the inference contract against synthetic fitted preprocessing."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.data.preprocess import FEATURE_COLUMNS, PreprocessingConfig, build_preprocessor
from src.inference.schema import (
    EDUCATION_TO_NUMBER,
    InputValidationError,
    NUMBER_TO_EDUCATION,
    build_input_schema,
    validate_input,
)


@pytest.fixture
def schema_artifacts(sample_adult_df):
    """Fit the actual preprocessing implementation on synthetic observations."""
    raw = sample_adult_df[FEATURE_COLUMNS].copy()
    for field in ("workclass", "occupation", "native-country"):
        raw.loc[raw.index[0], field] = "Unknown"
    preprocessor = build_preprocessor(PreprocessingConfig(excluded_features=("fnlwgt",)))
    preprocessor.fit(raw)
    pipeline = Pipeline([("preprocess", preprocessor)])
    required = [name for name in FEATURE_COLUMNS if name != "fnlwgt"]
    return pipeline, {"required_features": required}


@pytest.fixture
def schema(schema_artifacts):
    """Return the schema built from the fitted synthetic pipeline."""
    return build_input_schema(*schema_artifacts)


def test_raw_and_user_contract(schema, schema_artifacts):
    """Preserve the 13 raw fields and require 11 fields plus education."""
    assert schema.required_features == tuple(schema_artifacts[1]["required_features"])
    assert len(schema.required_features) == 13
    assert len(schema.required_user_fields) == 11
    assert schema.education_alternatives == ("education", "education-num")
    assert "fnlwgt" not in schema.required_features
    assert not set(schema.education_alternatives) & set(schema.required_user_fields)


@pytest.mark.parametrize("name,minimum,maximum", [
    ("age", 17, 90), ("education-num", 1, 16),
    ("capital-gain", 0, 99999), ("capital-loss", 0, 4356),
    ("hours-per-week", 1, 99),
])
def test_numeric_domains(schema, name, minimum, maximum):
    """Expose the approved integer ranges without categorical choices."""
    field = schema.feature(name)
    assert field.kind == "integer"
    assert (field.minimum, field.maximum) == (minimum, maximum)
    assert field.categories == ()


def test_education_mapping(schema):
    """Verify all 16 education categories and the reversible mapping."""
    expected = (
        "Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th",
        "11th", "12th", "HS-grad", "Some-college", "Assoc-voc",
        "Assoc-acdm", "Bachelors", "Masters", "Prof-school", "Doctorate",
    )
    assert schema.feature("education").categories == expected
    assert dict(EDUCATION_TO_NUMBER) == dict(zip(expected, range(1, 17)))
    assert dict(NUMBER_TO_EDUCATION) == dict(zip(range(1, 17), expected))


def test_fitted_vocabularies(schema, schema_artifacts):
    """Read the original fitted categories, including infrequent categories."""
    transformer = schema_artifacts[0].named_steps["preprocess"].named_steps["column_transformer"]
    columns = next(cols for name, _, cols in transformer.transformers_ if name == "categorical")
    encoder = transformer.named_transformers_["categorical"].named_steps["encoder"]
    for name, values in zip(columns, encoder.categories_):
        field = schema.feature(name)
        assert field.kind == "categorical"
        assert field.categories == tuple(values)
        assert field.minimum is None and field.maximum is None


def test_missing_and_sensitive_policy(schema):
    """Permit explicit Unknown only for supported fields and flag demographics."""
    assert {field.name for field in schema.fields if field.allows_unknown} == {
        "workclass", "occupation", "native-country",
    }
    assert {field.name for field in schema.fields if field.sensitive} == {"race", "sex"}
    assert "race" in schema.required_user_fields and "sex" in schema.required_user_fields


def test_schema_is_immutable(schema):
    """Prevent accidental changes to the shared contract and mapping."""
    with pytest.raises(FrozenInstanceError):
        schema.fields = ()
    with pytest.raises(FrozenInstanceError):
        schema.feature("age").minimum = 0
    with pytest.raises(TypeError):
        EDUCATION_TO_NUMBER["Bachelors"] = 99
    with pytest.raises(KeyError):
        schema.feature("fnlwgt")


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "reordered", "invalid"])
def test_incompatible_metadata(schema_artifacts, change):
    """Reject malformed or incompatible metadata instead of guessing fields."""
    model, metadata = schema_artifacts
    required = list(metadata["required_features"])
    if change == "missing":
        required.pop()
    elif change == "extra":
        required.append("fnlwgt")
    elif change == "duplicate":
        required[-1] = required[0]
    elif change == "reordered":
        required.reverse()
    else:
        required = "age"
    with pytest.raises(ValueError, match="Metadata"):
        build_input_schema(model, {"required_features": required})


def test_missing_metadata(schema_artifacts):
    """Require an explicit metadata declaration of the raw contract."""
    with pytest.raises(ValueError, match="Metadata"):
        build_input_schema(schema_artifacts[0], {})


def test_unfitted_pipeline(schema_artifacts):
    """Reject an unfitted preprocessor rather than inventing categories."""
    model = Pipeline([("preprocess", build_preprocessor())])
    with pytest.raises(ValueError, match="not fitted"):
        build_input_schema(model, schema_artifacts[1])


@pytest.mark.parametrize("attribute,value", [
    ("excluded_features", ()), ("education_representation", "both"),
])
def test_incompatible_representation(schema_artifacts, attribute, value):
    """Reject a fitted feature representation outside the approved contract."""
    model, metadata = schema_artifacts
    engineer = model.named_steps["preprocess"].named_steps["feature_engineering"]
    setattr(engineer, attribute, value)
    with pytest.raises(ValueError, match="representation"):
        build_input_schema(model, metadata)


def test_unknown_category_must_exist_in_fitted_vocabulary(schema_artifacts):
    """Do not introduce Unknown when the reference did not learn it."""
    model, metadata = schema_artifacts
    transformer = model.named_steps["preprocess"].named_steps["column_transformer"]
    columns = next(cols for name, _, cols in transformer.transformers_ if name == "categorical")
    encoder = transformer.named_transformers_["categorical"].named_steps["encoder"]
    index = list(columns).index("workclass")
    values = encoder.categories_[index]
    encoder.categories_[index] = values[values != "Unknown"]
    assert not build_input_schema(model, metadata).feature("workclass").allows_unknown


@pytest.fixture
def valid_payload():
    """Provide all independent fields and a canonical education category."""
    return {
        "age": 42, "education": "Bachelors", "capital-gain": 0,
        "capital-loss": 0, "hours-per-week": 45, "workclass": "Private",
        "marital-status": "Married-civ-spouse", "occupation": "Exec-managerial",
        "relationship": "Husband", "race": "White", "sex": "Male",
        "native-country": "United-States",
    }


def test_valid_input_order_and_immutability(schema, valid_payload):
    """Return a fresh ordered mapping without changing the original payload."""
    original = valid_payload.copy()
    result = validate_input(valid_payload, schema)
    assert tuple(result) == schema.required_features
    assert result == {**original, "education-num": 13}
    assert valid_payload == original
    assert result is not valid_payload
    assert all(type(result[f.name]) is int for f in schema.fields if f.kind == "integer")


@pytest.mark.parametrize("category,number", list(EDUCATION_TO_NUMBER.items()))
def test_education_alternatives(schema, valid_payload, category, number):
    """Derive either education field and accept consistent paired values."""
    valid_payload["education"] = category
    category_result = validate_input(valid_payload, schema)
    valid_payload.pop("education")
    valid_payload["education-num"] = number
    number_result = validate_input(valid_payload, schema)
    valid_payload["education"] = category
    paired_result = validate_input(valid_payload, schema)
    assert category_result == number_result == paired_result
    assert paired_result["education-num"] == number
    assert paired_result["education"] == category


def test_normalize_categories(schema, valid_payload):
    """Normalize case and surrounding whitespace without semantic guessing."""
    original = valid_payload.copy()
    for field in schema.fields:
        if field.kind == "categorical" and field.name in valid_payload:
            valid_payload[field.name] = f"  {valid_payload[field.name].swapcase()}  "
    assert validate_input(valid_payload, schema) == validate_input(original, schema)


@pytest.mark.parametrize("name", ["workclass", "occupation", "native-country"])
def test_explicit_unknown(schema, valid_payload, name):
    """Accept Unknown only when the field's fitted vocabulary supports it."""
    valid_payload[name] = " unknown "
    assert validate_input(valid_payload, schema)[name] == "Unknown"


@pytest.mark.parametrize("name", ["race", "sex", "education", "relationship", "marital-status"])
def test_unsupported_unknown(schema, valid_payload, name):
    """Reject Unknown for fields without an approved missing category."""
    valid_payload[name] = "Unknown"
    with pytest.raises(InputValidationError) as exc:
        validate_input(valid_payload, schema)
    assert "Unknown is not supported" in exc.value.errors[name]


@pytest.mark.parametrize("name", [
    "age", "capital-gain", "capital-loss", "hours-per-week", "workclass",
    "marital-status", "occupation", "relationship", "race", "sex", "native-country",
])
def test_missing_independent_fields(schema, valid_payload, name):
    """Never fabricate any missing independent feature, including zero amounts."""
    del valid_payload[name]
    with pytest.raises(InputValidationError) as exc:
        validate_input(valid_payload, schema)
    assert exc.value.errors[name] == "Required field is missing."


def test_missing_education(schema, valid_payload):
    """Require at least one education representation."""
    del valid_payload["education"]
    with pytest.raises(InputValidationError) as exc:
        validate_input(valid_payload, schema)
    assert "Provide education or education-num" in exc.value.errors["education"]


def test_conflicting_education(schema, valid_payload):
    """Reject two individually valid but inconsistent education values."""
    valid_payload["education-num"] = 14
    with pytest.raises(InputValidationError) as exc:
        validate_input(valid_payload, schema)
    assert "Conflicts" in exc.value.errors["education-num"]


@pytest.mark.parametrize("name", ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"])
def test_numeric_boundaries(schema, valid_payload, name):
    """Accept exact boundaries and reject values just outside the domain."""
    if name == "education-num":
        valid_payload.pop("education")
    field = schema.feature(name)
    for number in (field.minimum, field.maximum):
        valid_payload[name] = number
        assert validate_input(valid_payload, schema)[name] == number
    for number in (field.minimum - 1, field.maximum + 1):
        valid_payload[name] = number
        with pytest.raises(InputValidationError) as exc:
            validate_input(valid_payload, schema)
        assert name in exc.value.errors


@pytest.mark.parametrize("value", [True, False, 42.0, 42.5, "42", None, np.nan, np.inf, -np.inf, [], {}, np.bool_(True)])
def test_invalid_integer_types(schema, valid_payload, value):
    """Reject booleans, floats, numeric strings, nulls and other nonintegers."""
    valid_payload["age"] = value
    with pytest.raises(InputValidationError) as exc:
        validate_input(valid_payload, schema)
    assert exc.value.errors["age"] == "Expected an integer."


def test_numpy_integer_normalized(schema, valid_payload):
    """Convert valid NumPy integer scalars to ordinary Python integers."""
    valid_payload["age"] = np.int64(42)
    assert type(validate_input(valid_payload, schema)["age"]) is int


@pytest.mark.parametrize("value", [None, "", "  ", "?", "unsupported", "self-employed", 42, True, [], {}])
def test_invalid_categories(schema, valid_payload, value):
    """Reject empty, unknown, ambiguous and incorrectly typed categories."""
    valid_payload["workclass"] = value
    with pytest.raises(InputValidationError) as exc:
        validate_input(valid_payload, schema)
    assert "workclass" in exc.value.errors


@pytest.mark.parametrize("name", ["fnlwgt", "income", "extra", "Age"])
def test_unexpected_fields(schema, valid_payload, name):
    """Reject extra fields rather than silently ignoring them."""
    valid_payload[name] = 42
    with pytest.raises(InputValidationError) as exc:
        validate_input(valid_payload, schema)
    assert exc.value.errors[name] == "Unexpected field."


@pytest.mark.parametrize("payload", [None, [], "not a mapping", {1: "value"}])
def test_invalid_payload_structure(schema, payload):
    """Reject nonmapping inputs and mappings with nonstring feature names."""
    with pytest.raises(InputValidationError) as exc:
        validate_input(payload, schema)
    assert "__input__" in exc.value.errors


def test_multiple_errors_and_no_value_echo(schema, valid_payload):
    """Return all field issues without including raw submitted values."""
    del valid_payload["race"]
    valid_payload["age"] = -1
    valid_payload["occupation"] = "invalid-private-value"
    valid_payload["education-num"] = 12
    original = valid_payload.copy()
    with pytest.raises(InputValidationError) as exc:
        validate_input(valid_payload, schema)
    assert set(exc.value.errors) == {"race", "age", "occupation", "education-num"}
    assert "invalid-private-value" not in str(exc.value)
    assert valid_payload == original


def test_supplied_invalid_education_not_ignored(schema, valid_payload):
    """Do not overwrite an explicitly supplied invalid education value."""
    valid_payload["education-num"] = None
    with pytest.raises(InputValidationError) as exc:
        validate_input(valid_payload, schema)
    assert "education-num" in exc.value.errors


def test_normalized_input_preprocessing_parity(schema, schema_artifacts, valid_payload):
    """Match actual fitted preprocessing for canonical and normalized rows."""
    canonical = {**valid_payload, "education-num": 13}
    expected = pd.DataFrame([canonical], columns=schema.required_features)
    valid_payload["workclass"] = " private "
    actual = pd.DataFrame([validate_input(valid_payload, schema)])
    preprocessor = schema_artifacts[0].named_steps["preprocess"]
    np.testing.assert_array_equal(preprocessor.transform(actual), preprocessor.transform(expected))
