"""Test the inference contract against synthetic fitted preprocessing."""

from dataclasses import FrozenInstanceError

import pytest
from sklearn.pipeline import Pipeline

from src.data.preprocess import FEATURE_COLUMNS, PreprocessingConfig, build_preprocessor
from src.inference.schema import (
    EDUCATION_TO_NUMBER,
    NUMBER_TO_EDUCATION,
    build_input_schema,
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
