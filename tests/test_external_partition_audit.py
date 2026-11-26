"""Tests for label-free reconstruction and cross-partition overlap reporting."""

import pandas as pd
import pytest

from scripts.audit_external_partitions import (
    audit_pair, build_report, read_predictors, recover_test_predictors,
)
from src.data.preprocess import COLUMN_NAMES, FEATURE_COLUMNS, NUMERICAL_FEATURES


def make_row(weight=100, race="White", sex="Male", age=40):
    """Create a complete synthetic Adult predictor vector."""
    row = {column: (1 if column in NUMERICAL_FEATURES else "category")
           for column in FEATURE_COLUMNS}
    row.update(age=age, fnlwgt=weight, race=race, sex=sex)
    return row


def frame(*rows):
    """Create a predictor frame with the expected schema."""
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def test_recover_preserves_duplicate_multiplicity():
    """Reconstruct all held-out duplicates, not merely unique rows."""
    a, b, c = make_row(100), make_row(200), make_row(300)
    test = recover_test_predictors(frame(a, b, c, c), frame(a), frame(b))
    assert len(test) == 2
    assert test["fnlwgt"].tolist() == [300, 300]


def test_reject_unknown_development_rows():
    """Reject saved predictors absent from the source dataset."""
    with pytest.raises(ValueError, match="not a subset"):
        recover_test_predictors(frame(make_row(100), make_row(200)),
                                frame(make_row(300)), frame(make_row(200)))


def test_reject_partially_assigned_group():
    """Reject a full-vector group split between development and test."""
    a, b = make_row(100), make_row(200)
    with pytest.raises(ValueError, match="partially allocated"):
        recover_test_predictors(frame(a, a, b), frame(a), frame(b))


def test_reject_original_train_validation_overlap():
    """Reject identical original predictor groups across train and validation."""
    a, b = make_row(100), make_row(200)
    with pytest.raises(ValueError, match="train/validation"):
        recover_test_predictors(frame(a, a, b), frame(a), frame(a))


def test_overlap_is_counted_on_both_sides():
    """Measure asymmetric row overlap using the same group ID namespace."""
    a, b, c = make_row(100), make_row(200), make_row(300, age=50)
    features = [column for column in FEATURE_COLUMNS if column != "fnlwgt"]
    result = audit_pair(frame(a), frame(b, b, c), features)
    assert (result["shared_groups"], result["left_rows"], result["right_rows"]) == (1, 1, 2)
    assert result["right_fraction"] == pytest.approx(2 / 3)


def test_report_distinguishes_representations():
    """Verify original isolation and reduced-representation overlap separately."""
    a, b, c = make_row(100), make_row(200, age=50), make_row(300, race="Black")
    report = build_report(frame(a, b, c), frame(a), frame(b))
    assert report["target_columns_loaded"] is False
    assert len(report["overlap"]) == 36
    rows = {(r["representation"], r["pair"]): r for r in report["overlap"]}
    assert rows["original", "train/test"]["shared_groups"] == 0
    assert rows["no_fnlwgt_race", "train/test"]["shared_groups"] == 1
    assert rows["no_fnlwgt", "train/test"]["shared_groups"] == 0
    assert rows["no_fnlwgt_effective_raw", "train/test"]["shared_groups"] == 0
    assert rows["no_fnlwgt_race_effective_raw", "train/test"]["shared_groups"] == 1


def test_missing_predictors_match_across_formats():
    """Treat equivalent missing categorical cells as one predictor value."""
    a, b, c = make_row(100), make_row(200), make_row(300, age=50)
    a["occupation"] = "?"
    b["occupation"] = None
    report = build_report(frame(a, b, c), frame(a), frame(c))
    row = next(r for r in report["overlap"] if r["representation"] == "no_fnlwgt_race"
               and r["pair"] == "train/test")
    assert row["shared_groups"] == 1


def test_read_raw_and_saved_without_target(tmp_path):
    """Ignore arbitrary target values and the UCI test metadata line."""
    row = make_row()
    raw = tmp_path / "adult.test"
    raw.write_text("| metadata\n" + ", ".join(str(row.get(c, "not-a-label"))
                   for c in COLUMN_NAMES) + "\n", encoding="utf-8")
    saved = tmp_path / "raw_train.csv"
    frame(row).assign(income="not-a-label").to_csv(saved, index=False)
    assert read_predictors(raw, raw=True).equals(read_predictors(saved))
    assert "income" not in read_predictors(saved).columns


def test_main_writes_aggregate_report(tmp_path, monkeypatch):
    """Exercise the CLI with all three variant partitions and no usable labels."""
    import json
    from scripts import audit_external_partitions as audit

    a, b, c = make_row(100), make_row(200), make_row(300)
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    for filename, rows in (("adult.data", [a, b]), ("adult.test", [c])):
        (raw / filename).write_text("\n".join(
            ", ".join(str(row.get(column, "ignored")) for column in COLUMN_NAMES)
            for row in rows) + "\n", encoding="utf-8")
    for variant in audit.VARIANTS:
        processed = tmp_path / "data/processed" / variant
        processed.mkdir(parents=True)
        for name, row in (("train", a), ("val", b)):
            frame(row).assign(income="ignored").to_csv(processed / f"raw_{name}.csv", index=False)
    monkeypatch.setattr(audit, "EXPECTED_ROWS", 3)
    audit.main(["--root", str(tmp_path)])
    report = json.loads((tmp_path / "reports/external_partition_overlap.json").read_text())
    assert report["partition_rows"] == {"train": 1, "validation": 1, "test": 1}
    assert report["target_columns_loaded"] is False


def test_representation_a_overlap_with_weight_only_difference():
    """A matches across census weights but retains racial and sex distinctions."""
    a = make_row(100)
    b = make_row(200, age=50)
    c = make_row(300)
    d = make_row(400, race="Black")
    e = make_row(500, sex="Female")
    report = build_report(frame(a, b, c, d, e), frame(a), frame(b))
    rows = {(r["representation"], r["pair"]): r for r in report["overlap"]}
    assert rows["no_fnlwgt", "train/test"]["right_rows"] == 1
    assert rows["no_fnlwgt_race", "train/test"]["right_rows"] == 2
    assert rows["common_cv_grouping", "train/test"]["right_rows"] == 3
    assert rows["no_fnlwgt_effective_raw", "train/test"]["right_rows"] == 1
    assert rows["no_fnlwgt", "train/test"]["right_fraction"] == pytest.approx(1 / 3)
