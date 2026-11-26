"""Audit predictor overlap across the original saved data partitions, without labels."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocess import (
    CATEGORICAL_FEATURES, COLUMN_NAMES, FEATURE_COLUMNS, NUMERICAL_FEATURES,
    predictor_group_ids,
)

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("baseline", "robust", "interactions")
EXPECTED_ROWS = 48842
REPRESENTATIONS = {
    "original": (),
    "no_fnlwgt": ("fnlwgt",),
    "no_fnlwgt_race": ("fnlwgt", "race"),
    "no_fnlwgt_sex": ("fnlwgt", "sex"),
    "common_cv_grouping": ("fnlwgt", "race", "sex"),
    # Every current variant drops the redundant education string.
    "no_fnlwgt_effective_raw": ("fnlwgt", "education"),
    "no_fnlwgt_race_effective_raw": ("fnlwgt", "race", "education"),
    "no_fnlwgt_sex_effective_raw": ("fnlwgt", "sex", "education"),
    "no_fnlwgt_demographics_effective_raw": ("fnlwgt", "race", "sex", "education"),
}


def normalize_predictors(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the existing deterministic predictor cleaning, without target access."""
    if set(frame.columns) != set(FEATURE_COLUMNS) or frame.empty:
        raise ValueError("Expected a nonempty frame of all original predictors.")
    cleaned = frame.loc[:, FEATURE_COLUMNS].copy()
    for column in CATEGORICAL_FEATURES:
        cleaned[column] = cleaned[column].astype("string").str.strip()
    cleaned = cleaned.replace(["?", "", "nan"], np.nan)
    for column in NUMERICAL_FEATURES:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    if cleaned.isna().all(axis=1).any():
        raise ValueError("Completely empty predictor row; cannot establish partition identity.")
    return cleaned.reset_index(drop=True)


def read_predictors(path: Path, *, raw: bool = False) -> pd.DataFrame:
    """Read only predictor columns from a raw UCI or saved partition CSV."""
    options = dict(usecols=FEATURE_COLUMNS, dtype={c: str for c in CATEGORICAL_FEATURES},
                   keep_default_na=False, na_values=[""], float_precision="round_trip")
    if raw:
        options.update(header=None, names=COLUMN_NAMES, skipinitialspace=True, comment="|")
    return normalize_predictors(pd.read_csv(path, **options))


def recover_test_predictors(full: pd.DataFrame, train: pd.DataFrame,
                            val: pd.DataFrame) -> pd.DataFrame:
    """Recover held-out predictor rows by exact full-vector multiset subtraction."""
    frames = [normalize_predictors(frame) for frame in (full, train, val)]
    # Factorize jointly: independently assigned group IDs are not comparable.
    codes = predictor_group_ids(pd.concat(frames, ignore_index=True))
    full_ids, train_ids, val_ids = np.split(codes, [len(full), len(full) + len(train)])
    n_groups = int(codes.max()) + 1
    counts = [np.bincount(ids, minlength=n_groups) for ids in (full_ids, train_ids, val_ids)]
    total, train_count, val_count = counts
    dev_count = train_count + val_count
    if np.any(dev_count > total):
        raise ValueError("Saved development predictors are not a subset of the original dataset.")
    if np.any((train_count > 0) & (val_count > 0)):
        raise ValueError("Original full-vector train/validation group isolation is violated.")
    # Under the original splitter every exact group belongs to one partition.
    if np.any((dev_count > 0) & (dev_count != total)):
        raise ValueError("A full-vector group is partially allocated to development data.")
    test = frames[0].iloc[np.flatnonzero(dev_count[full_ids] == 0)].reset_index(drop=True)
    if test.empty or len(test) + len(train) + len(val) != len(full):
        raise ValueError("Reconstructed partition counts do not reconcile.")
    return test


def audit_pair(left: pd.DataFrame, right: pd.DataFrame, features: list[str]) -> dict:
    """Count common groups and affected rows on both sides of a partition pair."""
    codes = predictor_group_ids(pd.concat([left[features], right[features]], ignore_index=True))
    left_ids, right_ids = codes[:len(left)], codes[len(left):]
    shared = np.intersect1d(left_ids, right_ids)
    left_rows = int(np.isin(left_ids, shared).sum())
    right_rows = int(np.isin(right_ids, shared).sum())
    return {"shared_groups": len(shared), "left_rows": left_rows, "right_rows": right_rows,
            "left_total": len(left), "right_total": len(right),
            "left_fraction": left_rows / len(left), "right_fraction": right_rows / len(right)}


def build_report(full: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame) -> dict:
    """Build an aggregate-only overlap report under all relevant grouping bases."""
    full, train, val = (normalize_predictors(frame) for frame in (full, train, val))
    test = recover_test_predictors(full, train, val)
    partitions = {"train": train, "validation": val, "test": test}
    pairs = [(a, b, partitions[a], partitions[b]) for a, b in combinations(partitions, 2)]
    pairs.append(("development", "test", pd.concat([train, val], ignore_index=True), test))
    results = []
    for name, excluded in REPRESENTATIONS.items():
        features = [column for column in FEATURE_COLUMNS if column not in excluded]
        for left, right, X_left, X_right in pairs:
            results.append({"representation": name, "excluded_features": list(excluded),
                            "grouping_features": features, "pair": f"{left}/{right}",
                            **audit_pair(X_left, X_right, features)})
    return {"audit_type": "predictor_overlap_only", "target_columns_loaded": False,
            "test_reconstruction": "exact_full_predictor_group_complement",
            "partition_rows": {name: len(frame) for name, frame in partitions.items()},
            "full_rows": len(full), "overlap": results}


def main(argv=None) -> None:
    """Verify the saved variant partitions and write a JSON-only audit report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=Path("reports/external_partition_overlap.json"))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    full = pd.concat([read_predictors(root / "data/raw" / filename, raw=True)
                      for filename in ("adult.data", "adult.test")], ignore_index=True)
    if len(full) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} source rows, got {len(full)}.")
    processed = root / "data/processed"
    train = read_predictors(processed / VARIANTS[0] / "raw_train.csv")
    val = read_predictors(processed / VARIANTS[0] / "raw_val.csv")
    # Confirm all three variants use the same original predictor partitions.
    for variant in VARIANTS[1:]:
        for name, expected in (("train", train), ("val", val)):
            actual = read_predictors(processed / variant / f"raw_{name}.csv")
            if not actual.equals(expected):
                raise ValueError(f"Saved {name} predictors differ for {variant}.")
    report = build_report(full, train, val)
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Partition rows: {report['partition_rows']}")
    for row in report["overlap"]:
        print(f"{row['representation']} | {row['pair']}: {row['shared_groups']} groups, "
              f"right rows {row['right_rows']}/{row['right_total']} ({row['right_fraction']:.2%})")
    print(f"Audit saved: {output}")


if __name__ == "__main__":
    main()
