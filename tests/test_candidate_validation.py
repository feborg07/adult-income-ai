"""Verify frozen-candidate validation and group isolation without real test data."""
from argparse import Namespace
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from scripts import validate_candidates as validation
from src.data.preprocess import PreprocessingConfig, create_cv_folds, load_untransformed_partition, predictor_group_ids
from tests.test_train_cv import cv_workspace, tracked_mlflow


def test_frozen_shortlist():
    """Confirm the selected families, variants and simple grid benchmark."""
    plan = validation.read_mapping(validation.ROOT / 'configs/candidate_validation.yml')
    assert list(plan['candidates']) == ['lightgbm', 'xgboost', 'catboost', 'hgb_optuna', 'hgb_grid']
    assert plan['representation'] == 'no_fnlwgt'
    assert plan['candidates']['hgb_grid']['parameters'] == {'max_iter': 150, 'learning_rate': .1, 'max_leaf_nodes': 15}


def test_unseen_mask_uses_retained_inputs():
    """Exclude repeated predictor combinations even when unused fields differ."""
    train = pd.DataFrame({'age': [20, 30], 'fnlwgt': [1, 2], 'education': ['A', 'B']})
    val = pd.DataFrame({'age': [20, 40], 'fnlwgt': [3, 4], 'education': ['B', 'A']})
    assert validation.unseen_mask(train, val).tolist() == [False, True]


def test_group_calibration_split(cv_workspace):
    """Require disjoint retained groups and complete assignment of training rows."""
    args, _, _ = cv_workspace
    X, y = load_untransformed_partition(args.processed_dir / 'baseline')
    fit, cal = validation.calibration_indices(X, y, .2, 42)
    groups = predictor_group_ids(X.drop(columns=list(validation.GROUPING_EXCLUSIONS)))
    assert not np.intersect1d(groups[fit], groups[cal]).size
    assert np.array_equal(np.sort(np.r_[fit, cal]), np.arange(len(X)))


def synthetic_validation_args(cv_workspace, tmp_path):
    """Prepare a one-candidate plan and validation-only data for integration tests."""
    args, _, _ = cv_workspace
    folder = args.processed_dir / 'baseline'
    X, y = load_untransformed_partition(folder)
    # Include both repeated and previously unseen predictor vectors.
    Xval = X.iloc[:24].copy().reset_index(drop=True)
    Xval.loc[12:, 'age'] += 100
    Xval['fnlwgt'] += 10_000_000
    Xval.assign(income=y.iloc[:24].to_numpy()).to_csv(folder / 'raw_val.csv', index=False)
    folds = create_cv_folds(X, y, grouping_excluded_features=validation.GROUPING_EXCLUSIONS)
    effective = replace(PreprocessingConfig(**json.loads((folder / 'preprocessing_config.json').read_text())), excluded_features=('fnlwgt',))
    plan = {'seed': 42, 'calibration_size': .2, 'representation': 'no_fnlwgt',
            'raw_training_sha256': validation.training.file_sha256(folder / 'raw_train.csv'),
            'fold_sha256': hashlib.sha256(json.dumps([[p.tolist() for p in f] for f in folds]).encode()).hexdigest(),
            'candidates': {'synthetic': {'family': 'logistic_regression', 'variant': 'baseline',
                'source_run_id': 'synthetic-cv', 'cv_f1': .7,
                'effective_preprocessing_sha256': hashlib.sha256(json.dumps(asdict(effective), sort_keys=True).encode()).hexdigest(),
                'parameters': {'C': 1.0, 'max_iter': 5000, 'solver': 'lbfgs'}}}}
    path = tmp_path / 'plan.yml'
    path.write_text(yaml.safe_dump(plan))
    return Namespace(plan=path, config=validation.ROOT / 'configs/config.yml', processed_dir=args.processed_dir, output_dir=args.output_dir), Xval


def test_validation_execution(cv_workspace, tracked_mlflow, tmp_path):
    """Run a real fit, score unseen rows, log artifacts and preserve the reference."""
    args, Xval = synthetic_validation_args(cv_workspace, tmp_path)
    reference = cv_workspace[2]
    before = reference.read_bytes()
    output = validation.run_validation(args)
    summary = json.loads((output / 'candidate_validation_summary.json').read_text())
    assert summary['test_accessed'] is False
    assert summary['validation_rows'] == 24 and summary['unseen_validation_rows'] == 12
    candidate = summary['candidates'][0]
    assert 0 <= candidate['threshold'] <= 1
    assert 0 <= candidate['val_f1'] <= 1 and 0 <= candidate['unseen_val_f1'] <= 1
    assert reference.read_bytes() == before
    pipeline = joblib.load(output / 'synthetic/model_pipeline.joblib')
    assert pipeline.predict_proba(Xval.drop(columns=['fnlwgt'])).shape == (24, 2)
    assert tracked_mlflow[0].log_artifact.call_count == 4


def test_reject_changed_training_partition(cv_workspace, tmp_path):
    """Reject provenance mismatch before creating training or tracking artifacts."""
    args, _ = synthetic_validation_args(cv_workspace, tmp_path)
    path = args.processed_dir / 'baseline/raw_train.csv'
    with path.open('a') as handle:
        handle.write('\n')
    with pytest.raises(ValueError, match='Original training partition changed'):
        validation.run_validation(args)
    assert not args.output_dir.exists()
