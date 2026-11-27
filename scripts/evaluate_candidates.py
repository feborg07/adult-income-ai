"""Evaluate the Phase 3 frozen shortlist; default mode never reads test labels.

Run from the repository root with python -m scripts.evaluate_candidates.
Only --evaluate authorizes historical held-out scoring. No fitting or promotion.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import uuid

import joblib
import numpy as np
import pandas as pd

from scripts.validate_candidates import (
    GROUPING_EXCLUSIONS, OVERLAP_EXCLUSIONS, calibration_indices, unseen_mask,
)
from src.data.preprocess import (
    FEATURE_COLUMNS, TARGET_COLUMN, PreprocessingConfig, clean_data,
    create_cv_folds, create_train_val_test_split, encode_target,
    load_and_concatenate_data, load_untransformed_partition, split_features_target,
    validate_data,
)
from src.models import train as training
from src.models.configuration import read_mapping
from src.models.tasks import binary_classification as binary

ROOT = Path(__file__).resolve().parents[1]
EXECUTION = 'b60d1c21c4784ddfac1214a9da73aac9'
SELECTED = 'xgboost'
REFERENCE = 'reference'
VARIANTS = ('baseline', 'robust', 'interactions')
EXPECTED_ROWS = (31263, 7815, 9764)
EXPECTED_UNSEEN = 7737
MARGIN = .01
# Anchor the reviewed metadata independently of the generated model directory.
# Canonical JSON ignores formatting and line-ending differences, not changed values.
METADATA_SHA = {
    'candidate_validation_summary.json': '5aa4b223dff91fa7dc8fe6e3253e9ac44ca27e55756d1fbc824a2bf58b091531',
    'frozen_plan.json': '86245b3a46d3ceacf47e7cb06945acd2584833079fc8529aac3cffe85541db30',
    'partition_identity.json': '06b109de5f75c4dbdb22509f2554fbf684cf91033f5e2eaa523cc37d851436f5',
    'best_model_metadata.json': '208b14e022f98518a1c804ac68c4948dba48cffe624deb58b8d86ea09a1973ea',
}


def require(condition, message):
    """Fail closed when an integrity or evaluation invariant is violated."""
    if not condition:
        raise ValueError(message)


def canonical_sha(value):
    """Hash JSON values with deterministic key order and whitespace."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_json(path):
    """Read a JSON artifact without altering it."""
    return json.loads(path.read_text(encoding='utf-8'))


def anchored_json(path):
    """Verify a reviewed metadata artifact against its independent content hash."""
    data = read_json(path)
    require(canonical_sha(data) == METADATA_SHA[path.name], f'Frozen metadata changed: {path}')
    return data


def verify_file(path, expected, *, allow_newlines=False):
    """Verify exact bytes, optionally accepting only CRLF/LF conversion for CSVs."""
    actual = training.file_sha256(path)
    if actual == expected:
        return actual
    if allow_newlines:
        lf = path.read_bytes().replace(b'\r\n', b'\n')
        alternatives = (lf, lf.replace(b'\n', b'\r\n'))
        if any(hashlib.sha256(data).hexdigest() == expected for data in alternatives):
            return actual
    raise ValueError(f'File checksum mismatch: {path} (actual {actual}, expected {expected})')


def load_frozen_metadata(root):
    """Check the approved shortlist, per-model records and model bytes before loading."""
    folder = root / 'models/runs' / EXECUTION
    reference = root / 'models/reference'
    summary = anchored_json(folder / 'candidate_validation_summary.json')
    plan = anchored_json(folder / 'frozen_plan.json')
    identity = anchored_json(folder / 'partition_identity.json')
    ref = anchored_json(reference / 'best_model_metadata.json')
    require(plan == read_mapping(root / 'configs/candidate_validation.yml'), 'Tracked frozen plan changed.')
    require(all(summary.get(k) == v for k, v in identity.items()), 'Partition metadata disagrees.')
    require(identity['execution_id'] == EXECUTION and identity['test_accessed'] is False,
            'Unexpected validation execution or test exposure.')
    require(identity['representation'] == plan['representation'] == 'no_fnlwgt', 'Representation mismatch.')
    require(identity['raw_training_sha256'] == plan['raw_training_sha256'] and
            identity['fold_sha256'] == plan['fold_sha256'], 'Frozen training/fold identity mismatch.')
    require(identity['overlap_excluded_features'] == list(OVERLAP_EXCLUSIONS) and
            identity['grouping_excluded_features'] == list(GROUPING_EXCLUSIONS), 'Grouping definition changed.')
    rows = summary['candidates']
    require(len(rows) == 5 and {r['candidate'] for r in rows} == set(plan['candidates']), 'Shortlist mismatch.')
    models = {}
    for row in rows:
        name = row['candidate']
        spec = plan['candidates'][name]
        require(row == read_json(folder / name / 'validation_result.json'), f'Candidate metadata mismatch: {name}')
        require(all(row.get(k) == v for k, v in spec.items()), f'Candidate plan mismatch: {name}')
        path = folder / name / 'model_pipeline.joblib'
        verify_file(path, row['model_sha256'])
        models[name] = dict(row, path=path, representation='no_fnlwgt')
    require(ref['positive_class'] == 1, 'Reference positive class mismatch.')
    path = reference / ref['model_filename']
    verify_file(path, ref['model_sha256'])
    models[REFERENCE] = dict(candidate=REFERENCE, family=ref['model_family'],
        variant=ref['preprocessing_variant'], threshold=ref['decision_threshold'],
        model_sha256=ref['model_sha256'], val_f1=ref['validation_f1'],
        val_pr_auc=ref['validation_pr_auc'], val_roc_auc=ref['validation_roc_auc'],
        run_id=ref['mlflow_run_id'], representation='control', path=path)
    return models, plan, identity, ref


def verify_development(root, plan, identity):
    """Recompute development fingerprints, fold IDs and calibration membership only."""
    processed = root / 'data/processed'
    for variant in VARIANTS:
        for split, key in (('train', 'raw_training_sha256'), ('val', 'raw_validation_sha256')):
            verify_file(processed / variant / f'raw_{split}.csv', identity[key], allow_newlines=True)
    X, y = load_untransformed_partition(processed / 'baseline', 'train')
    Xval, yval = load_untransformed_partition(processed / 'baseline', 'val')
    require((len(X), len(Xval)) == EXPECTED_ROWS[:2], 'Development row counts changed.')
    folds = create_cv_folds(X, y, n_splits=3, calibration_size=plan['calibration_size'],
                            random_state=plan['seed'], grouping_excluded_features=GROUPING_EXCLUSIONS)
    fold_sha = hashlib.sha256(json.dumps([[p.tolist() for p in fold] for fold in folds]).encode()).hexdigest()
    require(fold_sha == identity['fold_sha256'], 'Historical fold fingerprint changed.')
    fit, cal = calibration_indices(X, y, plan['calibration_size'], plan['seed'])
    cal_sha = hashlib.sha256(json.dumps([fit.tolist(), cal.tolist()]).encode()).hexdigest()
    require(cal_sha == identity['calibration_split_sha256'], 'Calibration membership changed.')
    require((len(fit), len(cal)) == (identity['fit_rows'], identity['calibration_rows']), 'Calibration counts changed.')
    configs = {v: PreprocessingConfig(**read_json(processed / v / 'preprocessing_config.json')) for v in VARIANTS}
    for config in configs.values():
        require(not config.excluded_features and config.duplicate_aware_split and
                config.random_state == plan['seed'] and config.test_size == .2 and
                config.validation_size == .2, 'Original partition configuration changed.')
    for name, spec in plan['candidates'].items():
        effective = replace(configs[spec['variant']], excluded_features=('fnlwgt',))
        digest = hashlib.sha256(json.dumps(asdict(effective), sort_keys=True).encode()).hexdigest()
        require(digest == spec['effective_preprocessing_sha256'], f'Preprocessing identity changed: {name}')
    unseen = unseen_mask(X, Xval)
    require(int(unseen.sum()) == identity['unseen_validation_rows'], 'Unseen validation count changed.')
    return X, y, Xval, yval, configs['baseline'], unseen


def verify_pipeline(model, spec, Xval, yval, unseen, *, historical_prob=None):
    """Verify frozen schema, class ordering, threshold and validation probability behavior."""
    threshold = spec['threshold']
    require(isinstance(threshold, (int, float)) and np.isfinite(threshold) and 0 <= threshold <= 1,
            'Invalid frozen threshold.')
    engineer = model.named_steps['preprocess'].named_steps['feature_engineering']
    required = list(getattr(engineer, 'required_features_', engineer.feature_names_in_))
    expected = [c for c in FEATURE_COLUMNS if spec['representation'] == 'control' or c != 'fnlwgt']
    require(required == expected, f'Unexpected fitted input schema: {spec["candidate"]}')
    prob = binary.positive_probabilities(model, Xval)
    if spec['representation'] == 'no_fnlwgt':
        # The application must work without the census weight, not merely ignore its value.
        reduced = binary.positive_probabilities(model, Xval.drop(columns=['fnlwgt']))
        require(np.allclose(prob, reduced, rtol=0, atol=1e-10), 'Reduced schema prediction mismatch.')
    if historical_prob is not None:
        require(spec['candidate'] == REFERENCE, 'Historical matrix is reference-only.')
        require(historical_prob.shape == prob.shape, 'Reference validation probability alignment mismatch.')
        # CSV parsing can change tree routing at numerical split boundaries.
        require(np.array_equal(prob >= threshold, historical_prob >= threshold),
                'Reference validation class predictions differ from historical matrix.')
    for prefix, mask in [('val', np.ones(len(Xval), dtype=bool)), ('unseen_val', unseen)]:
        # Reproduce the legacy reference metrics using its original scoring input.
        metric_prob = historical_prob if prefix == 'val' and historical_prob is not None else prob
        metrics = binary.evaluate(yval.iloc[np.flatnonzero(mask)], metric_prob[mask], threshold)
        for key, actual in metrics.items():
            field = f'{prefix}_{key}'
            if field not in spec:
                continue
            expected_metric = spec[field]
            matches = actual == expected_metric if isinstance(actual, list) else np.isclose(actual, expected_metric, rtol=0, atol=1e-7)
            require(matches, f'Frozen validation {field} mismatch: {spec["candidate"]}')
    return required


def reference_validation_probabilities(root, model, Xval, yval, ref):
    """Reproduce legacy validation scores from checksum-verified saved features."""
    folder = root / 'data/processed' / ref['preprocessing_variant']
    feature_path, target_path = folder / 'X_val.csv', folder / 'y_val.csv'
    for path in (feature_path, target_path):
        verify_file(path, ref['dataset_sha256'][path.name], allow_newlines=True)
    pd.testing.assert_series_equal(yval.reset_index(drop=True), read_saved_target(target_path),
                                   check_dtype=False, check_names=False)
    # Match the original training loader's default CSV parsing behavior.
    saved = pd.read_csv(feature_path)
    transformed = model.named_steps['preprocess'].transform(Xval)
    named = model.named_steps['feature_names'].transform(transformed)
    require(list(named.columns) == list(saved.columns) and named.shape == saved.shape and
            np.allclose(named.to_numpy(), saved.to_numpy(), rtol=1e-9, atol=1e-10),
            'Reference validation matrix differs from the reconstructed raw transform.')
    return binary.positive_probabilities(model.named_steps['classifier'], saved)


def preflight(root):
    """Load trusted frozen models and verify development provenance without test access."""
    specs, plan, identity, ref = load_frozen_metadata(root)
    X, y, Xval, yval, config, unseen = verify_development(root, plan, identity)
    models = {}
    for name, spec in specs.items():
        model = joblib.load(spec['path'])  # Only checksum-verified local project artifacts.
        historical_prob = (reference_validation_probabilities(root, model, Xval, yval, ref)
                           if name == REFERENCE else None)
        spec['required_features'] = verify_pipeline(model, spec, Xval, yval, unseen,
                                                    historical_prob=historical_prob)
        models[name] = model
    return specs, models, identity, ref, (X, y, Xval, yval, config)


def assert_partition_equal(X, y, expected_X, expected_y, label):
    """Check exact ordered predictors and labels, allowing only dtype differences."""
    pd.testing.assert_frame_equal(X.reset_index(drop=True), expected_X.reset_index(drop=True),
                                  check_dtype=False, check_exact=True, obj=f'{label} predictors')
    pd.testing.assert_series_equal(y.reset_index(drop=True), expected_y.reset_index(drop=True),
                                   check_dtype=False, check_names=False, check_exact=True, obj=f'{label} labels')


def reconstruct_partitions(raw_paths, config, development):
    """Recover the original test rows in memory using the original deterministic split."""
    X, y, Xval, yval = development
    cleaned = clean_data(load_and_concatenate_data(raw_paths), remove_duplicates=False)
    validate_data(cleaned)
    all_X, all_y = split_features_target(encode_target(cleaned))
    train, val, test, yt, yv, ytest = create_train_val_test_split(
        all_X, all_y, test_size=config.test_size, validation_size=config.validation_size,
        random_state=config.random_state, duplicate_aware=config.duplicate_aware_split)
    assert_partition_equal(train, yt, X, y, 'train')
    assert_partition_equal(val, yv, Xval, yval, 'validation')
    return test.reset_index(drop=True), ytest.reset_index(drop=True)


def read_saved_target(path):
    """Load the historical single-column binary target with strict schema checks."""
    frame = pd.read_csv(path)
    require(list(frame.columns) == [TARGET_COLUMN], f'Unexpected target schema: {path}')
    return binary.validate_target(frame[TARGET_COLUMN], str(path))


def verify_test_alignment(root, Xtest, ytest, reference, ref_metadata):
    """Verify historical test matrices/labels and exact ordered inference alignment."""
    require(len(Xtest) == len(ytest) == EXPECTED_ROWS[2], 'Historical test row count changed.')
    processed = root / 'data/processed'
    hashes = {}
    for variant in VARIANTS:
        folder = processed / variant
        path = folder / 'y_test.csv'
        hashes[variant] = {'y_test.csv': verify_file(path, ref_metadata['dataset_sha256']['y_test.csv'], allow_newlines=True)}
        pd.testing.assert_series_equal(ytest, read_saved_target(path), check_dtype=False, check_names=False)
        hashes[variant]['X_test.csv'] = training.file_sha256(folder / 'X_test.csv')
    # Compare the old fitted reference transform with the original processed matrix.
    path = processed / 'baseline/X_test.csv'
    verify_file(path, ref_metadata['dataset_sha256']['X_test.csv'], allow_newlines=True)
    saved = pd.read_csv(path, float_precision='round_trip')
    transformed = reference.named_steps['preprocess'].transform(Xtest)
    named = reference.named_steps['feature_names'].transform(transformed)
    require(list(named.columns) == list(saved.columns) and named.shape == saved.shape and
            np.allclose(named.to_numpy(), saved.to_numpy(), rtol=1e-9, atol=1e-10),
            'Reconstructed test predictor order differs from historical reference matrix.')
    return hashes


def score_model(model, spec, X, y, unseen):
    """Score the identical full and shared unseen rows with the frozen threshold."""
    require(len(X) == len(y) == len(unseen), 'Evaluation alignment mismatch.')
    probabilities = binary.positive_probabilities(model, X)
    row = {k: spec[k] for k in ('candidate', 'family', 'variant', 'representation', 'threshold', 'model_sha256', 'run_id')}
    row.update({key: spec[key] for key in ('cv_f1', 'val_f1', 'unseen_val_f1', 'required_features') if key in spec})
    for prefix, mask in [('test', np.ones(len(X), dtype=bool)), ('unseen_test', unseen)]:
        target = y.iloc[np.flatnonzero(mask)]
        binary.require_both_classes(target, prefix)
        metrics = binary.evaluate(target, probabilities[mask], spec['threshold'])
        row[f'{prefix}_rows'] = len(target)
        row.update({f'{prefix}_{key}': value for key, value in metrics.items()})
    return row


def performance_gate(rows):
    """Apply only the predeclared performance checks, never rank test candidates."""
    by_name = {row['candidate']: row for row in rows}
    selected, reference = by_name[SELECTED], by_name[REFERENCE]
    checks = {}
    for metric in ('test_f1', 'unseen_test_f1'):
        minimum = reference[metric] - MARGIN
        checks[metric] = {'candidate': selected[metric], 'reference': reference[metric],
                          'minimum': minimum, 'delta': selected[metric] - reference[metric],
                          'passed': bool(selected[metric] >= minimum - 1e-12)}
    return {'selected_candidate': SELECTED, 'margin': MARGIN, 'checks': checks,
            'passed': all(c['passed'] for c in checks.values()), 'promotion_authorized': False}


def run_evaluation(root, verified):
    """Score once per explicit invocation, write aggregate reports and log to MLflow."""
    specs, models, identity, ref, development = verified
    X, y, Xval, yval, config = development
    raw_paths = [root / 'data/raw' / name for name in ('adult.data', 'adult.test')]
    raw_hashes = {path.name: training.file_sha256(path) for path in raw_paths}
    Xtest, ytest = reconstruct_partitions(raw_paths, config, (X, y, Xval, yval))
    hashes = verify_test_alignment(root, Xtest, ytest, models[REFERENCE], ref)
    unseen = unseen_mask(pd.concat([X, Xval], ignore_index=True), Xtest)
    require(int(unseen.sum()) == EXPECTED_UNSEEN, 'Shared unseen test count changed.')
    rows = [score_model(models[name], spec, Xtest, ytest, unseen) for name, spec in specs.items()]
    # Check frozen binaries again before publishing; no model artifacts are written.
    for spec in specs.values():
        verify_file(spec['path'], spec['model_sha256'])
    execution = uuid.uuid4().hex
    output = root / 'reports/heldout' / execution
    output.mkdir(parents=True, exist_ok=False)
    report = {'execution_id': execution, 'source_execution_id': EXECUTION,
              'git_commit': training.git_revision(), 'evaluation_protocol': 'frozen_heldout',
              'historical_test_previously_used': True, 'test_accessed': True,
              'selected_candidate': SELECTED, 'partition_identity': identity,
              'raw_source_sha256': raw_hashes, 'saved_test_sha256': hashes,
              'frozen_metadata_content_sha256': METADATA_SHA,
              'evaluator_sha256': training.file_sha256(Path(__file__)),
              'test_rows': len(Xtest), 'unseen_test_rows': int(unseen.sum()),
              'unseen_definition': 'Absent from original train plus validation on retained raw predictors',
              'overlap_excluded_features': list(OVERLAP_EXCLUSIONS),
              'unseen_mask_sha256': hashlib.sha256(unseen.astype('uint8').tobytes()).hexdigest(),
              'performance_gate': performance_gate(rows), 'candidates': rows,
              'promotion_authorized': False, 'fairness_and_robustness_pending': True}
    pd.DataFrame(rows).to_csv(output / 'heldout_comparison.csv', index=False)
    training.atomic_json_write(output / 'heldout_summary.json', report)
    config_map = training.load_training_config(root / 'configs/config.yml')
    paths = config_map['paths']
    training.configure_mlflow(config_map['mlflow']['experiment'] + '-heldout-evaluation',
                              root / paths['mlflow_db'], root / paths['mlflow_artifacts'])
    with training.mlflow.start_run(run_name=f'heldout-{execution}'):
        training.mlflow.set_tags({'evaluation_protocol': 'frozen_heldout', 'source_execution_id': EXECUTION,
                                  'selected_candidate': SELECTED, 'promotion_authorized': 'false'})
        training.mlflow.log_params({'test_rows': len(Xtest), 'unseen_test_rows': int(unseen.sum()), 'f1_margin': MARGIN})
        training.mlflow.log_metrics({f'{row["candidate"]}.{key}': value for row in rows for key, value in row.items()
                                     if key.startswith(('test_', 'unseen_test_')) and isinstance(value, (float, int))})
        for path in (output / 'heldout_comparison.csv', output / 'heldout_summary.json'):
            training.mlflow.log_artifact(str(path), artifact_path='heldout')
    print(f'Held-out reports: {output}')
    print(f'Performance gate: {report["performance_gate"]["passed"]}; promotion not authorized.')
    return output


def main(argv=None):
    """Run safe preflight by default; require explicit authorization to open test labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate', action='store_true', help='Explicitly score the frozen historical test partition.')
    args = parser.parse_args(argv)
    verified = preflight(ROOT)
    print('Preflight passed: six frozen pipelines, development partitions and validation parity verified.')
    if args.evaluate:
        run_evaluation(ROOT, verified)
    else:
        print('No test labels accessed. Use --evaluate only after approval.')


if __name__ == '__main__':
    main()
