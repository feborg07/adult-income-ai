"""Test the frozen held-out evaluator using synthetic data only."""
from dataclasses import replace
import hashlib
import json
from unittest.mock import MagicMock, Mock

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from scripts import evaluate_candidates as evaluation
from src.data.preprocess import (
    COLUMN_NAMES, PreprocessingConfig, build_preprocessor, clean_data,
    create_train_val_test_split, encode_target, split_features_target,
)
from src.models.tasks import binary_classification as binary


@pytest.fixture
def synthetic_partitions(tmp_path, sample_adult_df):
    """Create a deterministic synthetic raw source and its original split."""
    records = []
    for index in range(20):
        batch = sample_adult_df.copy()
        batch['age'] += index
        batch['fnlwgt'] += 1000 * index
        records.append(batch)
    frame = pd.concat(records, ignore_index=True)
    frame.loc[0, 'occupation'] = '?'
    frame = pd.concat([frame, frame.iloc[[1]]], ignore_index=True)
    raw = tmp_path / 'adult.data'
    frame[COLUMN_NAMES].to_csv(raw, header=False, index=False)
    X, y = split_features_target(encode_target(clean_data(frame)))
    config = PreprocessingConfig()
    Xt, Xv, Xs, yt, yv, ys = create_train_val_test_split(X, y)
    return raw, config, (Xt, yt, Xv, yv), (Xs.reset_index(drop=True), ys.reset_index(drop=True))


@pytest.fixture
def fitted_models(synthetic_partitions):
    """Fit two synthetic pipelines; production evaluator never fits any estimator."""
    _, config, (Xt, yt, Xv, yv), _ = synthetic_partitions
    models, specs = {}, {}
    unseen = evaluation.unseen_mask(Xt, Xv)
    for name, representation in [('xgboost', 'no_fnlwgt'), ('reference', 'control')]:
        cfg = replace(config, excluded_features=('fnlwgt',)) if representation == 'no_fnlwgt' else config
        prep = build_preprocessor(cfg)
        transformed = pd.DataFrame(prep.fit_transform(Xt), columns=prep.get_feature_names_out())
        estimator = LogisticRegression(max_iter=1000).fit(transformed, yt)
        model = evaluation.training.build_deployable_pipeline(prep, estimator, transformed.columns, transformed)
        threshold = .35
        spec = {'candidate': name, 'family': 'logistic_regression', 'variant': 'baseline',
                'representation': representation, 'threshold': threshold, 'model_sha256': 'synthetic',
                'run_id': name}
        probabilities = binary.positive_probabilities(model, Xv)
        for prefix, mask in [('val', np.ones(len(Xv), dtype=bool)), ('unseen_val', unseen)]:
            metrics = binary.evaluate(yv.iloc[np.flatnonzero(mask)], probabilities[mask], threshold)
            spec.update({f'{prefix}_{k}': v for k, v in metrics.items()})
        models[name], specs[name] = model, spec
    return models, specs, unseen


def test_canonical_metadata_hash(tmp_path, monkeypatch):
    """Accept equivalent JSON formatting but reject changed frozen content."""
    path = tmp_path / 'frozen_plan.json'
    obj = {'seed': 42, 'threshold': .35}
    anchors = {path.name: evaluation.canonical_sha(obj)}
    path.write_text(json.dumps(obj, indent=4))
    assert evaluation.anchored_json(path, anchors) == obj
    path.write_text(json.dumps(dict(obj, threshold=.5)))
    with pytest.raises(ValueError, match='Frozen metadata changed'):
        evaluation.anchored_json(path, anchors)


def test_file_checksum(tmp_path):
    """Reject changed model bytes and optionally tolerate CSV line endings only."""
    path = tmp_path / 'artifact.csv'
    original = b'a,b\r\n1,2\r\n'
    expected = hashlib.sha256(original).hexdigest()
    path.write_bytes(original.replace(b'\r\n', b'\n'))
    with pytest.raises(ValueError, match='checksum mismatch'):
        evaluation.verify_file(path, expected)
    assert evaluation.verify_file(path, expected, allow_newlines=True)
    path.write_bytes(b'a,b\n1,3\n')
    with pytest.raises(ValueError, match='checksum mismatch'):
        evaluation.verify_file(path, expected, allow_newlines=True)


def test_reconstruct_original_split(synthetic_partitions):
    """Recover exact historical row order, target alignment and duplicate preservation."""
    raw, config, dev, expected = synthetic_partitions
    actual = evaluation.reconstruct_partitions([raw], config, dev)
    evaluation.assert_partition_equal(*actual, *expected, 'test')
    assert len(dev[0]) + len(dev[2]) + len(actual[0]) == 241


def test_reject_development_order_change(synthetic_partitions):
    """Refuse a reconstructed split that disagrees with frozen development rows."""
    raw, config, (X, y, Xv, yv), _ = synthetic_partitions
    with pytest.raises(AssertionError, match='train predictors'):
        evaluation.reconstruct_partitions([raw], config, (X.iloc[::-1], y.iloc[::-1], Xv, yv))


def test_unseen_uses_entire_development():
    """Exclude matches from either training or validation despite unused-feature differences."""
    train = pd.DataFrame({'age': [20], 'fnlwgt': [10], 'education': ['a']})
    val = pd.DataFrame({'age': [30], 'fnlwgt': [20], 'education': ['b']})
    test = pd.DataFrame({'age': [20, 30, 40], 'fnlwgt': [11, 21, 31], 'education': ['b', 'a', 'c']})
    assert evaluation.unseen_mask(pd.concat([train, val]), test).tolist() == [False, False, True]


def test_validation_parity(synthetic_partitions, fitted_models, tmp_path):
    """Verify both serialized schema types and frozen validation probabilities."""
    _, _, (_, _, Xv, yv), _ = synthetic_partitions
    models, specs, unseen = fitted_models
    for name, model in models.items():
        path = tmp_path / f'{name}.joblib'
        joblib.dump(model, path)
        required = evaluation.verify_pipeline(joblib.load(path), specs[name], Xv, yv, unseen)
        assert ('fnlwgt' in required) == (name == 'reference')


def test_validation_metric_mismatch(synthetic_partitions, fitted_models):
    """Reject altered validation metrics or incompatible threshold metadata."""
    _, _, (_, _, Xv, yv), _ = synthetic_partitions
    models, specs, unseen = fitted_models
    with pytest.raises(ValueError, match='Frozen validation val_f1 mismatch'):
        evaluation.verify_pipeline(models['xgboost'], dict(specs['xgboost'], val_f1=-1), Xv, yv, unseen)
    with pytest.raises(ValueError, match='Invalid frozen threshold'):
        evaluation.verify_pipeline(models['xgboost'], dict(specs['xgboost'], threshold=float('nan')), Xv, yv, unseen)


def test_score_uses_frozen_threshold_and_shared_mask():
    """Use each frozen threshold without re-optimization or default predictions."""
    model = Mock(classes_=np.array([1, 0]))
    model.predict_proba.return_value = np.array([[.4, .6], [.2, .8], [.45, .55], [.1, .9]])
    X = pd.DataFrame({'age': [10, 20, 30, 40]})
    y = pd.Series([1, 0, 1, 0])
    unseen = np.array([True, True, False, False])
    spec = {'candidate': 'xgboost', 'family': 'xgboost', 'variant': 'baseline',
            'representation': 'no_fnlwgt', 'threshold': .35, 'model_sha256': 'hash', 'run_id': 'run'}
    row = evaluation.score_model(model, spec, X, y, unseen)
    assert row['test_f1'] == row['unseen_test_f1'] == 1
    assert row['test_rows'] == 4 and row['unseen_test_rows'] == 2
    assert row['test_confusion_matrix'] == [[2, 0], [0, 2]]
    model.fit.assert_not_called()
    model.predict.assert_not_called()
    model.predict_proba.assert_called_once()


@pytest.mark.parametrize('selected,unseen,passed', [(.72, .73, True), (.71, .73, True), (.709, .75, False), (.75, .729, False)])
def test_performance_gate(selected, unseen, passed):
    """Require both predeclared margins, regardless of any alternative winner."""
    rows = [{'candidate': 'xgboost', 'test_f1': selected, 'unseen_test_f1': unseen},
            {'candidate': 'reference', 'test_f1': .72, 'unseen_test_f1': .74},
            {'candidate': 'other', 'test_f1': .99, 'unseen_test_f1': .99}]
    result = evaluation.performance_gate(rows, 'xgboost')
    assert result['passed'] is passed
    assert result['selected_candidate'] == 'xgboost'
    assert result['promotion_authorized'] is False


def test_alignment_checks_original_matrix(synthetic_partitions, fitted_models, tmp_path, monkeypatch):
    """Require historical target bytes and reference transform parity before scoring."""
    _, _, _, (X, y) = synthetic_partitions
    models, _, _ = fitted_models
    model = models['reference']
    transformed = model.named_steps['feature_names'].transform(model.named_steps['preprocess'].transform(X))
    for variant in evaluation.VARIANTS:
        folder = tmp_path / 'data/processed' / variant
        folder.mkdir(parents=True)
        y.to_csv(folder / 'y_test.csv', index=False)
        transformed.to_csv(folder / 'X_test.csv', index=False)
    folder = tmp_path / 'data/processed/baseline'
    hashes = {name: evaluation.training.file_sha256(folder / name) for name in ('X_test.csv', 'y_test.csv')}
    monkeypatch.setattr(evaluation, 'EXPECTED_ROWS', (0, 0, len(X)))
    result = evaluation.verify_test_alignment(tmp_path, X, y, model, {'dataset_sha256': hashes})
    assert len(result) == 3
    with pytest.raises(ValueError, match='predictor order'):
        evaluation.verify_test_alignment(tmp_path, X.iloc[::-1], y, model, {'dataset_sha256': hashes})
    with pytest.raises(AssertionError):
        evaluation.verify_test_alignment(tmp_path, X, 1-y, model, {'dataset_sha256': hashes})


def test_preflight_default_never_scores(monkeypatch):
    """Default CLI performs only the development preflight, never test evaluation."""
    context = evaluation.EvaluationContext('a'*32, 'synthetic', {'metadata': 'hash'})
    verified = ({'synthetic': {'model_sha256':'candidate-sha','val_f1':.72,'threshold':.35}},
                {}, {}, {'model_sha256': 'sha'}, (), context)
    monkeypatch.setattr(evaluation, 'preflight', Mock(return_value=verified))
    scorer = Mock()
    monkeypatch.setattr(evaluation, 'run_evaluation', scorer)
    args = ['--run-id', context.run_id]
    evaluation.main(args)
    scorer.assert_not_called()
    with pytest.raises(ValueError, match='approval SHA-256 mismatch'):
        evaluation.main(args + ['--evaluate'])
    scorer.assert_not_called()
    evaluation.main(args + ['--evaluate', '--approval-sha256', context.approval_sha256])
    scorer.assert_called_once_with(evaluation.ROOT, verified, approval_sha256=context.approval_sha256)


def test_failed_preflight_prevents_scoring(monkeypatch):
    """Even explicit evaluation cannot bypass an integrity failure."""
    monkeypatch.setattr(evaluation, 'preflight', Mock(side_effect=ValueError('bad model')))
    scorer = Mock()
    monkeypatch.setattr(evaluation, 'run_evaluation', scorer)
    with pytest.raises(ValueError, match='bad model'):
        evaluation.main(['--run-id', 'a'*32, '--evaluate'])
    scorer.assert_not_called()


def test_report_integration(synthetic_partitions, fitted_models, tmp_path, monkeypatch):
    """Generate aggregate comparison artifacts and log without altering frozen models."""
    raw, config, dev, (Xtest, ytest) = synthetic_partitions
    models, specs, _ = fitted_models
    root = tmp_path / 'project'
    raw_dir = root / 'data/raw'
    raw_dir.mkdir(parents=True)
    (raw_dir / 'adult.data').write_bytes(raw.read_bytes())
    (raw_dir / 'adult.test').write_bytes(b'fixture')
    for name, model in models.items():
        path = tmp_path / f'{name}.joblib'
        joblib.dump(model, path)
        specs[name].update(path=path, model_sha256=evaluation.training.file_sha256(path))
    unseen = evaluation.unseen_mask(pd.concat([dev[0], dev[2]]), Xtest)
    monkeypatch.setattr(evaluation, 'EXPECTED_UNSEEN', int(unseen.sum()))
    monkeypatch.setattr(evaluation, 'reconstruct_partitions', Mock(return_value=(Xtest, ytest)))
    monkeypatch.setattr(evaluation, 'verify_test_alignment', Mock(return_value={}))
    monkeypatch.setattr(evaluation.training, 'load_training_config', Mock(return_value={
        'paths': {'mlflow_db': 'mlflow.db', 'mlflow_artifacts': 'mlartifacts'}, 'mlflow': {'experiment': 'test'}}))
    monkeypatch.setattr(evaluation.training, 'configure_mlflow', Mock())
    tracker = MagicMock()
    monkeypatch.setattr(evaluation.training, 'mlflow', tracker)
    # Nonselected candidates are verified but never scored against the historical test set.
    other = Mock()
    other.predict_proba.side_effect = AssertionError('Nonselected candidate scored')
    models['other'] = other
    specs['other'] = dict(specs['xgboost'], candidate='other')
    context = evaluation.EvaluationContext('a'*32, 'xgboost', {})
    output = evaluation.run_evaluation(root, (specs, models, {}, {}, (*dev, config), context),
                                       approval_sha256=context.approval_sha256)
    summary = json.loads((output / 'heldout_summary.json').read_text())
    assert summary['test_rows'] == len(Xtest) and len(summary['candidates']) == 2
    assert summary['promotion_authorized'] is False
    assert summary['fairness_and_robustness_pending'] is True
    assert tracker.log_artifact.call_count == 2
    for spec in specs.values():
        assert evaluation.training.file_sha256(spec['path']) == spec['model_sha256']


def test_development_fingerprints(synthetic_partitions, tmp_path, monkeypatch):
    """Recompute actual CV and calibration identities without any test files."""
    from dataclasses import asdict
    from src.data.preprocess import create_cv_folds, save_config

    _, config, (X, y, Xv, yv), _ = synthetic_partitions
    X, y, Xv, yv = (obj.reset_index(drop=True) for obj in (X, y, Xv, yv))
    folder = tmp_path / 'data/processed'
    for variant in evaluation.VARIANTS:
        path = folder / variant
        path.mkdir(parents=True)
        X.assign(income=y).to_csv(path / 'raw_train.csv', index=False)
        Xv.assign(income=yv).to_csv(path / 'raw_val.csv', index=False)
        save_config(config, path / 'preprocessing_config.json')
    # Derive synthetic expected hashes using the frozen production protocol.
    folds = create_cv_folds(X, y, grouping_excluded_features=evaluation.GROUPING_EXCLUSIONS)
    fold_hash = hashlib.sha256(json.dumps([[p.tolist() for p in f] for f in folds]).encode()).hexdigest()
    fit, cal = evaluation.calibration_indices(X, y, .2, 42)
    config_hash = hashlib.sha256(json.dumps(asdict(replace(config, excluded_features=('fnlwgt',))), sort_keys=True).encode()).hexdigest()
    identity = {'raw_training_sha256': evaluation.training.file_sha256(folder / 'baseline/raw_train.csv'),
                'raw_validation_sha256': evaluation.training.file_sha256(folder / 'baseline/raw_val.csv'),
                'fold_sha256': fold_hash, 'calibration_split_sha256': hashlib.sha256(json.dumps([fit.tolist(), cal.tolist()]).encode()).hexdigest(),
                'fit_rows': len(fit), 'calibration_rows': len(cal), 'validation_rows': len(Xv),
                'unseen_validation_rows': int(evaluation.unseen_mask(X, Xv).sum())}
    plan = {'seed': 42, 'calibration_size': .2, 'candidates': {'synthetic': {'variant': 'baseline', 'effective_preprocessing_sha256': config_hash}}}
    monkeypatch.setattr(evaluation, 'EXPECTED_ROWS', (len(X), len(Xv), 0))
    actual = evaluation.verify_development(tmp_path, plan, identity)
    assert len(actual[0]) == len(X) and len(actual[2]) == len(Xv)
    assert not list(folder.rglob('*test*'))
    with pytest.raises(ValueError, match='Historical fold fingerprint changed'):
        evaluation.verify_development(tmp_path, plan, dict(identity, fold_sha256='bad'))
    with (folder / 'robust/raw_train.csv').open('a') as handle:
        handle.write('\n')
    with pytest.raises(ValueError, match='File checksum mismatch'):
        evaluation.verify_development(tmp_path, plan, identity)


def test_reference_historical_matrix(synthetic_partitions, fitted_models, tmp_path):
    """Verify legacy validation CSV bytes, row alignment and original classifier scoring."""
    _, _, (_, _, Xv, yv), _ = synthetic_partitions
    models, _, _ = fitted_models
    model = models['reference']
    folder = tmp_path / 'data/processed/baseline'
    folder.mkdir(parents=True)
    matrix = model.named_steps['feature_names'].transform(model.named_steps['preprocess'].transform(Xv))
    matrix.to_csv(folder / 'X_val.csv', index=False)
    yv.to_csv(folder / 'y_val.csv', index=False)
    ref = {'preprocessing_variant': 'baseline', 'dataset_sha256': {
        name: evaluation.training.file_sha256(folder / name) for name in ('X_val.csv', 'y_val.csv')}}
    prob = evaluation.reference_validation_probabilities(tmp_path, model, Xv, yv, ref)
    expected = binary.positive_probabilities(model.named_steps['classifier'], pd.read_csv(folder / 'X_val.csv'))
    np.testing.assert_array_equal(prob, expected)
    with pytest.raises(AssertionError):
        evaluation.reference_validation_probabilities(tmp_path, model, Xv, 1-yv, ref)
    with pytest.raises(ValueError, match='matrix differs'):
        evaluation.reference_validation_probabilities(tmp_path, model, Xv.iloc[::-1], yv, ref)
    with (folder / 'X_val.csv').open('a') as stream:
        stream.write('changed')
    with pytest.raises(ValueError, match='checksum mismatch'):
        evaluation.reference_validation_probabilities(tmp_path, model, Xv, yv, ref)


def test_reference_historical_ranking_parity(synthetic_partitions, fitted_models, monkeypatch):
    """Accept changed rankings only on legacy path while enforcing frozen decisions and scores."""
    _, _, (_, _, Xv, yv), _ = synthetic_partitions
    models, specs, unseen = fitted_models
    model = models['reference']
    spec = {k: v for k, v in specs['reference'].items() if not k.startswith(('val_', 'unseen_val_'))}
    raw = np.where(yv.to_numpy() == 1, .8, .7)
    historic = np.where(yv.to_numpy() == 1, .7, .8)
    spec.update({f'val_{k}': v for k, v in binary.evaluate(yv, historic, spec['threshold']).items()})
    monkeypatch.setattr(binary, 'positive_probabilities', Mock(return_value=raw))
    with pytest.raises(ValueError, match='val_pr_auc mismatch'):
        evaluation.verify_pipeline(model, spec, Xv, yv, unseen)
    evaluation.verify_pipeline(model, spec, Xv, yv, unseen, historical_prob=historic)
    with pytest.raises(ValueError, match='val_pr_auc mismatch'):
        evaluation.verify_pipeline(model, dict(spec, val_pr_auc=-1), Xv, yv, unseen, historical_prob=historic)
    bad = historic.copy()
    bad[0] = .1
    with pytest.raises(ValueError, match='class predictions differ'):
        evaluation.verify_pipeline(model, spec, Xv, yv, unseen, historical_prob=bad)


def test_validation_winner_and_approval_binding():
    """Select an arbitrary validation winner and bind approval to the active reference."""
    rows = [{'candidate': 'future', 'val_f1': .75, 'test_f1': .1},
            {'candidate': 'other', 'val_f1': .7, 'test_f1': .99}]
    assert evaluation.select_validation_winner(rows) == 'future'
    with pytest.raises(ValueError, match='Validation tie'):
        evaluation.select_validation_winner([dict(r, val_f1=.75) for r in rows])
    with pytest.raises(ValueError, match='Validation F1 gate'):
        evaluation.select_validation_winner([dict(rows[0], val_f1=.6)])
    ctx = evaluation.EvaluationContext('a'*32, 'future', {'reference': 'old'})
    assert ctx.approval_sha256 != evaluation.EvaluationContext('a'*32, 'future', {'reference': 'new'}).approval_sha256
    assert ctx.approval_sha256 != evaluation.EvaluationContext('b'*32, 'future', ctx.anchors).approval_sha256


def test_reduced_reference_test_alignment(synthetic_partitions, tmp_path, monkeypatch):
    """Check saved hashes without transforming the full matrix with a reduced reference."""
    _, _, _, (X, y) = synthetic_partitions
    saved = {}
    for variant in evaluation.VARIANTS:
        folder = tmp_path / 'data/processed' / variant
        folder.mkdir(parents=True)
        X.to_csv(folder / 'X_test.csv', index=False)
        y.to_csv(folder / 'y_test.csv', index=False)
        saved[variant] = {name: evaluation.training.file_sha256(folder / name)
                          for name in ('X_test.csv', 'y_test.csv')}
    monkeypatch.setattr(evaluation, 'EXPECTED_ROWS', (0, 0, len(X)))
    model = Mock()
    ref = {'evidence': {'heldout': {'saved_test_sha256': saved}}}
    assert evaluation.verify_test_alignment(tmp_path, X, y, model, ref) == saved
    assert not model.mock_calls
    (tmp_path / 'data/processed/robust/X_test.csv').write_text('tampered')
    with pytest.raises(ValueError, match='checksum mismatch'):
        evaluation.verify_test_alignment(tmp_path, X, y, model, ref)


def test_preflight_future_run_with_promoted_reference(synthetic_partitions, fitted_models, tmp_path, monkeypatch):
    """Verify a new arbitrary shortlist against a reduced reference with no test files."""
    from dataclasses import asdict
    from src.data.preprocess import create_cv_folds, save_config

    _, config, (X, y, Xv, yv), _ = synthetic_partitions
    X, y, Xv, yv = (obj.reset_index(drop=True) for obj in (X, y, Xv, yv))
    models, specs, _ = fitted_models
    for variant in evaluation.VARIANTS:
        folder = tmp_path / 'data/processed' / variant
        folder.mkdir(parents=True)
        X.assign(income=y).to_csv(folder / 'raw_train.csv', index=False)
        Xv.assign(income=yv).to_csv(folder / 'raw_val.csv', index=False)
        save_config(config, folder / 'preprocessing_config.json')
    baseline = tmp_path / 'data/processed/baseline'
    folds = create_cv_folds(X, y, grouping_excluded_features=evaluation.GROUPING_EXCLUSIONS)
    fit, cal = evaluation.calibration_indices(X, y, .2, 42)
    run_id = 'a'*32
    identity = dict(execution_id=run_id, evaluation_protocol='shortlist_validation', test_accessed=False,
                    representation='no_fnlwgt', raw_training_sha256=evaluation.training.file_sha256(baseline/'raw_train.csv'),
                    raw_validation_sha256=evaluation.training.file_sha256(baseline/'raw_val.csv'),
                    fold_sha256=hashlib.sha256(json.dumps([[p.tolist() for p in f] for f in folds]).encode()).hexdigest(),
                    grouping_excluded_features=list(evaluation.GROUPING_EXCLUSIONS), overlap_excluded_features=list(evaluation.OVERLAP_EXCLUSIONS),
                    fit_rows=len(fit), calibration_rows=len(cal), validation_rows=len(Xv),
                    unseen_validation_rows=int(evaluation.unseen_mask(X, Xv).sum()),
                    calibration_split_sha256=hashlib.sha256(json.dumps([fit.tolist(), cal.tolist()]).encode()).hexdigest())
    candidate = 'future_candidate'
    run = tmp_path/'models/runs'/run_id
    folder = run/candidate
    folder.mkdir(parents=True)
    model = models['xgboost']
    path = folder/'model_pipeline.joblib'
    joblib.dump(model, path)
    spec = dict(specs['xgboost'], candidate=candidate, model_sha256=evaluation.training.file_sha256(path))
    spec.update(effective_preprocessing_sha256=hashlib.sha256(json.dumps(asdict(replace(config, excluded_features=('fnlwgt',))),sort_keys=True).encode()).hexdigest())
    plan = dict(seed=42, calibration_size=.2, representation='no_fnlwgt', raw_training_sha256=identity['raw_training_sha256'],
                fold_sha256=identity['fold_sha256'], candidates={candidate:{k:spec[k] for k in ('family','variant','effective_preprocessing_sha256')}})
    for path, value in ((folder/'validation_result.json',spec), (run/'frozen_plan.json',plan),
                        (run/'partition_identity.json',identity), (run/'candidate_validation_summary.json',dict(identity,candidates=[spec]))):
        path.write_text(json.dumps(value))
    ref = tmp_path/'models/reference'
    ref.mkdir()
    joblib.dump(model, ref/'best_model_pipeline.joblib')
    metadata = dict(model_filename='best_model_pipeline.joblib', model_sha256=evaluation.training.file_sha256(ref/'best_model_pipeline.joblib'),
                    positive_class=1, representation='no_fnlwgt', model_family='logistic_regression', preprocessing_variant='baseline',
                    decision_threshold=spec['threshold'], validation_f1=spec['val_f1'], validation_pr_auc=spec['val_pr_auc'],
                    validation_roc_auc=spec['val_roc_auc'], mlflow_run_id='previous',
                    dataset_sha256={'raw_train.csv':identity['raw_training_sha256'],'raw_val.csv':identity['raw_validation_sha256']})
    (ref/'best_model_metadata.json').write_text(json.dumps(metadata))
    monkeypatch.setattr(evaluation, 'EXPECTED_ROWS', (len(X),len(Xv),0))
    monkeypatch.setattr(evaluation, 'MIN_VALIDATION_F1', 0.)  # Synthetic models need not meet the production gate.
    verified = evaluation.preflight(tmp_path, run_id)
    assert verified[-1].selected == candidate
    assert verified[0]['reference']['representation'] == 'no_fnlwgt'
    assert not (tmp_path/'data/raw').exists()
    # A changed frozen reference changes the approval identity even with identical model bytes.
    metadata['mlflow_run_id'] = 'another-release'
    (ref/'best_model_metadata.json').write_text(json.dumps(metadata))
    changed = evaluation.preflight(tmp_path, run_id)
    assert changed[-1].approval_sha256 != verified[-1].approval_sha256
    with pytest.raises(ValueError, match='Preflight approval'):
        evaluation.run_evaluation(tmp_path, changed, approval_sha256=verified[-1].approval_sha256)


def test_paired_robustness_uses_exact_scenarios(synthetic_partitions, fitted_models, tmp_path):
    """Bind both robustness reports to verified drift data and reject changed labels."""
    from scripts import evaluate_fairness as audit
    from src.data import drift_generator as drift

    _, _, _, (X, y) = synthetic_partitions
    models, specs, _ = fitted_models
    root = tmp_path/'data/drift'
    drift.save_drift_scenarios(X, y, root, verification={'verified': True})
    mask = np.ones(len(X), dtype=bool)
    report = {'execution_id':'b'*32, 'candidates':[evaluation.score_model(model,specs[name],X,y,mask)
                                                for name,model in models.items()]}
    context = evaluation.EvaluationContext('a'*32,'xgboost',{})
    target = root/'severe/y_test.csv'
    original = target.read_bytes()
    (1-y).to_csv(target,index=False)
    with pytest.raises(AssertionError):
        audit.run_paired_robustness(tmp_path, report, specs, models, X, y, context)
    target.write_bytes(original)
    output = audit.run_paired_robustness(tmp_path, report, specs, models, X, y, context)
    for name in ('xgboost','reference'):
        result = evaluation.read_json(output/name/'robustness_report.json')
        assert result['heldout_execution_id'] == report['execution_id']
        assert set(result['scenario_file_sha256']) == set(drift.SCENARIOS)
