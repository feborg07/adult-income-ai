"""Audit frozen selected and reference pipelines on the approved historical test rows.

Run from the repository root: python -m scripts.evaluate_fairness
This is a descriptive audit, not model selection or a promotion authorization.
"""
import argparse
import hashlib
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from scripts import evaluate_candidates as heldout
from src.models.tasks import binary_classification as binary

ROOT = heldout.ROOT
GROUPINGS = (('sex',), ('race',), ('sex', 'race'))
# Descriptive support flag, not an acceptance threshold or significance test.
MIN_SUPPORT = 30
METRICS = ('selection_rate', 'recall', 'false_positive_rate', 'false_negative_rate', 'precision', 'f1')


def safe_rate(numerator, denominator):
    """Return a rate or None when its denominator is zero."""
    return float(numerator / denominator) if denominator else None


def subgroup_metrics(y, predictions):
    """Compute unweighted counts and rates for positive income class 1."""
    y, predictions = np.asarray(y), np.asarray(predictions)
    heldout.require(y.ndim == predictions.ndim == 1 and len(y) == len(predictions) > 0,
                    'Invalid subgroup alignment.')
    heldout.require(np.isin(y, [0, 1]).all() and np.isin(predictions, [0, 1]).all(),
                    'Subgroup targets and predictions must be binary.')
    tp = int(((y == 1) & (predictions == 1)).sum())
    tn = int(((y == 0) & (predictions == 0)).sum())
    fp = int(((y == 0) & (predictions == 1)).sum())
    fn = int(((y == 1) & (predictions == 0)).sum())
    n, positives, negatives = len(y), tp + fn, tn + fp
    return {'rows': n, 'actual_positive': positives, 'actual_negative': negatives,
            'predicted_positive': tp + fp, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'prevalence': safe_rate(positives, n), 'selection_rate': safe_rate(tp + fp, n),
            'precision': safe_rate(tp, tp + fp), 'recall': safe_rate(tp, positives),
            'false_positive_rate': safe_rate(fp, negatives),
            'false_negative_rate': safe_rate(fn, positives),
            'f1': safe_rate(2 * tp, 2 * tp + fp + fn), 'accuracy': safe_rate(tp + tn, n),
            'low_support': min(n, positives, negatives, tp + fp) < MIN_SUPPORT}


def group_metrics(X, y, predictions, columns):
    """Compute metrics for all observed demographic groups without dropping missing values."""
    heldout.require(len(X) == len(y) == len(predictions), 'Demographic alignment mismatch.')
    heldout.require(set(columns).issubset(X.columns), 'Missing demographic columns.')
    demographics = X[list(columns)].reset_index(drop=True).astype('string').fillna('<missing>')
    y, predictions = np.asarray(y), np.asarray(predictions)
    rows = []
    # Use positional indices so masks remain aligned with reconstructed test rows.
    for key, indices in demographics.groupby(list(columns), sort=True, observed=True).indices.items():
        key = key if isinstance(key, tuple) else (key,)
        rows.append({'group': ' | '.join(map(str, key)),
                     **dict(zip(columns, map(str, key))),
                     **subgroup_metrics(y[indices], predictions[indices])})
    heldout.require(sum(row['rows'] for row in rows) == len(X), 'Subgroup coverage mismatch.')
    return rows


def disparity_summary(rows):
    """Compute max-minus-min gaps; return None if any group rate is undefined."""
    result = {'group_count': len(rows), 'low_support_groups': sum(row['low_support'] for row in rows)}
    for metric in METRICS:
        values = [row[metric] for row in rows if row[metric] is not None]
        result[f'{metric}_defined_groups'] = len(values)
        result[f'{metric}_gap'] = (max(values) - min(values)
                                  if len(values) == len(rows) and len(values) >= 2 else None)
    # Equal opportunity compares TPR; equalized odds compares both TPR and FPR.
    result['demographic_parity_difference'] = result['selection_rate_gap']
    rates = [row['selection_rate'] for row in rows]
    result['demographic_parity_ratio'] = (safe_rate(min(rates), max(rates))
                                          if len(rates) >= 2 else None)
    result['equal_opportunity_difference'] = result['recall_gap']
    gaps = [result['recall_gap'], result['false_positive_rate_gap']]
    result['equalized_odds_difference'] = max(gaps) if all(g is not None for g in gaps) else None
    return result


def verify_report(report, identity, specs, context):
    """Bind the audit to the completed held-out evaluation and frozen model identities."""
    expected = {'source_execution_id': context.run_id, 'evaluation_protocol': 'frozen_heldout',
                'selected_candidate': context.selected, 'partition_identity': identity,
                'preflight_approval_sha256': context.approval_sha256,
                'frozen_metadata_content_sha256': context.anchors,
                'test_accessed': True, 'historical_test_previously_used': True,
                'promotion_authorized': False, 'test_rows': heldout.EXPECTED_ROWS[2],
                'unseen_test_rows': heldout.EXPECTED_UNSEEN,
                'overlap_excluded_features': list(heldout.OVERLAP_EXCLUSIONS)}
    heldout.require(all(report.get(k) == v for k, v in expected.items()), 'Held-out report identity mismatch.')
    rows = report['candidates']
    by_name = {row['candidate']: row for row in rows}
    heldout.require(len(rows) == len(by_name) == 2 and set(by_name) == {context.selected, heldout.REFERENCE},
                    'Held-out shortlist mismatch.')
    for name in by_name:
        spec = specs[name]
        heldout.require(all(by_name[name].get(k) == spec[k] for k in
                           ('model_sha256', 'threshold', 'representation', 'run_id', 'family', 'variant')),
                        f'Held-out model identity mismatch: {name}')
    gate = heldout.performance_gate(rows, context.selected)
    heldout.require(gate == report['performance_gate'] and gate['passed'], 'Held-out performance gate mismatch.')
    return by_name


def verify_predictions(y, probabilities, threshold, expected, prefix):
    """Require aggregate metrics to reproduce the already reported held-out results."""
    metrics = binary.evaluate(y, probabilities, threshold)
    heldout.require(len(y) == expected[f'{prefix}_rows'], 'Held-out prediction row mismatch.')
    for key, value in metrics.items():
        recorded = expected[f'{prefix}_{key}']
        matches = (np.isclose(value, recorded, rtol=0, atol=1e-10)
                   if isinstance(value, float) else value == recorded)
        heldout.require(matches, f'Held-out metric mismatch: {expected["candidate"]} {prefix}_{key}')


def audit_predictions(X, y, probabilities, threshold, unseen, expected):
    """Audit full and unseen subsets after verifying aggregate prediction parity."""
    heldout.require(len(X) == len(y) == len(probabilities) == len(unseen), 'Audit alignment mismatch.')
    rows, gaps = [], []
    predictions = (probabilities >= threshold).astype(int)
    for prefix, mask in [('test', np.ones(len(X), dtype=bool)), ('unseen_test', unseen)]:
        indices = np.flatnonzero(mask)
        target, frame = y.iloc[indices], X.iloc[indices]
        verify_predictions(target, probabilities[mask], threshold, expected, prefix)
        for columns in GROUPINGS:
            metrics = group_metrics(frame, target, predictions[mask], columns)
            context = {'candidate': expected['candidate'], 'partition': prefix,
                       'attribute': '+'.join(columns), 'threshold': threshold}
            rows.extend(dict(context, **row) for row in metrics)
            gaps.append(dict(context, **disparity_summary(metrics)))
    return rows, gaps


def run_audit(root, summary_path, *, include_robustness=False):
    """Verify frozen provenance, reconstruct test rows, and write aggregate-only audit reports."""
    report = heldout.read_json(summary_path)
    specs, models, identity, ref, development, context = heldout.preflight(root, report['source_execution_id'])
    expected = verify_report(report, identity, specs, context)
    X, y, Xval, yval, config = development
    raw = [root / 'data/raw' / name for name in ('adult.data', 'adult.test')]
    for path in raw:
        heldout.verify_file(path, report['raw_source_sha256'][path.name])
    # Use the same reconstruction and alignment checks as the final evaluator.
    Xtest, ytest = heldout.reconstruct_partitions(raw, config, (X, y, Xval, yval))
    hashes = heldout.verify_test_alignment(root, Xtest, ytest, models[heldout.REFERENCE], ref)
    heldout.require(hashes == report['saved_test_sha256'], 'Held-out dataset fingerprint mismatch.')
    unseen = heldout.unseen_mask(pd.concat([X, Xval], ignore_index=True), Xtest)
    mask_sha = hashlib.sha256(unseen.astype('uint8').tobytes()).hexdigest()
    heldout.require(int(unseen.sum()) == heldout.EXPECTED_UNSEEN and mask_sha == report['unseen_mask_sha256'],
                    'Held-out unseen subset mismatch.')
    rows, gaps = [], []
    for name in (context.selected, heldout.REFERENCE):
        probabilities = binary.positive_probabilities(models[name], Xtest)
        metrics, disparities = audit_predictions(Xtest, ytest, probabilities, specs[name]['threshold'], unseen, expected[name])
        rows.extend(metrics)
        gaps.extend(disparities)
    for spec in specs.values():
        heldout.verify_file(spec['path'], spec['model_sha256'])
    if include_robustness:
        run_paired_robustness(root, report, specs, models, Xtest, ytest, context)
    execution = uuid.uuid4().hex
    output = root / 'reports/fairness' / execution
    output.mkdir(parents=True, exist_ok=False)
    audit = {'execution_id': execution, 'heldout_execution_id': report['execution_id'],
             'heldout_summary_sha256': heldout.training.file_sha256(summary_path),
             'source_execution_id': context.run_id, 'preflight_approval_sha256': context.approval_sha256, 'git_commit': heldout.training.git_revision(),
             'evaluator_sha256': heldout.training.file_sha256(Path(__file__)),
             'heldout_evaluator_sha256': heldout.training.file_sha256(Path(heldout.__file__)),
             'raw_source_sha256': report['raw_source_sha256'], 'saved_test_sha256': hashes,
             'unseen_mask_sha256': mask_sha, 'test_rows': len(Xtest), 'unseen_test_rows': int(unseen.sum()),
             'models': {name: {k: specs[name][k] for k in ('model_sha256', 'threshold', 'representation')}
                        for name in (context.selected, heldout.REFERENCE)},
             'methodology': {'positive_class': 1, 'weighting': 'unweighted observations for both models',
                             'groupings': [list(g) for g in GROUPINGS], 'gap': 'maximum minus minimum across observed groups',
                             'undefined_rate': 'null; gaps undefined if any group rate is undefined',
                             'low_support_rule': f'fewer than {MIN_SUPPORT} rows, actual positives, actual negatives or predicted positives',
                             'equalized_odds': 'maximum of TPR and FPR gaps',
                             'limitations': ['Historical test partition was previously used.',
                                             'Descriptive disparities only; small groups may be unstable.',
                                             'No causal, statistical significance or legal fairness conclusion.',
                                             'No quantitative fairness acceptance threshold was agreed.']},
             'fairness_review_required': True, 'promotion_authorized': False,
             'subgroups': rows, 'disparities': gaps}
    pd.DataFrame(rows).to_csv(output / 'subgroup_metrics.csv', index=False)
    pd.DataFrame(gaps).to_csv(output / 'disparity_summary.csv', index=False)
    heldout.training.atomic_json_write(output / 'fairness_summary.json', audit)
    # Log only aggregate metrics and artifacts; never publish individual predictions.
    training = heldout.training
    config_map = training.load_training_config(root / 'configs/config.yml')
    paths = config_map['paths']
    training.configure_mlflow(config_map['mlflow']['experiment'] + '-fairness-evaluation',
                              root / paths['mlflow_db'], root / paths['mlflow_artifacts'])
    with training.mlflow.start_run(run_name=f'fairness-{execution}'):
        training.mlflow.set_tags({'heldout_execution_id': report['execution_id'], 'promotion_authorized': 'false'})
        training.mlflow.log_metrics({f'{row["candidate"]}.{row["partition"]}.{row["attribute"].replace("+", "_")}.{key}': value
                                     for row in gaps for key, value in row.items()
                                     if key not in ('threshold',) and isinstance(value, (float, int))})
        for path in sorted(output.iterdir()):
            training.mlflow.log_artifact(str(path), artifact_path='fairness')
    print(f'Fairness reports: {output}')
    print('Descriptive fairness assessment complete; review required; promotion not authorized.')
    return output


def run_paired_robustness(root, report, specs, models, Xtest, ytest, context):
    """Score both frozen models against the same verified deterministic drift scenarios."""
    from src.data import drift_generator as drift
    from src.models import robustness

    run_id = report['execution_id']
    heldout.require(isinstance(run_id, str) and len(run_id) == 32 and
                    all(c in '0123456789abcdef' for c in run_id), 'Invalid held-out execution ID.')
    drift_root = root / 'data/drift'
    metadata_path = drift_root / 'drift_metadata.json'
    metadata = heldout.read_json(metadata_path)
    heldout.require(metadata['reference_verification']['verified'] is True and
                    metadata['reference_rows'] == len(Xtest) and
                    set(metadata['scenarios']) == set(drift.SCENARIOS), 'Drift provenance mismatch.')
    scenarios, hashes = {}, {}
    for scenario in drift.SCENARIOS:
        path = drift_root / scenario
        X, y = robustness.load_dataset(path / 'X_test.csv', path / 'y_test.csv')
        expected = drift.generate_drift_scenario(Xtest, scenario, metadata['random_state'])
        heldout.assert_partition_equal(X, y, expected, ytest, f'drift {scenario}')
        heldout.require(metadata['scenarios'][scenario]['sha256'] == drift.dataframe_sha256(expected),
                        f'Drift fingerprint mismatch: {scenario}')
        scenarios[scenario] = (X, y)
        hashes[scenario] = {name: heldout.training.file_sha256(path / name)
                            for name in ('X_test.csv', 'y_test.csv')}
    results = {}
    for name in (context.selected, heldout.REFERENCE):
        spec = specs[name]
        scores = {scenario: robustness.evaluate_model(models[name], X, y, spec['threshold'])
                  for scenario, (X, y) in scenarios.items()}
        control = next(row for row in report['candidates'] if row['candidate'] == name)
        for key, value in scores['control'].items():
            field = 'test_' + key
            if field in control:
                heldout.require(value == control[field], f'Robustness control mismatch: {name}/{key}')
        results[name] = scores
    output = root / 'reports/robustness' / run_id
    output.mkdir(parents=True, exist_ok=False)
    for name, scores in results.items():
        spec = specs[name]
        meta = {'model_family': spec['family'], 'preprocessing_variant': spec['variant'],
                'model_sha256': spec['model_sha256']}
        path, _ = robustness.save_reports(scores, robustness.compare_to_control(scores), meta,
                                           output / name, spec['threshold'])
        record = heldout.read_json(path)
        record.update(heldout_execution_id=run_id, drift_metadata_sha256=heldout.canonical_sha(metadata),
                      scenario_file_sha256=hashes)
        heldout.training.atomic_json_write(path, record)
    print(f'Paired robustness reports: {output}')
    return output


def main(argv=None):
    """Run the audit against the previously approved held-out summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--heldout-summary', type=Path, required=True,
                        help='Path to the completed held-out summary JSON.')
    parser.add_argument('--robustness', action='store_true', help='Also score paired deterministic drift scenarios.')
    args = parser.parse_args(argv)
    run_audit(ROOT, ROOT / args.heldout_summary, include_robustness=args.robustness)


if __name__ == '__main__':
    main()
