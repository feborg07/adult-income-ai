"""Validate frozen CV candidates on the original validation partition, never test."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import uuid
import re
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from src.data.preprocess import (
    PreprocessingConfig, build_preprocessor, create_cv_folds,
    create_train_test_split, load_untransformed_partition, predictor_group_ids,
)
from src.models import train as training
from src.models.cross_validation import COMMON_GROUPING_EXCLUSIONS
from src.models.configuration import read_mapping
from src.models.tasks import binary_classification as binary

ROOT = Path(__file__).resolve().parents[1]
GROUPING_EXCLUSIONS = COMMON_GROUPING_EXCLUSIONS
# All shortlisted preprocessing variants remove the redundant education string.
OVERLAP_EXCLUSIONS = ('fnlwgt', 'education')


def unseen_mask(train, validation, excluded=OVERLAP_EXCLUSIONS):
    """Identify validation vectors absent from the entire original training partition."""
    combined = pd.concat([train, validation], ignore_index=True).drop(columns=list(excluded))
    groups = predictor_group_ids(combined)
    return ~np.isin(groups[len(train):], groups[:len(train)])


def calibration_indices(X, y, size, seed):
    """Split original training rows using the same conservative grouping basis as CV."""
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    grouped = X.drop(columns=list(GROUPING_EXCLUSIONS))
    fit, cal, yfit, ycal = create_train_test_split(grouped, y, test_size=size,
                                                   random_state=seed, duplicate_aware=True)
    binary.require_both_classes(yfit, 'fit')
    binary.require_both_classes(ycal, 'calibration')
    ids = predictor_group_ids(grouped)
    if np.intersect1d(ids[fit.index], ids[cal.index]).size:
        raise RuntimeError('Calibration groups overlap with fitting groups.')
    return fit.index.to_numpy(), cal.index.to_numpy()


def evaluate_candidate(estimator, config, X, y, Xval, yval, fit, cal, unseen):
    """Fit fresh preprocessing and estimator, freeze threshold, then score validation."""
    prep = build_preprocessor(config)
    Xfit = pd.DataFrame(prep.fit_transform(X.iloc[fit]), columns=prep.get_feature_names_out())
    Xcal = pd.DataFrame(prep.transform(X.iloc[cal]), columns=Xfit.columns)
    Xscore = pd.DataFrame(prep.transform(Xval), columns=Xfit.columns)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        estimator.fit(Xfit, y.iloc[fit])
    threshold = binary.optimal_threshold(y.iloc[cal], binary.positive_probabilities(estimator, Xcal))
    probabilities = binary.positive_probabilities(estimator, Xscore)
    metrics = {}
    for label, mask in [('val', np.ones(len(Xval), dtype=bool)), ('unseen_val', unseen)]:
        target = yval.iloc[np.flatnonzero(mask)]
        binary.require_both_classes(target, label)
        values = binary.evaluate(target, probabilities[mask], threshold)
        values['f1_default_050'] = binary.default_f1(target, probabilities[mask])
        metrics.update({f'{label}_{key}': value for key, value in values.items()})
    pipeline = training.build_deployable_pipeline(prep, estimator, Xfit.columns, Xfit)
    if not np.allclose(binary.positive_probabilities(pipeline, Xval), probabilities):
        raise RuntimeError('Deployable pipeline predictions differ.')
    return pipeline, threshold, metrics, sum(issubclass(w.category, ConvergenceWarning) for w in caught)


def run_validation(args):
    """Verify frozen provenance, fit shortlisted models and log separate MLflow runs."""
    plan = read_mapping(args.plan)
    config = training.load_training_config(args.config)
    paths = config['paths']
    processed = Path(args.processed_dir or ROOT / paths['processed_dir']).resolve()
    output_root = Path(args.output_dir or ROOT / paths['output_dir']).resolve()
    reference = (ROOT / 'models/reference').resolve()
    if output_root == reference or reference in output_root.parents or output_root in reference.parents:
        raise ValueError('Validation output must be separate from the reference model.')
    rep = training.resolve_cv_representation(config)
    if rep['representation'] != plan['representation'] or rep['representation'] != 'no_fnlwgt':
        raise ValueError('Frozen representation mismatch.')
    settings = training.resolve_cv_settings(config)
    if config['seed'] != plan['seed'] or settings['calibration_size'] != plan['calibration_size']:
        raise ValueError('Frozen calibration settings mismatch.')
    if settings['n_splits'] != plan.get('n_splits', 3):
        raise ValueError('Frozen fold count mismatch.')
    candidates = plan['candidates']
    if not candidates or any(not isinstance(name, str) or not re.fullmatch(r'[a-zA-Z0-9_-]+', name)
                             or name == 'reference' for name in candidates):
        raise ValueError('Invalid candidate identifiers.')
    variants = sorted({spec['variant'] for spec in candidates.values()})
    for variant in variants:
        if training.file_sha256(processed / variant / 'raw_train.csv') != plan['raw_training_sha256']:
            raise ValueError('Original training partition changed.')
    val_hashes = {training.file_sha256(processed / v / 'raw_val.csv') for v in variants}
    if len(val_hashes) != 1:
        raise ValueError('Validation rows differ across variants.')
    X, y = load_untransformed_partition(processed / variants[0], 'train')
    Xval, yval = load_untransformed_partition(processed / variants[0], 'val')
    # Verify the original CV fold identity before any fitting.
    indices = create_cv_folds(X, y, **settings, random_state=plan['seed'],
                              grouping_excluded_features=GROUPING_EXCLUSIONS)
    fold_hash = hashlib.sha256(json.dumps([[p.tolist() for p in fold] for fold in indices]).encode()).hexdigest()
    if fold_hash != plan['fold_sha256']:
        raise ValueError('Historical CV fold identity changed.')
    effective = {}
    for variant in variants:
        base = PreprocessingConfig(**json.loads((processed / variant / 'preprocessing_config.json').read_text()))
        if base.excluded_features:
            raise ValueError('Expected original full-input preprocessing configuration.')
        effective[variant] = replace(base, excluded_features=rep['excluded_features'])
    prototypes = {}
    for name, spec in candidates.items():
        digest = hashlib.sha256(json.dumps(asdict(effective[spec['variant']]), sort_keys=True).encode()).hexdigest()
        if digest != spec['effective_preprocessing_sha256']:
            raise ValueError(f'Preprocessing configuration changed: {name}')
        prototypes[name] = training.ALGORITHM_REGISTRY.get(spec['family'], task_type='binary_classification').build(spec['parameters'], plan['seed'])
    fit, cal = calibration_indices(X, y, plan['calibration_size'], plan['seed'])
    unseen = unseen_mask(X, Xval)
    binary.require_both_classes(yval.iloc[np.flatnonzero(unseen)], 'unseen validation')
    # Record actual assignments and immutable source configuration.
    execution = uuid.uuid4().hex
    output = output_root / execution
    output.mkdir(parents=True, exist_ok=False)
    metadata = {'execution_id': execution, 'evaluation_protocol': 'shortlist_validation',
                'git_commit': training.git_revision(), 'plan_sha256': training.file_sha256(args.plan),
                'raw_training_sha256': plan['raw_training_sha256'], 'raw_validation_sha256': next(iter(val_hashes)),
                'fold_sha256': fold_hash, 'representation': rep['representation'],
                'grouping_excluded_features': list(GROUPING_EXCLUSIONS), 'overlap_excluded_features': list(OVERLAP_EXCLUSIONS),
                'fit_rows': len(fit), 'calibration_rows': len(cal), 'validation_rows': len(Xval),
                'unseen_validation_rows': int(unseen.sum()), 'test_accessed': False,
                'calibration_split_sha256': hashlib.sha256(json.dumps([fit.tolist(), cal.tolist()]).encode()).hexdigest()}
    training.atomic_json_write(output / 'frozen_plan.json', plan)
    training.atomic_json_write(output / 'partition_identity.json', metadata)
    training.configure_mlflow(config['mlflow']['experiment'] + '-shortlist-validation',
                              ROOT / paths['mlflow_db'], ROOT / paths['mlflow_artifacts'])
    rows = []
    for name, spec in candidates.items():
        folder = output / name
        folder.mkdir()
        with training.mlflow.start_run(run_name=f'validate-{name}') as run:
            training.mlflow.set_tags({'evaluation_protocol': 'shortlist_validation', 'training_execution_id': execution,
                                      'source_cv_run_id': spec['source_run_id'], 'candidate': name,
                                      'model_family': spec['family'], 'preprocessing_variant': spec['variant'],
                                      'candidate_for_promotion': 'false'})
            training.mlflow.log_params({'seed': plan['seed'], 'variant': spec['variant'], 'model': spec['family'],
                                        'representation': rep['representation'], 'data_version': plan['raw_training_sha256'],
                                        'raw_validation_sha256': metadata['raw_validation_sha256'],
                                        'calibration_split_sha256': metadata['calibration_split_sha256'],
                                        'effective_preprocessing_sha256': spec['effective_preprocessing_sha256'],
                                        'threshold_policy': 'training_only_group_calibration',
                                        **{f'model_{k}': str(v) for k,v in spec['parameters'].items()}})
            pipeline, threshold, metrics, warning_count = evaluate_candidate(prototypes[name], effective[spec['variant']], X, y, Xval, yval, fit, cal, unseen)
            row = {'candidate': name, 'run_id': run.info.run_id, **spec, 'threshold': threshold,
                   **metrics, 'convergence_warning_count': warning_count}
            model_path = folder / 'model_pipeline.joblib'
            joblib.dump(pipeline, model_path)
            if not np.allclose(binary.positive_probabilities(joblib.load(model_path), Xval), binary.positive_probabilities(pipeline, Xval)):
                raise RuntimeError('Serialized candidate predictions differ.')
            row['model_sha256'] = training.file_sha256(model_path)
            training.atomic_json_write(folder / 'validation_result.json', row)
            training.mlflow.log_metrics({'cv_reference_f1': spec['cv_f1'], 'decision_threshold': threshold,
                                         'convergence_warning_count': warning_count,
                                         **{k:v for k,v in metrics.items() if not isinstance(v,list)}})
            for artifact in (model_path, folder / 'validation_result.json', output / 'partition_identity.json', output / 'frozen_plan.json'):
                training.mlflow.log_artifact(str(artifact), artifact_path='validation')
        rows.append(row)
        print(f'{name}: CV={spec["cv_f1"]:.6f} | val={metrics["val_f1"]:.6f} | unseen={metrics["unseen_val_f1"]:.6f}')
    # Preserve shortlist ordering: no implicit promotion or model-selection policy.
    pd.DataFrame(rows).to_csv(output / 'candidate_validation.csv', index=False)
    training.atomic_json_write(output / 'candidate_validation_summary.json', {**metadata, 'candidates': rows})
    print(f'Validation artifacts: {output}')
    return output


def main():
    """Parse arguments and run validation without accessing the held-out test set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True, help='Reviewed frozen candidate plan YAML.')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/config.yml')
    parser.add_argument('--processed-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    run_validation(parser.parse_args())


if __name__ == '__main__':
    main()
