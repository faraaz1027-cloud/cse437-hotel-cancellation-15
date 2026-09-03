"""User-approved verification reporting supplement; never selects a model from test."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import brier_score_loss
from threadpoolctl import threadpool_limits

from .development_eda import read_selected_rows
from .feature_audit import load_development
from .final_evaluation import checked_selection, sha256
from .modeling import cancellation_probability, classification_metrics, make_model_pipeline
from .splitting import check_assignments
from .tuning import build_frozen_pipeline

SELECTION_SHA256 = '68c4072f5c95e3a9f927a8b70a9e96aea8adbd4f7509f4d1c30ba6f7889f3b1b'
METRICS = ['f1', 'accuracy', 'precision', 'recall', 'roc_auc', 'tn', 'fp', 'fn', 'tp']


def comparison_protocol(selection, selection_hash):
    if selection_hash != SELECTION_SHA256:
        raise ValueError('The approved tuning selection changed.')
    forest = selection['best_by_family']['random_forest']
    if forest['candidate'] != 'rf_12' or forest['search_parameters'] != {
        'model__class_weight': 'balanced', 'model__max_depth': None,
        'model__min_samples_leaf': 10,
    }:
        raise ValueError('The development-selected Random Forest changed.')
    return {
        'analysis': 'test_comparison', 'scope': 'reporting-only late test comparison',
        'authorization': 'User approved adding the baseline and Random Forest test comparison.',
        'timing_disclosure': 'Authorized after evaluation Logistic Regression test results were known; not a preregistered simultaneous three-model test.',
        'selection_sha256': selection_hash,
        'selected_model_unchanged': 'logistic_regression',
        'threshold': 0.5, 'positive_class': 1,
        'baseline': {'family': 'majority', 'representation': 'none',
                     'strategy': 'most_frequent', 'random_state': 42},
        'random_forest': forest,
        'development_rows': 95415, 'test_rows': 23795,
        'test_policy': 'Fit only on frozen development rows. No tuning, reselection, threshold changes, or test-driven feature changes.',
        'logistic_policy': 'Reuse and verify saved evaluation predictions; do not refit or overwrite that model.',
        'undefined_precision_policy': 'Report zero when no cancellations are predicted.',
        'metrics': METRICS + ['brier_score'],
    }


def freeze_protocol(path, protocol):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError('Existing reporting protocol differs; refusing overwrite.')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(protocol, indent=2, allow_nan=False) + '\n')


def assert_aligned(predictions, assignments, target):
    for column in ['cohort_row', 'source_row_id']:
        if not np.array_equal(predictions[column].to_numpy(), assignments[column].to_numpy()):
            raise ValueError('evaluation test row order changed: ' + column)
    if not np.array_equal(predictions.actual.to_numpy(), np.asarray(target)):
        raise ValueError('evaluation labels differ from the frozen target.')


def run_test_comparison(root, *, require_cached=False):
    root = Path(root)
    out = root / 'data/processed/results/test_comparison'
    if require_cached:
        required = ('comparison_protocol.json', 'comparison_summary.json',
                    'test_comparison.csv', 'random_forest_test_probabilities.csv.gz')
        missing = [name for name in required if not (out / name).is_file()]
        if missing:
            raise FileNotFoundError('Frozen comparison evidence is missing; verification '
                                    'must not retrain: ' + ', '.join(missing))
    selection, _, selection_hash = checked_selection(root)
    protocol = comparison_protocol(selection, selection_hash)
    evaluation_path = root / 'data/processed/results/evaluation/evaluation_summary.json'
    evaluation = json.loads(evaluation_path.read_text())
    protected = {**evaluation['output_sha256'],
                 'data/processed/results/evaluation/evaluation_summary.json': sha256(evaluation_path),
                 'data/processed/results/tuning/final_selection.json': selection_hash}
    for relative, expected in protected.items():
        if sha256(root / relative) != expected:
            raise ValueError('Frozen artifact changed: ' + relative)
    freeze_protocol(out / 'comparison_protocol.json', protocol)
    summary_path = out / 'comparison_summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        for relative, expected in summary['output_sha256'].items():
            if sha256(root / relative) != expected:
                raise ValueError('Saved supplement changed: ' + relative)
        return pd.read_csv(out / 'test_comparison.csv'), summary

    X_dev, assignments, input_hashes = load_development(root)
    check_assignments(assignments)
    dev_mask = assignments.partition.eq('development').to_numpy()
    test_mask = assignments.partition.eq('test').to_numpy()
    if (int(dev_mask.sum()), int(test_mask.sum())) != (95415, 23795):
        raise ValueError('Frozen partition sizes changed.')
    target_path = root / 'data/processed/eligibility_target.csv.gz'
    if sha256(target_path) != selection['input_sha256'][target_path.name]:
        raise ValueError('Frozen target changed.')
    y_dev = read_selected_rows(target_path, dev_mask).is_canceled.reset_index(drop=True)
    forest_selection = {**protocol['random_forest'], 'analysis': 'tuning', 'threshold': 0.5}
    forest = build_frozen_pipeline(forest_selection)
    baseline = make_model_pipeline('majority', 'none')
    # Both estimators and every learned transformation see development data only.
    with threadpool_limits(limits=1):
        baseline.fit(X_dev, y_dev)
        forest.fit(X_dev, y_dev)
    forest_state = joblib.hash(forest)
    X_test = read_selected_rows(root / 'data/processed/eligibility_candidates.csv.gz', test_mask).reset_index(drop=True)
    y_test = read_selected_rows(target_path, test_mask).is_canceled.reset_index(drop=True)
    test_assignments = assignments.loc[test_mask].reset_index(drop=True)
    saved = pd.concat([pd.read_csv(root / f'data/processed/results/evaluation/test_predictions_{part:02d}.csv.gz')
                       for part in range(1, 5)], ignore_index=True)
    assert_aligned(saved, test_assignments, y_test)
    probabilities = {
        'majority_baseline': cancellation_probability(baseline, X_test),
        'logistic_regression': saved.cancellation_probability.to_numpy(),
        'random_forest': cancellation_probability(forest, X_test),
    }
    if joblib.hash(forest) != forest_state:
        raise AssertionError('Test prediction changed the fitted Random Forest.')
    rows = []
    for family, probability in probabilities.items():
        metrics = classification_metrics(y_test, probability, 0.5)
        if family == 'logistic_regression':
            for name in METRICS:
                if not np.isclose(metrics[name], evaluation['metrics'][name], rtol=0, atol=1e-12):
                    raise AssertionError('Saved Logistic Regression metric changed: ' + name)
        rows.append({'model': family, 'selected_before_test': family == 'logistic_regression',
                     'test_rows': len(y_test), 'threshold': 0.5, **metrics,
                     'brier_score': float(brier_score_loss(y_test, probability))})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out / 'test_comparison.csv', index=False)
    probability_export = pd.DataFrame({
        'source_row_id': test_assignments.source_row_id,
        'actual': y_test,
        'random_forest_probability': probabilities['random_forest'],
    })
    probability_export.to_csv(out / 'random_forest_test_probabilities.csv.gz', index=False,
                              compression={'method': 'gzip', 'mtime': 0})
    for relative, expected in protected.items():
        if sha256(root / relative) != expected:
            raise AssertionError('Protected artifact was modified: ' + relative)
    summary = {
        'analysis': 'test_comparison', 'status': 'comparison supplement complete; overall verification remains open',
        'development_rows_fitted': len(X_dev), 'test_rows_evaluated': len(y_test),
        'selected_model': 'logistic_regression', 'model_reselected_from_test': False,
        'threshold_changed': False, 'frozen_evaluation_artifacts_unchanged': True,
        'forest_unchanged_after_prediction': True,
        'test_timing': protocol['timing_disclosure'],
        'baseline_rule': 'Development majority is class 0; every test cancellation probability is 0.',
        'input_sha256': {**input_hashes, target_path.name: sha256(target_path)},
        'protected_sha256': protected,
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                    'pandas': pd.__version__, 'scikit_learn': sklearn.__version__, 'joblib': joblib.__version__},
        'output_sha256': {str(p.relative_to(root)): sha256(p) for p in
                          [out / 'comparison_protocol.json', out / 'test_comparison.csv',
                           out / 'random_forest_test_probabilities.csv.gz']},
    }
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return comparison, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    table, _ = run_test_comparison(parser.parse_args().root)
    print(table.to_string(index=False))
