"""tuning exhaustive, development-only hyperparameter search; no global refit."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, ParameterGrid
from threadpoolctl import threadpool_limits

from .development_eda import read_selected_rows
from .feature_audit import load_development
from .modeling import make_model_pipeline, cancellation_probability, classification_metrics
from .splitting import development_cv

GRIDS = {
    'logistic_regression': {'model__C': [.01, .1, 1.0, 10.0],
                            'model__class_weight': [None, 'balanced']},
    'random_forest': {'model__max_depth': [8, 16, None],
                      'model__min_samples_leaf': [1, 10],
                      'model__class_weight': [None, 'balanced']},
}
METRICS = ['f1', 'accuracy', 'precision', 'recall', 'roc_auc']
COUNTS = ['tn', 'fp', 'fn', 'tp']
PROTOCOL = {
    'analysis': 'tuning', 'method': 'exhaustive GridSearchCV', 'parameter_grids': GRIDS,
    'representations': {'logistic_regression': 'selected', 'random_forest': 'selected'},
    'selection_percentile': 75, 'pca_retained': False,
    'fixed_settings_source': 'src/modeling.py MODEL_SETTINGS; only listed model parameters vary',
    'candidate_counts': {'logistic_regression': 8, 'random_forest': 12},
    'folds': 3, 'expected_fits': 60,
    'cv': 'immutable validation expanding forward folds; development-relative indices',
    'seed': 42, 'search_n_jobs': 1, 'forest_n_jobs': 2,
    'primary_metric': 'unweighted three-fold mean cancellation-class F1',
    'secondary_metrics': METRICS[1:],
    'threshold': .5, 'threshold_rule': 'class 1 when cancellation probability >= 0.5',
    'decision_rule': 'highest mean F1; exact ties use family order then ParameterGrid order',
    'family_order': list(GRIDS), 'refit': False, 'error_score': 'raise',
    'return_train_score': True, 'resampling': None, 'threshold_search': False,
    'class_weighting': 'None or balanced; balanced weights computed from each training fold',
    'budget_rationale': 'Complete 20-setting grid targets linear regularization and forest capacity/imbalance; tree count remains 100 and max_features remains sqrt.',
    'final_test_policy': 'No final test fitting, transformation, scoring or label summaries; reserve for evaluation.',
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_protocol(path):
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != PROTOCOL:
        raise ValueError('Frozen tuning protocol changed; version and justify explicitly.')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(PROTOCOL, indent=2) + '\n')


def score_pipeline(estimator, X, y):
    """Use positive-class probabilities, including >=.5 ties, for every score."""
    return classification_metrics(y, cancellation_probability(estimator, X), .5)


def run_search(family, X, y, cv, param_grid=None, verbose=0):
    if family not in GRIDS:
        raise ValueError('Unknown tuning family.')
    grid = GRIDS[family] if param_grid is None else param_grid
    search = GridSearchCV(make_model_pipeline(family, 'selected'), grid,
                          scoring=score_pipeline, cv=cv, refit=False,
                          n_jobs=1, error_score='raise', return_train_score=True,
                          verbose=verbose)
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        search.fit(X, y)
    if hasattr(search, 'best_estimator_'):
        raise AssertionError('tuning must not refit a model on all development rows.')
    return search


def rank_candidates(table):
    if table.empty or not np.isfinite(table.mean_f1).all():
        raise ValueError('Candidate mean F1 must be nonempty and finite.')
    return table.sort_values(['mean_f1', 'candidate_order'],
                             ascending=[False, True], kind='stable').reset_index(drop=True)


def check_untuned_control(old, current, family):
    """Hard metrics must match tightly; permit tiny secondary forest AUC drift.

    The first strict audit found forest AUC differences up to 1.14e-8 with
    identical threshold metrics. Floating-point accumulation near tied scores
    can affect AUC ranking. This numerical check does not alter model scoring,
    the grid, threshold, primary metric or selection rule.
    """
    checks = []
    for metric in METRICS:
        tolerance = 1e-7 if family == 'random_forest' and metric == 'roc_auc' else 1e-12
        np.testing.assert_allclose(old[metric], current[metric], atol=tolerance, rtol=0)
        for fold, before, after in zip([1, 2, 3], old[metric], current[metric]):
            checks.append({'family': family, 'fold': fold, 'metric': metric,
                           'model_comparison_value': float(before), 'tuning_value': float(after),
                           'absolute_difference': float(abs(before - after)),
                           'absolute_tolerance': tolerance})
    return checks


def build_frozen_pipeline(selection):
    """Construct, but do not fit, the development-selected pipeline for evaluation."""
    if selection.get('analysis') != 'tuning' or selection.get('threshold') != .5:
        raise ValueError('Unknown selection version or changed threshold.')
    family = selection['family']
    if family not in GRIDS or selection.get('representation') != 'selected':
        raise ValueError('Unknown family or representation.')
    params = selection['search_parameters']
    if params not in list(ParameterGrid(GRIDS[family])):
        raise ValueError('Selected parameters were not in the frozen search space.')
    pipe = make_model_pipeline(family, 'selected').set_params(**params)
    if pipe.named_steps['model'].get_params(deep=False) != selection['estimator_parameters']:
        raise ValueError('Estimator defaults/settings changed since selection.')
    return pipe


def plot_tuning(root, comparison):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['Logistic regression', 'Random forest']
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for shift, column, label, color in [(-.18, 'untuned_mean_f1', 'model comparison untuned', '#9aa8b2'),
                                      (.18, 'tuned_mean_f1', 'tuning best grid setting', '#267f9c')]:
        bars = ax.bar(x + shift, comparison[column], .34, label=label, color=color)
        ax.bar_label(bars, fmt='%.3f', padding=4)
    ax.set_xticks(x, labels); ax.set_ylim(0, 1)
    ax.set_ylabel('Mean cancellation-class F1')
    ax.set_title('Tuning on the same three forward development folds', loc='left')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='lower left', frameon=False)
    fig.text(.13, .025, 'Selected features • Threshold 0.5 • No final test evaluation\n'
             'Best-grid scores reuse development data; they are not unbiased final estimates.', fontsize=9)
    fig.subplots_adjust(left=.13, right=.98, top=.87, bottom=.23)
    path = Path(root) / 'figures/08_hyperparameter_tuning.png'
    fig.savefig(path, dpi=150, facecolor='white'); plt.close(fig)
    return path


def run_tuning(root):
    root = Path(root); output = root / 'data/processed/results/tuning'
    write_protocol(output / 'search_protocol.json')  # freeze before inspecting new scores
    previous = json.loads((root / 'data/processed/results/model_comparison/model_summary.json').read_text())
    for path, expected in previous['output_sha256'].items():
        if sha256(root / path) != expected:
            raise ValueError('model comparison evidence changed: ' + path)
    X, assignments, hashes = load_development(root)
    target_path = root / 'data/processed/eligibility_target.csv.gz'
    hashes[target_path.name] = sha256(target_path)
    if hashes[target_path.name] != previous['input_sha256'][target_path.name]:
        raise ValueError('Frozen target changed.')
    if hashes != previous['input_sha256']:
        raise ValueError('Frozen feature/split lineage changed.')
    y = read_selected_rows(target_path, assignments.partition.eq('development')).is_canceled
    if len(X) != len(y) or set(y.unique()) != {0, 1}:
        raise ValueError('Development targets do not align.')
    cv = list(development_cv(assignments))
    dev = assignments.loc[assignments.partition.eq('development')].reset_index(drop=True)
    fold_specs = []
    for fold, (tr, va) in enumerate(cv, 1):
        if np.intersect1d(tr, va).size:
            raise ValueError('Training/validation overlap.')
        fold_specs.append({'fold': fold, 'train_rows': len(tr), 'validation_rows': len(va),
                           'training_canceled': int(y.iloc[tr].sum()),
                           'validation_canceled': int(y.iloc[va].sum()),
                           'validation_membership_sha256': hashlib.sha256(
                               dev.iloc[va].source_row_id.to_csv(index=False).encode()).hexdigest()})
    rows, candidates, all_params = [], [], {}
    order = 0
    for family in GRIDS:
        print(f'\ntuning: {family}, {len(list(ParameterGrid(GRIDS[family])))} settings × 3 frozen folds', flush=True)
        search = run_search(family, X, y, cv, verbose=2)
        raw = pd.DataFrame(search.cv_results_)
        raw['params'] = raw.params.map(lambda p: json.dumps(p, sort_keys=True))
        raw.to_csv(output / f'{family}_cv_results.csv', index=False)
        for i, params in enumerate(search.cv_results_['params']):
            candidate = ('lr' if family == 'logistic_regression' else 'rf') + f'_{i+1:02d}'
            all_params[candidate] = {'family': family, 'representation': 'selected',
                                     'search_parameters': params,
                                     'estimator_parameters': make_model_pipeline(family, 'selected').set_params(
                                         **params).named_steps['model'].get_params(deep=False)}
            item = {'candidate': candidate, 'candidate_order': order, 'family': family,
                    'representation': 'selected', 'parameters_json': json.dumps(params, sort_keys=True),
                    'mean_fit_seconds': float(search.cv_results_['mean_fit_time'][i])}
            for metric in METRICS:
                item['mean_' + metric] = float(search.cv_results_['mean_test_' + metric][i])
            item['mean_training_f1'] = float(search.cv_results_['mean_train_f1'][i])
            item['mean_f1_gap'] = item['mean_training_f1'] - item['mean_f1']
            item['fold_sd_f1'] = float(np.std([search.cv_results_[f'split{k}_test_f1'][i] for k in range(3)], ddof=1))
            candidates.append(item)
            for k, spec in enumerate(fold_specs):
                row = {'candidate': candidate, 'family': family, **spec}
                for metric in METRICS + COUNTS:
                    value = search.cv_results_[f'split{k}_test_{metric}'][i]
                    row[metric] = int(value) if metric in COUNTS else float(value)
                row['training_f1'] = float(search.cv_results_[f'split{k}_train_f1'][i])
                if sum(row[c] for c in COUNTS) != spec['validation_rows']:
                    raise AssertionError('Confusion counts do not reconcile.')
                rows.append(row)
            order += 1
        print(f'Completed {family} grid; no global refit.', flush=True)
    scores = pd.DataFrame(candidates); folds = pd.DataFrame(rows)
    if len(scores) != 20 or len(folds) != 60:
        raise AssertionError('Search did not complete all declared fits.')
    old_folds = pd.read_csv(root / 'data/processed/results/model_comparison/fold_results.csv')
    old_means = pd.read_csv(root / 'data/processed/results/model_comparison/model_comparison.csv').set_index('candidate')
    comparisons = []; best_by_family = {}; parity_checks = []
    for family, old_key, baseline_params in [
        ('logistic_regression', 'lr_selected', {'model__C': 1.0, 'model__class_weight': None}),
        ('random_forest', 'rf_selected', {'model__max_depth': None, 'model__min_samples_leaf': 1, 'model__class_weight': None})]:
        same = [key for key, config in all_params.items() if config['family'] == family and config['search_parameters'] == baseline_params]
        if len(same) != 1:
            raise AssertionError('Exactly one untuned control must occur in each grid.')
        old = old_folds.loc[old_folds.candidate.eq(old_key)].sort_values('fold')
        current = folds.loc[folds.candidate.eq(same[0])].sort_values('fold')
        parity_checks.extend(check_untuned_control(old, current, family))
        if old.validation_membership_sha256.tolist() != current.validation_membership_sha256.tolist():
            raise AssertionError('Validation membership changed.')
        best = rank_candidates(scores.loc[scores.family.eq(family)]).iloc[0]
        best_by_family[family] = {'candidate': str(best.candidate), **all_params[best.candidate],
                                  'mean_f1': float(best.mean_f1)}
        comparisons.append({'family': family, 'untuned_candidate': old_key,
                            'tuned_candidate': best.candidate,
                            'untuned_mean_f1': float(old_means.loc[old_key, 'mean_f1']),
                            'tuned_mean_f1': float(best.mean_f1),
                            'f1_change': float(best.mean_f1 - old_means.loc[old_key, 'mean_f1'])})
    comparison = pd.DataFrame(comparisons)
    winner = rank_candidates(scores).iloc[0]
    selection = {'analysis': 'tuning', 'candidate': str(winner.candidate), **all_params[winner.candidate],
                 'threshold': .5, 'threshold_rule': PROTOCOL['threshold_rule'],
                 'mean_development_f1': float(winner.mean_f1),
                 'best_by_family': best_by_family,
                 'input_sha256': hashes, 'search_protocol_sha256': sha256(output / 'search_protocol.json'),
                 'scope': 'frozen development-selected settings; no fitted estimator or final test result',
                 'evaluation_policy': 'Refit the selected pipeline on development only, then evaluate held-out data once; never reselect from test scores.'}
    frozen = build_frozen_pipeline(selection)
    if hasattr(frozen.named_steps['representation'], 'preprocessor_'):
        raise AssertionError('The selected pipeline must be unfitted.')
    scores.to_csv(output / 'candidate_results.csv', index=False)
    folds.to_csv(output / 'fold_results.csv', index=False)
    comparison.to_csv(output / 'tuning_comparison.csv', index=False)
    pd.DataFrame(parity_checks).to_csv(output / 'control_parity.csv', index=False)
    (output / 'candidate_parameters.json').write_text(json.dumps(all_params, indent=2) + '\n')
    (output / 'final_selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    figure = plot_tuning(root, comparison)
    summary = {'analysis': 'tuning', 'status': 'completed', 'development_rows': len(X),
               'candidates': len(scores), 'model_fits': len(folds), 'folds': 3,
               'preferred_candidate': str(winner.candidate), 'preferred_family': str(winner.family),
               'mean_f1': float(winner.mean_f1), 'untuned_controls_match_model_comparison': True,
               'control_parity_policy': {'threshold_metrics_and_logistic_auc_atol': 1e-12,
                                        'forest_auc_atol': 1e-7,
                                        'note': 'First strict audit detected forest AUC drift up to 1.14e-8; only the numerical check was relaxed, not scoring or selection.'},
               'convergence_warnings': 0, 'failed_fits': 0,
               'test_rows_fitted_transformed_or_scored': 0, 'test_target_distribution_computed': False,
               'full_development_model_fitted': False, 'threshold_tuned': False,
               'input_sha256': hashes, 'source_code_sha256': {
                   str(p.relative_to(root)): sha256(p) for p in [root / 'src/tuning.py', root / 'src/modeling.py', root / 'src/representation.py']},
               'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                           'pandas': pd.__version__, 'scikit_learn': sklearn.__version__},
               'limitations': ['Reused development folds create selection optimism; no nested CV or final test estimate.',
                              'The bounded grid is not proof of global optimality; winning boundary values need not be expanded.',
                              'Feature selection and threshold stay fixed; class weighting changes the precision/recall tradeoff.',
                              'Temporal shift, repeated profiles and retrospective feature-timing limitations remain.']}
    files = [p for p in output.iterdir() if p.is_file() and p.name not in ['tuning_summary.json', 'README.md']] + [figure]
    summary['output_sha256'] = {str(p.relative_to(root)): sha256(p) for p in sorted(files)}
    (output / 'tuning_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return summary, scores, folds, comparison, selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    summary, scores, _, comparison, selection = run_tuning(parser.parse_args().root)
    print(rank_candidates(scores).to_string(index=False))
    print(comparison.to_string(index=False))
    print(json.dumps({'preferred_family': selection['family'], 'parameters': selection['search_parameters'],
                      'mean_f1': summary['mean_f1']}, indent=2))


if __name__ == '__main__':
    main()
