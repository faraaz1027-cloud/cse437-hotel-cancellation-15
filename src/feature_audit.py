"""Verify feature engineering in frozen development folds; export aggregate evidence only."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from .development_eda import read_selected_rows
from .feature_engineering import (BookingFeatureEngineer, make_feature_preprocessor,
    DERIVED_COLUMNS, FEATURE_COLUMNS, FEATURE_LOG_COLUMNS, FEATURE_NUMERIC_COLUMNS,
    FEATURE_CATEGORICAL_COLUMNS, FEATURE_MISSING_COLUMNS, MONTH_NUMBERS)
from .preprocessing_audit import checked_matrix, fitted_state
from .splitting import check_assignments, development_cv


def load_development(root):
    """Verify lineage and row alignment, then load only selected feature rows.

    Target values are not read. Parsing source chunks is solely for filtering;
    test features are not transformed, summarized, fitted, or exported.
    """
    root = Path(root)
    processed, splits = root/'data/processed', root/'data/processed/splits'
    plan_path = splits/'validation_split_plan.json'
    plan = json.loads(plan_path.read_text())
    paths = {name: processed/name for name in ['eligibility_candidates.csv.gz', 'eligibility_metadata.csv.gz']}
    paths['validation_assignments.csv.gz'] = splits/plan['assignment_file']
    expected = {**plan['upstream_output_sha256'], 'validation_assignments.csv.gz': plan['assignment_sha256']}
    hashes = {}
    for name, path in paths.items():
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if hashes[name] != expected[name]:
            raise ValueError(f'Frozen input changed: {name}')
    hashes['validation_split_plan.json'] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    assignments = pd.read_csv(paths['validation_assignments.csv.gz'])
    check_assignments(assignments)
    mask = assignments.partition.eq('development').to_numpy()
    X = read_selected_rows(paths['eligibility_candidates.csv.gz'], mask)
    meta = read_selected_rows(paths['eligibility_metadata.csv.gz'], mask)
    dev = assignments.loc[mask].reset_index(drop=True)
    if len(X) != 95415 or meta.source_row_id.tolist() != dev.source_row_id.tolist():
        raise ValueError('Development cohort or row alignment changed.')
    return X, assignments, hashes


def run_feature_audit(root):
    root = Path(root)
    X, assignments, hashes = load_development(root)
    # This full-development view uses fixed formulas only, never learned fitting.
    derived = BookingFeatureEngineer().fit_transform(X)
    stats = derived[list(DERIVED_COLUMNS)].describe().T.rename_axis('feature').reset_index()
    stats['missing_before_imputation'] = derived[list(DERIVED_COLUMNS)].isna().sum().to_numpy()
    folds, schemas = [], {}
    for number, (train_idx, val_idx) in enumerate(development_cv(assignments), 1):
        train, val = X.iloc[train_idx], X.iloc[val_idx]
        pipe = make_feature_preprocessor()
        encoded_train = pipe.fit_transform(train)
        before = fitted_state(pipe)
        encoded_val = pipe.transform(val)
        assert before == fitted_state(pipe), 'Validation changed learned state'
        train_info, val_info = checked_matrix(encoded_train, len(train)), checked_matrix(encoded_val, len(val))
        assert train_info['columns'] == val_info['columns']
        clean_train = pipe.named_steps['features'].transform(train)
        clean_val = pipe.named_steps['features'].transform(val)
        medians = {}
        for branch_name, columns in [('log_numeric', FEATURE_LOG_COLUMNS), ('numeric', FEATURE_NUMERIC_COLUMNS)]:
            branch = pipe.named_steps['columns'].named_transformers_[branch_name]
            actual = branch.named_steps['impute'].statistics_
            np.testing.assert_allclose(actual, clean_train[list(columns)].median().fillna(0))
            imputed = branch.named_steps['impute'].transform(clean_train[list(columns)])
            scaled_input = np.log1p(imputed) if branch_name == 'log_numeric' else imputed
            np.testing.assert_allclose(branch.named_steps['scale'].mean_, scaled_input.mean(axis=0), atol=1e-10)
            assert int(branch.named_steps['scale'].n_samples_seen_) == len(train)
            medians.update(dict(zip(columns, map(float, actual))))
        encoder = pipe.named_steps['columns'].named_transformers_['categorical']
        for c, vocabulary in zip(FEATURE_CATEGORICAL_COLUMNS, encoder.categories_):
            assert set(vocabulary) == set(clean_train[c]), 'Vocabulary was not fitted on training only'
        tree = make_feature_preprocessor(scale_numeric=False)
        tree_train = tree.fit_transform(train)
        tree_before = fitted_state(tree)
        tree_val = tree.transform(val)
        assert tree_before == fitted_state(tree)
        checked_matrix(tree_train, len(train)); checked_matrix(tree_val, len(val))
        names = pipe.get_feature_names_out().tolist()
        assert len(names) == len(set(names)) == encoded_train.shape[1]
        assert names == tree.get_feature_names_out().tolist()
        assert tree_train.shape == encoded_train.shape and tree_val.shape == encoded_val.shape
        unseen = sorted(set(val.arrival_date_month) - set(train.arrival_date_month))
        unseen_rows = val.arrival_date_month.isin(unseen)
        cyclic = clean_val.loc[unseen_rows, ['arrival_month_sin', 'arrival_month_cos']].to_numpy()
        assert np.isfinite(cyclic).all()
        np.testing.assert_allclose((cyclic**2).sum(axis=1), 1, atol=1e-12)
        schemas[f'fold_{number}'] = names
        folds.append({'fold': number, 'training': train_info, 'validation': val_info,
            'training_medians_derived': {k: v for k, v in medians.items() if k in DERIVED_COLUMNS},
            'validation_missing_derived_before_imputation': {c: int(clean_val[c].isna().sum()) for c in DERIVED_COLUMNS},
            'validation_months_absent_from_training': unseen,
            'validation_rows_with_unseen_month': int(unseen_rows.sum()),
            'unseen_months_have_valid_cyclic_coordinates': True,
            'training_only_statistics_verified': True, 'validation_did_not_change_fitted_state': True,
            'scaled_and_unscaled_variants_verified': True})
    summary = {'analysis': 'features', 'status': 'completed', 'development_rows': len(X), 'input_candidate_fields': X.shape[1],
        'retained_source_fields': 24, 'derived_fields': list(DERIVED_COLUMNS),
        'fields_before_encoding': len(FEATURE_COLUMNS), 'fixed_missing_indicators': list(FEATURE_MISSING_COLUMNS),
        'month_names_replaced_by_cyclic_pair': True,
        'development_zero_night_bookings_retained': int(derived.total_nights.eq(0).sum()),
        'development_unknown_total_guests': int(derived.total_guests.isna().sum()),
        'development_no_recorded_history': int(derived.has_booking_history.eq(0).sum()),
        'development_company_code_recorded': int(derived.company_code_recorded.eq(1).sum()),
        'test_rows_fitted_or_transformed': 0, 'target_values_read': False, 'predictive_models_trained': 0,
        'rows_removed': 0, 'folds': folds, 'input_sha256': hashes,
        'runtime': {'python': platform.python_version(), 'pandas': pd.__version__, 'numpy': np.__version__,
                    'scikit_learn': sklearn.__version__},
        'limitations': ['Derived features are candidates, not a validated final feature set.',
            'Totals are formed before imputation; separately imputed totals need not equal sums of imputed components.',
            'Company-code presence is a recording indicator, not proof of corporate payment or booking-time availability.',
            'Cyclic month coordinates encode proximity but do not establish observed seasonal coverage or solve temporal drift.',
            'Source history and other feature timing remain retrospective; no live prediction claim is made.',
            'Statistical feature selection, dimensionality reduction, and performance comparisons remain for additional analyses.']}
    output = root/'data/processed/features'; output.mkdir(parents=True, exist_ok=True)
    stats.to_csv(output/'derived_feature_statistics.csv', index=False)
    (output/'feature_schemas.json').write_text(json.dumps(schemas, indent=2)+'\n')
    summary['output_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [output/'derived_feature_statistics.csv', output/'feature_schemas.json']}
    (output/'feature_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    return summary, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    summary, _ = run_feature_audit(args.root)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
