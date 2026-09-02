"""Step 12 search budget, threshold policy, fold isolation and frozen handoff."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from src.modeling import make_model_pipeline
from src.representation import BookingRepresentation
from src.tuning import GRIDS, PROTOCOL, build_frozen_pipeline, check_untuned_control, rank_candidates, run_search, score_pipeline, write_protocol
from test_feature_engineering import fixture


class TuningTests(unittest.TestCase):
    def test_frozen_grid_counts_and_untuned_controls(self):
        self.assertEqual([len(list(ParameterGrid(g))) for g in GRIDS.values()], [8, 12])
        self.assertEqual(PROTOCOL['expected_fits'], 60)
        self.assertIn({'model__C': 1.0, 'model__class_weight': None}, list(ParameterGrid(GRIDS['logistic_regression'])))
        self.assertIn({'model__max_depth': None, 'model__min_samples_leaf': 1,
                       'model__class_weight': None}, list(ParameterGrid(GRIDS['random_forest'])))

    def test_multi_metric_scorer_uses_probability_ties_not_native_predict(self):
        class Estimator:
            classes_ = np.array([0, 1])
            def predict(self, X):
                raise AssertionError('Native predict may use a different tie rule.')
            def predict_proba(self, X):
                return np.array([[.9, .1], [.5, .5], [.6, .4], [.1, .9]])
        scores = score_pipeline(Estimator(), [1, 2, 3, 4], [0, 0, 1, 1])
        self.assertEqual(scores['f1'], .5)
        self.assertEqual(scores['roc_auc'], .75)
        self.assertEqual(sum(scores[c] for c in ['tn', 'fp', 'fn', 'tp']), 4)

    def test_search_clones_fold_training_and_never_refits_global_data(self):
        X = pd.concat([fixture()] * 3, ignore_index=True)
        X['adr'] = np.arange(12) + 20
        X['country'] = ['TRAIN'] * 4 + ['VAL1'] * 4 + ['VAL2'] * 4
        y = np.tile([0, 0, 1, 1], 3)
        cv = [(np.arange(4), np.arange(4, 8)), (np.arange(8), np.arange(8, 12))]
        records = []
        original_fit = BookingRepresentation.fit
        def traced_fit(obj, train, target=None):
            records.append((train.index.tolist(), np.asarray(target).tolist()))
            return original_fit(obj, train, target)
        with patch.object(BookingRepresentation, 'fit', traced_fit):
            search = run_search('logistic_regression', X, y, cv,
                                {'model__C': [.1, 1.0], 'model__class_weight': [None]})
        self.assertEqual(len(records), 4)
        self.assertEqual([len(r[0]) for r in records], [4, 8, 4, 8])
        for indices, target in records:
            self.assertEqual(target, y[indices].tolist())
            self.assertNotIn(8, indices)
        self.assertFalse(hasattr(search, 'best_estimator_'))
        self.assertEqual(len(search.cv_results_['params']), 2)
        self.assertTrue(np.isfinite(search.cv_results_['mean_test_f1']).all())

    def test_protocol_rejects_silent_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'protocol.json'
            write_protocol(path); write_protocol(path)
            modified = json.loads(path.read_text()); modified['threshold'] = .6
            path.write_text(json.dumps(modified))
            with self.assertRaises(ValueError): write_protocol(path)

    def test_candidate_ranking_is_deterministic_and_rejects_nan(self):
        table = pd.DataFrame({'candidate': ['later', 'first', 'worse'],
                              'candidate_order': [2, 0, 1], 'mean_f1': [.7, .7, .6]})
        self.assertEqual(rank_candidates(table).candidate.tolist(), ['first', 'later', 'worse'])
        table.loc[0, 'mean_f1'] = np.nan
        with self.assertRaises(ValueError): rank_candidates(table)

    def test_frozen_selection_builds_unfitted_pipeline_and_rejects_drift(self):
        params = {'model__C': .1, 'model__class_weight': 'balanced'}
        expected = make_model_pipeline('logistic_regression', 'selected').set_params(**params)
        selection = {'step': 12, 'family': 'logistic_regression', 'representation': 'selected',
                     'threshold': .5, 'search_parameters': params,
                     'estimator_parameters': expected.named_steps['model'].get_params(deep=False)}
        result = build_frozen_pipeline(selection)
        self.assertFalse(hasattr(result.named_steps['representation'], 'preprocessor_'))
        for key, value in [('threshold', .6), ('representation', 'full'),
                           ('search_parameters', {'model__C': 99}), ('estimator_parameters', {})]:
            bad = copy.deepcopy(selection); bad[key] = value
            with self.assertRaises(ValueError): build_frozen_pipeline(bad)

    def test_control_parity_only_allows_tiny_secondary_forest_auc_variation(self):
        old = pd.DataFrame({m: [.7, .72, .74] for m in ['f1','accuracy','precision','recall','roc_auc']})
        current = old.copy(); current['roc_auc'] += 2e-8
        self.assertEqual(len(check_untuned_control(old,current,'random_forest')),15)
        with self.assertRaises(AssertionError): check_untuned_control(old,current,'logistic_regression')
        current['f1'] += 1e-5
        with self.assertRaises(AssertionError): check_untuned_control(old,current,'random_forest')
        current = old.copy(); current['roc_auc'] += .001
        with self.assertRaises(AssertionError): check_untuned_control(old,current,'random_forest')


if __name__ == '__main__': unittest.main()
