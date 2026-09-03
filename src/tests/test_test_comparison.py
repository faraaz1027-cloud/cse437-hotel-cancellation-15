import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.test_comparison import (SELECTION_SHA256, assert_aligned,
                                 comparison_protocol, freeze_protocol)


class TestReportingComparison(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[2]
        self.selection = json.loads((self.root / 'data/processed/results/tuning/final_selection.json').read_text())

    def test_freezes_development_winner_and_late_timing(self):
        protocol = comparison_protocol(self.selection, SELECTION_SHA256)
        self.assertEqual(protocol['selected_model_unchanged'], 'logistic_regression')
        self.assertEqual(protocol['random_forest']['candidate'], 'rf_12')
        self.assertEqual(protocol['threshold'], 0.5)
        self.assertIn('after evaluation', protocol['timing_disclosure'])

    def test_rejects_selection_change(self):
        with self.assertRaises(ValueError):
            comparison_protocol(self.selection, 'changed')

    def test_rejects_forest_change(self):
        self.selection['best_by_family']['random_forest']['search_parameters']['model__min_samples_leaf'] = 1
        with self.assertRaises(ValueError):
            comparison_protocol(self.selection, SELECTION_SHA256)

    def test_protocol_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'protocol.json'
            freeze_protocol(path, {'threshold': 0.5})
            freeze_protocol(path, {'threshold': 0.5})
            with self.assertRaises(ValueError):
                freeze_protocol(path, {'threshold': 0.4})

    def test_alignment_checks_ids_and_labels(self):
        rows = pd.DataFrame({'cohort_row': [10, 20], 'source_row_id': [11, 21], 'actual': [0, 1]})
        assert_aligned(rows, rows, [0, 1])
        with self.assertRaises(ValueError):
            assert_aligned(rows, rows.iloc[::-1], [0, 1])
        with self.assertRaises(ValueError):
            assert_aligned(rows, rows, [1, 0])


if __name__ == '__main__':
    unittest.main()
