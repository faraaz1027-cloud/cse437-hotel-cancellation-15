import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout

from src.reproduction import (SELECTION, assert_frozen_unchanged, compare_tuning,
                              finish_development_run, frozen_manifest,
                              start_development_run, write_json)
from src.test_comparison import run_test_comparison
from src.tools.verify_notebooks import NOTEBOOKS, run_verification


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.parent = Path(self.temporary.name)
        self.root = self.parent / 'original'
        self.root.mkdir()
        for folder in ('data/processed', 'data/processed/splits', 'src', 'models',
                       'data/processed/results/tuning', 'data/processed/results/evaluation', 'data/processed/results/test_comparison'):
            (self.root / folder).mkdir(parents=True, exist_ok=True)
        (self.root / 'src/placeholder.py').write_text('# fixture\n')
        (self.root / 'data/processed/input.txt').write_text('untouched input')
        (self.root / 'models/frozen.joblib').write_bytes(b'untouched model')
        self.make_evidence(self.root)

    def make_evidence(self, root, winner='lr_06', changed=False):
        out = root / 'data/processed/results/tuning'
        out.mkdir(parents=True, exist_ok=True)
        configuration = {'candidate': winner, 'family': 'logistic_regression',
                         'representation': 'selected', 'threshold': .5,
                         'search_parameters': {'model__C': 1.0 if winner == 'lr_06' else .1,
                                               'model__class_weight': 'balanced'},
                         'estimator_parameters': {'C': 1.0 if winner == 'lr_06' else .1},
                         'mean_development_f1': .7321 if winner == 'lr_06' else .7317}
        write_json(out / 'final_selection.json', configuration)
        write_json(out / 'search_protocol.json', {'threshold': .5, 'grid': [.1, 1.]})
        write_json(out / 'candidate_parameters.json', {'lr_04': {'C': .1}, 'lr_06': {'C': 1.}})
        write_json(out / 'tuning_summary.json', {
            'input_sha256': {'input': 'same'}, 'source_code_sha256': {'src/tuning.py': 'same'},
            'runtime': {'python': '3.12'}})
        metrics = ['f1', 'accuracy', 'precision', 'recall', 'roc_auc']
        with (out / 'candidate_results.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['candidate'] + ['mean_' + k for k in metrics])
            writer.writeheader()
            for candidate, f1 in [('lr_04', .7317), ('lr_06', .7314 if changed else .7321)]:
                writer.writerow({'candidate': candidate, **{'mean_' + k: f1 for k in metrics}})
        with (out / 'fold_results.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['candidate', 'fold'] + metrics +
                                    ['tn', 'fp', 'fn', 'tp', 'training_f1'])
            writer.writeheader()
            for candidate in ('lr_04', 'lr_06'):
                writer.writerow({'candidate': candidate, 'fold': 1,
                                 **{k: .7 for k in metrics}, 'tn': 3, 'fp': 1,
                                 'fn': 1, 'tp': 3, 'training_f1': .8})

    def start(self):
        with patch('src.reproduction.environment_record', return_value={'test': True}):
            return start_development_run(self.root, parent=self.parent)

    def test_workspace_is_external_and_independent(self):
        before = frozen_manifest(self.root)
        run = self.start()
        self.assertNotIn(self.root, run.parents)
        self.assertFalse((run / 'models').exists())
        self.assertFalse((run / 'data/processed/results/evaluation').exists())
        self.assertFalse((run / SELECTION).exists())
        (run / 'data/processed/input.txt').write_text('new bytes')
        self.assertEqual((self.root / 'data/processed/input.txt').read_text(), 'untouched input')
        assert_frozen_unchanged(self.root, before)

    def test_rejects_workspace_inside_source(self):
        with self.assertRaises(ValueError):
            start_development_run(self.root, parent=self.root)

    def test_changed_winner_is_recorded_without_promoting(self):
        before = frozen_manifest(self.root)
        original = (self.root / SELECTION).read_bytes()
        run = self.start()
        self.make_evidence(run, winner='lr_04', changed=True)
        with patch('src.reproduction.environment_record', return_value={}):
            report = finish_development_run(self.root, run)
        self.assertEqual(report['status'], 'differences_detected')
        self.assertFalse(report['selection_matches'])
        self.assertTrue(report['candidate_metric_differences'])
        self.assertFalse(report['frozen_selection_promoted_from_rerun'])
        self.assertEqual((self.root / SELECTION).read_bytes(), original)
        assert_frozen_unchanged(self.root, before)

    def test_identical_evidence_has_narrow_match_status(self):
        run = self.start()
        self.make_evidence(run)
        report = compare_tuning(self.root, run)
        self.assertEqual(report['status'], 'matched_recorded_tuning_evidence')
        self.assertIn('not full-pipeline', report['scope'])

    def test_windows_newlines_and_path_keys_are_not_score_drift(self):
        run = self.start()
        self.make_evidence(run)
        protocol = run / 'data/processed/results/tuning/search_protocol.json'
        protocol.write_bytes(protocol.read_bytes().replace(b'\n', b'\r\n'))
        summary_path = run / 'data/processed/results/tuning/tuning_summary.json'
        summary = json.loads(summary_path.read_text())
        summary['source_code_sha256'] = {'src\\tuning.py': 'same'}
        write_json(summary_path, summary)
        report = compare_tuning(self.root, run)
        self.assertEqual(report['status'], 'matched_recorded_tuning_evidence')
        self.assertTrue(report['search_protocol_semantically_equal'])
        self.assertFalse(report['search_protocol_bytes_equal'])

    def test_frozen_mutation_fails(self):
        run = self.start()
        self.make_evidence(run)
        (self.root / 'models/frozen.joblib').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'frozen evidence changed'):
            finish_development_run(self.root, run)

    def test_cross_repository_run_rejected(self):
        run = self.start()
        with self.assertRaisesRegex(ValueError, 'another source'):
            finish_development_run(self.parent / 'another', run)

    def test_candidate_membership_change_rejected(self):
        run = self.start()
        self.make_evidence(run)
        table = run / 'data/processed/results/tuning/candidate_results.csv'
        table.write_text(table.read_text().replace('lr_04', 'lr_99'))
        with self.assertRaisesRegex(ValueError, 'membership changed'):
            compare_tuning(self.root, run)

    def test_missing_cache_never_trains(self):
        with patch('src.test_comparison.checked_selection') as selection_check:
            with self.assertRaisesRegex(FileNotFoundError, 'must not retrain'):
                run_test_comparison(self.root, require_cached=True)
            selection_check.assert_not_called()

    def test_scoring_code_and_hard_frozen_guard_remain(self):
        root = Path(__file__).resolve().parents[2]
        notebook = json.loads((root / 'notebooks/04_modeling_and_tuning.ipynb').read_text())
        source = '\n'.join(''.join(c['source']) if isinstance(c['source'], list) else c['source']
                           for c in notebook['cells'] if c['cell_type'] == 'code')
        self.assertIn('run_model_comparison(RUN_ROOT)', source)
        self.assertIn('run_tuning(RUN_ROOT)', source)
        self.assertNotIn('run_tuning(ROOT)', source)
        self.assertIn('finish_development_run(ROOT, RUN_ROOT)', source)
        final = (root / 'src/final_evaluation.py').read_text()
        self.assertIn("{'model__C': 1.0, 'model__class_weight': 'balanced'}", final)

    def runner_fixture(self):
        import nbformat
        folder = self.root / 'notebooks'
        folder.mkdir()
        for name in NOTEBOOKS:
            nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('pass')]),
                           folder / name)

    def test_runner_separates_execution_from_numerical_status(self):
        self.runner_fixture()
        # Stub kernels: this tests orchestration, not real notebook execution.
        comparison = {'status': 'differences_detected'}
        with patch('src.tools.verify_notebooks.environment_record', return_value={}), \
             patch('src.tools.verify_notebooks.collect_comparison', return_value=comparison), \
             patch('src.tools.verify_notebooks.subprocess.run') as process, \
             patch('src.tools.verify_notebooks.tempfile.mkdtemp',
                   return_value=str(self.parent / 'verification')):
            (self.parent / 'verification').mkdir()
            process.return_value.stdout = ''
            process.return_value.stderr = ''
            process.return_value.returncode = 0
            with redirect_stdout(io.StringIO()):
                result = run_verification(self.root, executor=lambda *args: None)
        self.assertEqual(result['notebook_execution_status'], 'passed')
        self.assertEqual(result['status'], 'passed_with_reproduction_differences')
        self.assertEqual(result['notebook_count'], 5)
        self.assertTrue(result['original_repository_unchanged'])
        self.assertFalse(result['submission_ready'])

    def test_runner_saves_failure_and_detects_deleted_frozen_file(self):
        self.runner_fixture()
        def fail(notebook, replica, manager):
            (replica / SELECTION).unlink()
            raise RuntimeError('simulated kernel failure')
        with patch('src.tools.verify_notebooks.environment_record', return_value={}), \
             patch('src.tools.verify_notebooks.subprocess.run') as process, \
             patch('src.tools.verify_notebooks.tempfile.mkdtemp',
                   return_value=str(self.parent / 'verification')):
            (self.parent / 'verification').mkdir()
            process.return_value.stdout = ''
            process.return_value.stderr = ''
            process.return_value.returncode = 0
            with redirect_stdout(io.StringIO()):
                result = run_verification(self.root, executor=fail)
        self.assertEqual(result['status'], 'failed_integrity_check')
        self.assertTrue(result['original_repository_unchanged'])
        self.assertFalse(result['frozen_evidence_unchanged'])
        self.assertTrue((self.parent / 'verification/traceback.txt').is_file())
        self.assertTrue((self.parent / 'verification/verification.json').is_file())


if __name__ == '__main__':
    unittest.main()
