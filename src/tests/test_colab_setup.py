import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.tools import colab_setup as setup


class ColabSetupTests(unittest.TestCase):
    def test_local_execution_does_not_install_or_clone(self):
        with patch.object(setup, 'in_colab', return_value=False), \
             patch.object(setup.subprocess, 'run') as run:
            self.assertIsNone(setup.prepare_colab())
            run.assert_not_called()

    def test_pins_match_project_requirements(self):
        root = Path(__file__).resolve().parents[2]
        requirements = (root / 'requirements.txt').read_text().splitlines()
        for name, (_, version) in setup.ANALYSIS_PACKAGES.items():
            self.assertIn(f'{name}=={version}', requirements)
        self.assertNotIn('ipykernel', setup.ANALYSIS_PACKAGES)

    def test_matching_packages_skip_install(self):
        with patch.object(setup, 'ANALYSIS_PACKAGES', {'example': ('example_not_loaded', '1.0')}), \
             patch.object(setup, 'installed_version', return_value='1.0'), \
             patch.object(setup.subprocess, 'run') as run:
            setup.ensure_analysis_packages()
            run.assert_not_called()

    def test_install_failure_stops_setup(self):
        with patch.object(setup, 'ANALYSIS_PACKAGES', {'example': ('example_not_loaded', '1.0')}), \
             patch.object(setup, 'installed_version', return_value=None), \
             patch.object(setup.subprocess, 'run', side_effect=RuntimeError('pip failed')):
            with self.assertRaisesRegex(RuntimeError, 'pip failed'):
                setup.ensure_analysis_packages()

    def test_install_missing_package_without_restart_if_not_loaded(self):
        with patch.object(setup, 'ANALYSIS_PACKAGES', {'example': ('example_not_loaded', '1.0')}), \
             patch.object(setup, 'installed_version', side_effect=[None, '1.0']), \
             patch.object(setup.subprocess, 'run') as run:
            setup.ensure_analysis_packages()
            self.assertIn('example==1.0', run.call_args.args[0])
            self.assertTrue(run.call_args.kwargs['check'])

    def test_already_loaded_old_package_requires_restart(self):
        with patch.object(setup, 'ANALYSIS_PACKAGES', {'example': ('example_loaded', '1.0')}), \
             patch.object(setup, 'installed_version', return_value='1.0'), \
             patch.dict(setup.sys.modules, {'example_loaded': SimpleNamespace(__version__='0.9')}):
            with self.assertRaisesRegex(RuntimeError, 'Restart session'):
                setup.ensure_analysis_packages()

    def test_existing_non_repository_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(setup.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'no existing files were overwritten'):
                setup.checked_checkout(folder)
            run.assert_not_called()

    def test_wrong_checkout_is_never_reset(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / '.git').mkdir()
            with patch.object(setup.subprocess, 'check_output', side_effect=['wrong\n', setup.SOURCE_URL]), \
                 patch.object(setup.subprocess, 'run') as run:
                with self.assertRaisesRegex(RuntimeError, 'will not reset'):
                    setup.checked_checkout(folder)
                run.assert_not_called()

    def test_matching_checkout_is_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / '.git').mkdir()
            with patch.object(setup.subprocess, 'check_output', side_effect=[setup.SOURCE_COMMIT, setup.SOURCE_URL]), \
                 patch.object(setup.subprocess, 'run') as run:
                self.assertEqual(setup.checked_checkout(folder), Path(folder))
                run.assert_not_called()

    def test_new_checkout_uses_exact_commit(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'new-checkout'
            with patch.object(setup.subprocess, 'check_output', side_effect=[setup.SOURCE_COMMIT, setup.SOURCE_URL]), \
                 patch.object(setup.subprocess, 'run') as run:
                setup.checked_checkout(target)
                self.assertEqual(run.call_args_list[1].args[0][-2:], ['--detach', setup.SOURCE_COMMIT])

    def test_protected_input_mismatch_stops(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'input'; path.write_bytes(b'original')
            expected = hashlib.sha256(b'original').hexdigest()
            with patch.object(setup, 'PROTECTED_FILES', {'input': expected}):
                setup.verify_inputs(folder)
                path.write_bytes(b'changed')
                with self.assertRaisesRegex(RuntimeError, 'Protected input differs'):
                    setup.verify_inputs(folder)
                self.assertEqual(path.read_bytes(), b'changed')

    def test_all_notebooks_embed_identical_bootstrap_before_analysis(self):
        root = Path(__file__).resolve().parents[2]
        expected = (root / 'src/tools/colab_setup.py').read_text()
        notebooks = sorted((root / 'notebooks').glob('0[1-5]_*.ipynb'))
        self.assertEqual(len(notebooks), 5)
        for path in notebooks:
            book = json.loads(path.read_text())
            codes = [c for c in book['cells'] if c['cell_type'] == 'code']
            self.assertEqual(codes[0]['id'], 'colab-bootstrap')
            self.assertEqual(codes[0]['source'], expected)
            instructions = next(c for c in book['cells'] if c['id'] == 'colab-instructions')
            self.assertIn('Restart session', instructions['source'])


if __name__ == '__main__':
    unittest.main()
