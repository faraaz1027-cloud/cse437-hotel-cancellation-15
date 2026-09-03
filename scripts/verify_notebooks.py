"""Fresh-kernel execution with explicit, separate numerical-reproduction status."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from src.reproduction import (assert_frozen_unchanged, environment_record,
                              frozen_manifest, write_json)

NOTEBOOKS = ('01_data_audit_and_eda.ipynb', '02_preprocessing.ipynb',
             '03_feature_engineering.ipynb', '04_modeling_and_tuning.ipynb',
             '05_evaluation_and_error_analysis.ipynb')


def source_manifest(root):
    paths = [p for directory in ('data', 'models', 'notebooks', 'src', 'scripts',
                                 'figures', 'report')
             for p in (root / directory).rglob('*')
             if p.is_file() and '__pycache__' not in p.parts]
    paths += [p for p in root.iterdir() if p.is_file()]
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def execute_notebook(notebook, replica, manager):
    km = KernelManager(kernel_name='cse437-repro', kernel_spec_manager=manager)
    NotebookClient(notebook, timeout=1800, allow_errors=False, km=km,
                   resources={'metadata': {'path': str(replica)}}).execute()


def collect_comparison(notebook, temporary):
    cell = next((c for c in notebook.cells if c.get('id') == 'reproduction-summary'), None)
    if cell is None:
        raise ValueError('Notebook 04 lacks the required reproduction comparison.')
    output = ''.join(o.text for o in cell.outputs
                     if o.output_type == 'stream' and o.name == 'stdout')
    report = json.loads(output)
    run = Path(report['run_directory'])
    saved = json.loads((run / 'reproduction_comparison.json').read_text(encoding='utf-8'))
    if saved != report or not report['frozen_evidence_unchanged']:
        raise ValueError('Reproduction comparison is missing or inconsistent.')
    destination = temporary / 'development_run'
    destination.mkdir()
    # Preserve fresh results and diagnostics without copying processed row-level inputs again.
    for relative in ('data/results', 'figures'):
        shutil.copytree(run / relative, destination / relative)
    for filename in ('run_context.json', 'reproduction_comparison.json'):
        shutil.copy2(run / filename, destination / filename)
    write_json(temporary / 'reproduction_comparison.json', report)
    return report


def run_verification(root, executor=execute_notebook):
    root = Path(root).resolve()
    temporary = Path(tempfile.mkdtemp(prefix='cse437-reproduction-'))
    print('Verification directory: ' + str(temporary), flush=True)
    replica = temporary / 'repository'
    before = source_manifest(root)
    frozen_before = frozen_manifest(root)
    shutil.copytree(root, replica, ignore=shutil.ignore_patterns(
        '__pycache__', '*.pyc', '.git', '.venv*', 'venv'))
    kernel_root = temporary / 'kernels'
    kernel = kernel_root / 'cse437-repro'
    kernel.mkdir(parents=True)
    write_json(kernel / 'kernel.json', {
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'CSE437 reproduction', 'language': 'python'})
    manager = KernelSpecManager(kernel_dirs=[str(kernel_root)])
    records, comparison = [], None
    result = {'status': 'running', 'notebook_execution_status': 'pending',
              'numerical_reproduction_status': 'not_checked',
              'environment': environment_record(), 'isolated_copy': str(replica),
              'verification_directory': str(temporary),
              'scope': 'Fresh notebook execution; compare new development tuning separately '
                       'and verify cached final evaluation. No final-model refit.',
              'submission_ready': False}
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'],
                            capture_output=True, text=True, check=False)
    (temporary / 'pip_freeze.txt').write_text(freeze.stdout + freeze.stderr, encoding='utf-8')
    result['pip_freeze_exit_code'] = freeze.returncode
    current = None
    notebook = None
    try:
        found = tuple(p.name for p in sorted((replica / 'notebooks').glob('0[1-5]_*.ipynb')))
        if found != NOTEBOOKS:
            raise ValueError('Expected exactly the five named analysis notebooks.')
        for name in NOTEBOOKS:
            current = name
            path = replica / 'notebooks' / name
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            for cell in notebook.cells:
                if cell.cell_type == 'code':
                    cell.outputs = []
                    cell.execution_count = None
            start = time.monotonic()
            print('RUN ' + name, flush=True)
            executor(notebook, replica, manager)
            nbformat.validate(notebook)
            if any(o.output_type == 'error' for c in notebook.cells
                   if c.cell_type == 'code' for o in c.outputs):
                raise ValueError('Notebook contains an error output.')
            if name == NOTEBOOKS[3]:
                comparison = collect_comparison(notebook, temporary)
            assert_frozen_unchanged(replica, frozen_before)
            count = sum(cell.cell_type == 'code' for cell in notebook.cells)
            notebook.metadata.execution_provenance = {
                'method': 'fresh ipykernel, top-to-bottom in isolated repository copy',
                'executed_code_cells': count, 'fresh_jupyter_kernel_verified': True,
                'canonical_nbformat_validation': True,
                'frozen_evidence_unchanged': True,
                'note': 'Execution success is not exact numerical reproduction.'}
            if 'workflow_repair' in notebook.metadata:
                notebook.metadata.workflow_repair['fresh_kernel_validated'] = True
                notebook.metadata.workflow_repair['retained_outputs'] = 'replaced by this verification run'
            nbformat.write(notebook, path)
            records.append({'notebook': name, 'code_cells': count,
                            'seconds': round(time.monotonic() - start, 3),
                            'fresh_kernel': True, 'canonical_format_valid': True, 'errors': 0})
            print('PASS ' + name, flush=True)
        if comparison is None:
            raise ValueError('A tuning reproduction comparison is required.')
        result['notebook_execution_status'] = 'passed'
        result['numerical_reproduction_status'] = comparison['status']
        result['status'] = ('passed_with_reproduction_differences'
                            if comparison['status'] != 'matched_recorded_tuning_evidence'
                            else 'passed_execution_and_recorded_tuning_comparison')
        result['reproduction_comparison_file'] = str(temporary / 'reproduction_comparison.json')
        result['development_artifacts'] = str(temporary / 'development_run')
        result['frozen_evaluation_status'] = 'cached_evidence_verified; not retrained'
    except Exception as error:
        if notebook is not None and current is not None:
            nbformat.write(notebook, replica / 'notebooks' / current)
        result.update({'status': 'failed_or_blocked', 'notebook_execution_status': 'failed_or_blocked',
                       'notebook': current, 'error_type': type(error).__name__, 'error': str(error)})
        (temporary / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        result['notebooks'] = records
        result['notebook_count'] = len(records)
        result['original_repository_unchanged'] = source_manifest(root) == before
        try:
            result['frozen_evidence_unchanged'] = frozen_manifest(replica) == frozen_before
        except (OSError, ValueError) as integrity_error:
            result['frozen_evidence_unchanged'] = False
            result['integrity_error'] = str(integrity_error)
        if not result['original_repository_unchanged'] or not result['frozen_evidence_unchanged']:
            result['status'] = 'failed_integrity_check'
            result['notebook_execution_status'] = 'failed_integrity_check'
        if comparison is not None:
            result['numerical_reproduction_status'] = comparison['status']
        write_json(temporary / 'verification.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=PROJECT)
    result = run_verification(parser.parse_args().root)
    if result['notebook_execution_status'] != 'passed':
        raise SystemExit(1)
    # Distinct nonzero exit code: execution passed, numerical review still required.
    if result['numerical_reproduction_status'] == 'differences_detected':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
