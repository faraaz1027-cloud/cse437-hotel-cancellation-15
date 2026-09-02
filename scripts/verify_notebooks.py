"""Execute five notebooks in fresh kernels in an isolated copy; preserve source evidence."""
import argparse
import hashlib
import json
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    temporary = Path(tempfile.mkdtemp(prefix='cse437-reproduction-'))
    replica = temporary / 'repository'
    shutil.copytree(root, replica, ignore=shutil.ignore_patterns('__pycache__', '.git', '.venv'))
    kernel_root = temporary / 'kernels'
    kernel = kernel_root / 'cse437-repro'
    kernel.mkdir(parents=True)
    (kernel / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'CSE437 reproduction', 'language': 'python'}))
    manager = KernelSpecManager(kernel_dirs=[str(kernel_root)])
    before = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for folder in ['data', 'models', 'notebooks'] for p in (root / folder).rglob('*')
              if p.is_file() and '__pycache__' not in p.parts}
    records = []
    for path in sorted((replica / 'notebooks').glob('0[1-5]_*.ipynb')):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                cell.outputs = []
                cell.execution_count = None
        start = time.monotonic()
        km = KernelManager(kernel_name='cse437-repro', kernel_spec_manager=manager)
        client = NotebookClient(notebook, timeout=1800, allow_errors=False, km=km,
                                resources={'metadata': {'path': str(replica)}})
        print('RUN ' + path.name, flush=True)
        try:
            client.execute()
        except Exception as error:
            failure = {'status': 'failed_or_blocked', 'notebook': path.name,
                       'completed_notebooks': records,
                       'error_type': type(error).__name__, 'error': str(error),
                       'isolated_copy': str(replica)}
            (temporary / 'verification.json').write_text(json.dumps(failure, indent=2) + '\n')
            print('Verification did not pass. See ' + str(temporary / 'verification.json'), flush=True)
            raise
        nbformat.validate(notebook)
        count = sum(cell.cell_type == 'code' for cell in notebook.cells)
        notebook.metadata.execution_provenance = {
            'method': 'fresh ipykernel, top-to-bottom in isolated repository copy',
            'executed_code_cells': count, 'fresh_jupyter_kernel_verified': True,
            'canonical_nbformat_validation': True,
            'original_model_and_evidence_preserved': True,
            'note': 'Reproduction verification does not authorize model reselection or test-based tuning.'}
        nbformat.write(notebook, path)
        records.append({'notebook': path.name, 'code_cells': count,
                        'seconds': round(time.monotonic() - start, 3),
                        'fresh_kernel': True, 'canonical_format_valid': True, 'errors': 0})
        print('PASS ' + path.name, flush=True)
    for relative, expected in before.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected, relative
    result = {'status': 'passed', 'python': platform.python_version(), 'notebooks': records,
              'notebook_count': len(records), 'original_repository_unchanged': True,
              'scope': 'Fresh-kernel execution; final notebook verifies saved frozen test evidence.',
              'isolated_copy': str(replica)}
    assert len(records) == 5
    (temporary / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
