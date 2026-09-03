"""Isolate development reruns; compare them without changing frozen evaluation."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import platform
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path


FROZEN_FOLDERS = ('data/processed/results/tuning', 'data/processed/results/evaluation',
                  'data/processed/results/test_comparison', 'models')
FROZEN_FIGURES = ('figures/08_hyperparameter_tuning.png',
                  'figures/09_final_test_performance.png',
                  'figures/10_final_error_analysis.png')
SELECTION = 'data/processed/results/tuning/final_selection.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    """New diagnostic files use portable paths and LF, without touching originals."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n',
                          encoding='utf-8', newline='\n')


def frozen_manifest(root):
    root = Path(root).resolve()
    if not (root / SELECTION).is_file():
        raise FileNotFoundError('Published frozen selection is required.')
    paths = [p for folder in FROZEN_FOLDERS for p in (root / folder).rglob('*')
             if p.is_file() and '__pycache__' not in p.parts]
    paths += [root / p for p in FROZEN_FIGURES if (root / p).is_file()]
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(paths)}


def assert_frozen_unchanged(root, before):
    if frozen_manifest(root) != before:
        raise ValueError('Published frozen evidence changed during reproduction.')


def environment_record():
    import numpy
    import pandas
    import scipy
    import sklearn
    from threadpoolctl import threadpool_info
    details = io.StringIO()
    with redirect_stdout(details):
        numpy.show_config()
        scipy.show_config()
    return {'python': platform.python_version(), 'executable': sys.executable,
            'platform': platform.platform(), 'numpy': numpy.__version__,
            'pandas': pandas.__version__, 'scipy': scipy.__version__,
            'scikit_learn': sklearn.__version__, 'threadpools': threadpool_info(),
            'numeric_build_configuration': details.getvalue()}


def start_development_run(root, parent=None):
    """Copy inputs into a new external directory; copy no final-test results/model."""
    root = Path(root).resolve()
    parent = Path(parent or tempfile.gettempdir()).resolve()
    if parent == root or root in parent.parents:
        raise ValueError('Development runs must be outside the source repository.')
    before = frozen_manifest(root)
    # No hardlinks: later writes cannot modify the source through a shared inode.
    run = Path(tempfile.mkdtemp(prefix='cse437-development-', dir=parent))
    write_json(run / 'run_context.json', {
        'source_root': str(root), 'frozen_before': before,
        'environment': environment_record(),
        'scope': 'development-only rerun; never promote its selection to final evaluation',
    })
    for relative in ('data/processed', 'src'):
        source = root / relative
        if not source.is_dir():
            raise FileNotFoundError(source)
        shutil.copytree(source, run / relative,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'results', 'verification'))
    (run / 'figures').mkdir()
    assert_frozen_unchanged(root, before)
    return run


def _read_json(root, relative):
    return json.loads((Path(root) / relative).read_text(encoding='utf-8'))


def _portable_keys(value):
    if isinstance(value, dict):
        return {k.replace('\\', '/'): _portable_keys(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_portable_keys(v) for v in value]
    return value


def _table_differences(original, rerun, relative, keys, columns):
    def read(root):
        with (Path(root) / relative).open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        result = {tuple(row[k] for k in keys): row for row in rows}
        if not rows or len(rows) != len(result):
            raise ValueError('Empty table or duplicate comparison keys: ' + relative)
        return result
    left, right = read(original), read(rerun)
    if left.keys() != right.keys():
        raise ValueError('Candidate/fold membership changed: ' + relative)
    differences = []
    for key in sorted(left):
        for column in columns:
            a, b = float(left[key][column]), float(right[key][column])
            if not math.isfinite(a) or not math.isfinite(b):
                raise ValueError('Nonfinite metric in reproduction comparison.')
            if a != b:
                differences.append({**dict(zip(keys, key)), 'metric': column,
                                    'published': a, 'rerun': b, 'delta': b - a})
    return differences


def compare_tuning(original, rerun):
    """Compare recorded development evidence; never read held-out metrics."""
    original, rerun = Path(original), Path(rerun)
    a, b = _read_json(original, SELECTION), _read_json(rerun, SELECTION)
    selected_keys = ('candidate', 'family', 'representation', 'search_parameters',
                     'estimator_parameters', 'threshold')
    published = {k: a[k] for k in selected_keys}
    repeated = {k: b[k] for k in selected_keys}
    prefix = 'data/processed/results/tuning/'
    left = _read_json(original, prefix + 'tuning_summary.json')
    right = _read_json(rerun, prefix + 'tuning_summary.json')
    fields = ('f1', 'accuracy', 'precision', 'recall', 'roc_auc')
    candidate_changes = _table_differences(original, rerun, prefix + 'candidate_results.csv',
                                          ('candidate',), ['mean_' + k for k in fields])
    fold_changes = _table_differences(original, rerun, prefix + 'fold_results.csv',
                                     ('candidate', 'fold'),
                                     (*fields, 'tn', 'fp', 'fn', 'tp', 'training_f1'))
    protocol_equal = (_read_json(original, prefix + 'search_protocol.json') ==
                      _read_json(rerun, prefix + 'search_protocol.json'))
    parameters_equal = (_read_json(original, prefix + 'candidate_parameters.json') ==
                        _read_json(rerun, prefix + 'candidate_parameters.json'))
    inputs_equal = _portable_keys(left['input_sha256']) == _portable_keys(right['input_sha256'])
    sources_equal = (_portable_keys(left['source_code_sha256']) ==
                     _portable_keys(right['source_code_sha256']))
    match = (published == repeated and not candidate_changes and not fold_changes and
             protocol_equal and parameters_equal and inputs_equal and sources_equal)
    return {
        'status': 'matched_recorded_tuning_evidence' if match else 'differences_detected',
        'selection_matches': published == repeated,
        'published_selection': published, 'rerun_selection': repeated,
        'published_mean_development_f1': a['mean_development_f1'],
        'rerun_mean_development_f1': b['mean_development_f1'],
        'candidate_metric_differences': candidate_changes,
        'fold_metric_differences': fold_changes,
        'search_protocol_semantically_equal': protocol_equal,
        'candidate_parameters_equal': parameters_equal,
        'recorded_input_hashes_equal': inputs_equal,
        'recorded_source_hashes_equal': sources_equal,
        'search_protocol_bytes_equal': digest(original / prefix / 'search_protocol.json') ==
                                       digest(rerun / prefix / 'search_protocol.json'),
        'published_runtime': left['runtime'], 'rerun_runtime': right['runtime'],
        'scope': 'tuning development evidence only; not full-pipeline numerical certification',
        'cause': 'Not established by this comparison; inspect environment and solver diagnostics.',
    }


def finish_development_run(root, run):
    root, run = Path(root).resolve(), Path(run).resolve()
    if root == run or root in run.parents:
        raise ValueError('Cannot record a rerun inside the source repository.')
    context = _read_json(run, 'run_context.json')
    if Path(context['source_root']).resolve() != root:
        raise ValueError('Development run belongs to another source repository.')
    assert_frozen_unchanged(root, context['frozen_before'])
    report = compare_tuning(root, run)
    report.update({'run_directory': str(run), 'frozen_evidence_unchanged': True,
                   'frozen_selection_promoted_from_rerun': False,
                   'environment': environment_record()})
    write_json(run / 'reproduction_comparison.json', report)
    return report
