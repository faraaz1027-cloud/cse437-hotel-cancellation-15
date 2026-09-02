"""Check report/evidence without starting kernels or changing frozen scientific results."""
import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import nbformat
import pandas as pd
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protected-docx', type=Path, default=None,
                        help='Optional private original; only its hash is recorded, never its contents/path.')
    parser.add_argument('--visual-review-confirmed', action='store_true',
                        help='Record a separately performed review of every page of this exact PDF.')
    args = parser.parse_args()
    text = (ROOT / 'report/report.md').read_text()
    pdf = PdfReader(ROOT / 'report/report.pdf')
    assert len(pdf.pages) == 10
    summary = text.split('## Summary\n\n')[1].split('\n\n**Submission status:')[0]
    assert 150 <= len(summary.split()) <= 200
    pdf_text = ' '.join(' '.join(p.extract_text().split()) for p in pdf.pages)
    assert ' '.join(summary.split()) in pdf_text
    required = ['1. Problem and Dataset', '2. Data Handling and Preprocessing',
                '3. Statistical Analysis', '4. Feature Engineering', '5. Modeling and Validation',
                '6. Hyperparameter Tuning', '7. Results, Visualization and Error Analysis',
                '8. Limitations and Next Steps', '9. Contributions', 'References']
    for heading in required:
        assert heading in text and heading in pdf_text, heading
    approved_questions = (ROOT / 'README.md').read_text().split('### Research questions\n\n')[1].split('\n\n')[0]
    for line in approved_questions.splitlines():
        question = re.sub(r'^\d+\. ', '', line)
        assert question in text and question in pdf_text, question
    assert 'Section 05' in pdf_text and 'Summer 2026' in pdf_text
    links = re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text)
    for target in links:
        if not target.startswith(('https://', 'http://', '#')):
            assert (ROOT / 'report' / target).exists(), target
    metrics = pd.read_csv(ROOT / 'data/results/step15/test_comparison.csv')
    for _, row in metrics.iterrows():
        for metric in ['f1', 'accuracy', 'precision', 'recall', 'roc_auc', 'brier_score']:
            assert f'{row[metric]:.6f}' in text, (row.model, metric)
    frozen_count = 0
    for relative in ['data/results/step12/tuning_summary.json',
                     'data/results/step13/evaluation_summary.json',
                     'data/results/step15/comparison_summary.json']:
        record = json.loads((ROOT / relative).read_text())
        for path, expected in record['output_sha256'].items():
            assert sha(ROOT / path) == expected, path
            frozen_count += 1
    selection_sha = sha(ROOT / 'data/results/step12/final_selection.json')
    assert selection_sha == 'e495222d6050492784b334110973b219dce2d3e9deaf516d311878462c6e47b6'
    notebooks = []
    for path in sorted((ROOT / 'notebooks').glob('0[1-5]_*.ipynb')):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        notebooks.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path),
                          'canonical_format': 'passed',
                          'code_cells': sum(c.cell_type == 'code' for c in notebook.cells)})
    assert len(notebooks) == 5
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-q'],
                           cwd=ROOT, capture_output=True, text=True)
    assert tests.returncode == 0, tests.stderr
    count = int(re.search(r'Ran (\d+) tests', tests.stderr).group(1))
    deps = subprocess.run([sys.executable, '-m', 'pip', 'check'], cwd=ROOT,
                          capture_output=True, text=True)
    assert deps.returncode == 0, deps.stdout + deps.stderr
    private_docx_checked = False
    if args.protected_docx:
        assert sha(args.protected_docx) == 'ef5d8fb97c094ad17cf8f40e21779aff24819c2531a1d416ff15453d89c61360'
        private_docx_checked = True
    raw = ROOT / 'data/raw/hotel_bookings.csv'
    if raw.exists():
        assert sha(raw) == '7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06'
    record = {
        'step': 15, 'status': 'report_checks_passed_submission_signoff_pending',
        'responsible_member': 'Sadat; both members review',
        'python': platform.python_version(), 'report_pages': len(pdf.pages),
        'summary_words': len(summary.split()), 'report_markdown_sha256': sha(ROOT / 'report/report.md'),
        'report_pdf_sha256': sha(ROOT / 'report/report.pdf'),
        'required_sections_and_unchanged_questions': 'passed',
        'internal_report_links': 'passed', 'test_table_matches_frozen_metrics': True,
        'frozen_artifact_hashes_checked': frozen_count, 'selection_sha256': selection_sha,
        'notebooks': notebooks, 'unit_tests_passed': count,
        'dependency_consistency': deps.stdout.strip(), 'private_original_docx_hash_checked': private_docx_checked,
        'raw_dataset_present_locally': raw.exists(),
        'fresh_kernel_status': 'pending - host startup permission restriction; local check required',
        'raw_publication_status': 'pending - local presence is not proof of GitHub publication',
        'member_contributions': 'unconfirmed - assigned responsibilities only',
        'visual_review': ('All 10 rendered pages separately reviewed' if args.visual_review_confirmed
                          else 'Separate manual review required after every PDF rebuild'),
        'submission_ready': False,
    }
    output = ROOT / 'report/final_verification.json'
    output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
