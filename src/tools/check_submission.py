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

ROOT = Path(__file__).resolve().parents[2]


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
    approved_questions = (ROOT / 'README.md').read_text().split('## Research questions\n\n')[1].split('\n\n')[0]
    for line in approved_questions.splitlines():
        question = re.sub(r'^\d+\. ', '', line)
        assert question in text and question in pdf_text, question
    assert 'Section 05' in pdf_text and 'Summer 2026' in pdf_text
    links = re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text)
    for target in links:
        if not target.startswith(('https://', 'http://', '#')):
            assert (ROOT / 'report' / target).exists(), target
    metrics = pd.read_csv(ROOT / 'data/processed/results/test_comparison/test_comparison.csv')
    for _, row in metrics.iterrows():
        for metric in ['f1', 'accuracy', 'precision', 'recall', 'roc_auc', 'brier_score']:
            assert f'{row[metric]:.6f}' in text, (row.model, metric)
    frozen_count = 0
    for relative in ['data/processed/results/tuning/tuning_summary.json',
                     'data/processed/results/evaluation/evaluation_summary.json',
                     'data/processed/results/test_comparison/comparison_summary.json']:
        record = json.loads((ROOT / relative).read_text())
        for path, expected in record['output_sha256'].items():
            assert sha(ROOT / path) == expected, path
            frozen_count += 1
    selection_sha = sha(ROOT / 'data/processed/results/tuning/final_selection.json')
    assert selection_sha == '68c4072f5c95e3a9f927a8b70a9e96aea8adbd4f7509f4d1c30ba6f7889f3b1b'
    notebooks = []
    for path in sorted((ROOT / 'notebooks').glob('0[1-5]_*.ipynb')):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        notebooks.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path),
                          'canonical_format': 'passed',
                          'code_cells': sum(c.cell_type == 'code' for c in notebook.cells)})
    assert len(notebooks) == 5
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'src/tests', '-q'],
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
        'analysis': 'test_comparison', 'status': 'report_checks_passed_submission_signoff_pending',
        'python': platform.python_version(), 'report_pages': len(pdf.pages),
        'summary_words': len(summary.split()), 'report_markdown_sha256': sha(ROOT / 'report/report.md'),
        'report_pdf_sha256': sha(ROOT / 'report/report.pdf'),
        'required_sections_and_unchanged_questions': 'passed',
        'internal_report_links': 'passed', 'test_table_matches_frozen_metrics': True,
        'frozen_artifact_hashes_checked': frozen_count, 'selection_sha256': selection_sha,
        'notebooks': notebooks, 'unit_tests_passed': count,
        'dependency_consistency': deps.stdout.strip(), 'private_original_docx_hash_checked': private_docx_checked,
        'raw_dataset_present_locally': raw.exists(),
        'fresh_kernel_status': 'Prior Windows run passed with numerical differences; reorganized notebooks require a new exact-source run.',
        'raw_publication_status': 'Original CSV publication verified by Git blob identity; local checksum checked again.',
        'member_contributions': 'User-confirmed responsibilities; both accounts appear in Git history; joint final review remains open.',
        'visual_review': ('All 10 rendered pages separately reviewed' if args.visual_review_confirmed
                          else 'Separate manual review required after every PDF rebuild'),
        'submission_ready': False,
    }
    output = ROOT / 'data/processed/verification/submission_checks.json'
    output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
