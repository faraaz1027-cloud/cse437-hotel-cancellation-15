# Reproducibility and submission requirements

Final verification is deferred. Documentation cleanup does not resolve or certify the outstanding checks below.

## Known execution issue

The original host could not start a separate Jupyter kernel. A subsequent local audit passed dependency consistency and 58 unit tests, and completed notebooks 01–04. Notebook 05 failed in the supplementary comparison with:

```text
ValueError: Frozen Step 12 model settings changed.
```

The audit reports different rerun settings from the frozen selection. The cause has not been established. Preserve the frozen model, threshold, scores and evidence; do not bypass the assertion or select settings using test performance.

Retain the failed verification record, rerun tuning tables, selected configuration, Python/package versions and any locally modified runner. Diagnose the mismatch separately. Unit tests and format checks are not substitutes for end-to-end execution.

## Source data

Place the untouched `hotel_bookings.csv` at `data/raw/hotel_bookings.csv`. Expected SHA-256:

`7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06`

The 16.86 MB raw file still requires publication under the faculty's under-50-MB rule. Preserve [source attribution](data/README.md) and CC BY 4.0 terms. The original acquisition date/version are unknown. Keep the original private CSE437 DOCX unchanged.

## Verification commands

Use Python 3.12 and an isolated repository copy. After resolving the known mismatch:

```bash
python -m pip install -r requirements.txt
python -m pip check
python -m unittest discover -s tests -q
python scripts/verify_notebooks.py
```

The runner creates a temporary copy and executes notebooks 01–05 in fresh kernels. A successful record must show five notebooks and zero errors. Save the record and executed copies; do not replace published outputs until reviewed. The runner has not yet completed a successful end-to-end run.

`report/final_verification.json` is a historical snapshot. Its earlier hashes and execution status do not certify files edited during cleanup.

## Report and authorship

- Confirm each member's actual implementation/review work and attributable commits; replace pending contribution records with truthful statements.
- Complete the required declaration before submission.
- Rebuild the PDF after report changes with `python scripts/build_report_pdf.py`; inspect every page and retain the 10-page limit.
- Preserve late-comparison timing, source limitations and unresolved checks.
- Confirm that raw data, five executed notebooks, code, dependencies, figures, saved model and both report formats are publicly accessible.

Submit one public repository link through the faculty's designated channel only after these requirements are satisfied:

https://github.com/faraaz1027-cloud/cse437-hotel-cancellation-15
