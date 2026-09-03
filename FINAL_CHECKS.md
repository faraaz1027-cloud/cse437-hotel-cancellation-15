# Reproducibility and submission requirements

The Step 15.1 workflow repair is implemented. The supplied Step 15.3 run completed all five notebooks in fresh kernels with zero errors and passed 70 tests. Execution passed with numerical reproduction differences, not an unqualified reproduction pass. See [verification review](report/verification_review.md).

## Diagnosed execution issue and repair

The original host could not start a separate Jupyter kernel. A subsequent local audit passed dependency consistency and 58 unit tests, and completed notebooks 01–04. Notebook 05 failed in the supplementary comparison with:

```text
ValueError: Frozen Step 12 model settings changed.
```

Notebook 04 regenerated tuning evidence and selected C=0.1, while notebook 05 expected the original C=1 model. Notebook 04 now creates a separate external development workspace. Its new comparison report records changed scores/settings without overwriting the original selection. Notebook 05 requires cached final-test evidence and retains the original settings/hash checks; missing evidence cannot trigger retraining.

The numerical cause of the earlier score differences remains unconfirmed. The runner captures environment/build details for further investigation. See [repair notes](report/reproducibility_repair.md).

Retain the failed verification record, rerun tuning tables, selected configuration, Python/package versions and any locally modified runner. Diagnose the mismatch separately. Unit tests and format checks are not substitutes for end-to-end execution.

## Source data

Place the untouched `hotel_bookings.csv` at `data/raw/hotel_bookings.csv`. Expected SHA-256:

`7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06`

Raw-data publication is reported complete by the group; its independent recheck is deferred. Preserve [source attribution](data/README.md) and CC BY 4.0 terms. The original acquisition date/version are unknown. Keep the original private CSE437 DOCX unchanged.

## Verification commands

Use Python 3.12 and a new clone of the repair commit. Do not pull over an existing working directory or failed-run evidence. Committed bytes must be preserved; this repository supplies a `.gitattributes` rule for that purpose.

```bash
python -m pip install -r requirements.txt
python -m pip check
python -m unittest discover -s tests -q
python scripts/verify_notebooks.py
```

The runner creates a temporary copy and executes notebooks 01–05 in fresh kernels. It also saves fresh tuning evidence separately. Save the printed verification directory, including `verification.json`, `reproduction_comparison.json`, `pip_freeze.txt`, `development_run/` and executed notebooks. If execution fails, preserve `traceback.txt` and the partial notebook outputs.

Interpret the result explicitly:

| Exit code | Meaning |
| --- | --- |
| 0 | All five notebooks executed and the recorded tuning comparison matched; not a claim of full-pipeline numerical reproduction or submission readiness. |
| 2 | All five notebooks executed, but development evidence differs. `status` is `passed_with_reproduction_differences`; review is still required. |
| 1 | Execution or integrity failed; retain diagnostics and do not mark verification complete. |

Require five completed notebooks, zero cell errors, `original_repository_unchanged: true` and `frozen_evidence_unchanged: true`. These conditions are present in the reviewed Step 15.3 bundle. A changed winner must be reported, never promoted to the original test evaluation. Do not change tolerances, settings or assertions merely to obtain exit code 0. The supplied archive lacks exact checkout commit provenance; a final exact-commit check remains pending. The numerical warning is documented, not resolved.

`report/final_verification.json` is a historical snapshot. Its earlier hashes and execution status do not certify files edited during cleanup.

## Report and authorship

- Assigned responsibilities are confirmed by the group and recorded in the report; verify attributable commits and complete joint review of the whole project.
- Review and finalize the author-supplied provisional declaration before submission, ensuring it accurately covers the assistance used.
- Rebuild the PDF after report changes with `python scripts/build_report_pdf.py`; inspect every page and retain the 10-page limit.
- Preserve late-comparison timing, source limitations and unresolved checks.
- Confirm that raw data, five executed notebooks, code, dependencies, figures, saved model and both report formats are publicly accessible.

Submit one public repository link through the faculty's designated channel only after these requirements are satisfied:

https://github.com/faraaz1027-cloud/cse437-hotel-cancellation-15
