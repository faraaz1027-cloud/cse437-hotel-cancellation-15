# CSE437 Group 15 — Progress and resume checkpoint

## Current position

**Step 8 is complete. Resume at Step 9 — Sadat.** Faraaz is the assigned owner of the completed Step 8 work; review and actual contributions remain for the members to record.

Repository: https://github.com/faraaz1027-cloud/cse437-hotel-cancellation-15

Faraaz owns Steps 1–8 and report Sections 1–3. Sadat owns Steps 9–15 and
report Sections 4–8/assembly. Both review the final project, references, and
actual contribution statement; each member commits actual work from their own
account. AI assistance is disclosed, not counted as work personally performed
by a member.

The original CSE437 document, approved dataset, target, problem statement, and
three research questions must remain unchanged. The approved wording is in
README.md. The project uses hotel booking demand and `is_canceled`, not the
separately uploaded Airbnb data or NYC map. Do not publish private source
documents or verbatim faculty feedback as part of resuming.

## Fifteen-step plan

| Step | Work | Owner | Status |
| --- | --- | --- | --- |
| 1 | Download and preserve original dataset | Faraaz | Supplied/audited; raw CSV repository upload and provenance details still pending |
| 2 | Repository setup | Faraaz | Complete; public repository and scaffold present |
| 3 | Record analysis commitments without rewriting proposal | Faraaz | Documented; focus on lead time, deposit type, and prior cancellations |
| 4 | Raw-data audit | Faraaz | Complete; Notebook 01 and audit evidence |
| 5 | Direct leakage removal and eligible cohort | Faraaz | Complete; four processed files published |
| 6 | Freeze holdout and forward validation | Faraaz | Complete; assignments and plan published |
| 7 | Documented preprocessing inside training folds | Faraaz | Complete; implementation, executed notebook, summary, schemas, and tests |
| 8 | Descriptive statistics and development-only EDA | Faraaz | Complete; 14 tables, three figures, Notebook 01 outputs, report Sections 1–3 draft, and handoff |
| 9 | Derived features | Sadat | **Next — not started** |
| 10 | Feature selection and dimensionality reduction | Sadat | Pending; both must be demonstrated |
| 11 | Baseline and two model families | Sadat | Pending |
| 12 | Hyperparameter tuning | Sadat | Pending |
| 13 | Final test evaluation and error analysis | Sadat | Pending |
| 14 | Answer the unchanged three questions | Sadat | Pending |
| 15 | Report, reproducibility, and submission | Sadat + both reviewers | Pending |

## What has been verified

- Original source: 119,390 rows and 32 columns; original bytes preserved.
- Eligible cohort: 119,210 rows; 180 known-zero-guest records excluded.
- Status fields excluded from predictors; target and metadata stored separately.
- Step 5 retains repeated records with groups; unusual ADR is handled without
  deleting more bookings.
- Development: 95,415 arrivals from 2015-07-01 through 2017-04-22.
- Final test: 23,795 arrivals from 2017-04-23 through 2017-08-31.
- Three expanding forward validation folds, with no within-fold row/group
  overlap or test contamination. See the immutable Step 6 plan.
- Step 7 keeps 25 initial source fields, excluding company ID, assigned room,
  booking changes, and waiting-list days. This is a policy choice, not the
  statistical selection/reduction required in Step 10.
- Missing numbers use training-fold medians; negative ADR becomes missing;
  missing country is `Unknown`; missing agent is `NoAgent` under source semantics.
- Log1p transforms four skewed numeric fields; standardization is training-only;
  the tree variant omits standardization. Categories use training-only one-hot
  vocabularies and safe unseen-category handling. Children/ADR have missing flags.
- Encoded widths are 328, 422, and 491 for folds 1–3; each fold's train/validation
  widths match and all encoded values are finite. No test rows processed.
- All 22 focused tests pass, including three new EDA checks. Notebook 02's 13 code cells executed sequentially
  in a fresh Python process with IPython and real outputs saved.
- Step 8 uses development rows only: 34,473 of 95,415 bookings canceled (36.13%).
  Lead-time and deposit relationships, non-monotonic prior cancellations,
  hotel differences, and repeated-profile sensitivity are documented.
- Equal total weight per duplicate-profile group gives an overall development
  rate of 25.26%, versus 36.13% per booking; it is an EDA sensitivity analysis,
  not a change to the cohort, split, or model-fitting weights.
- Notebook 01's five new EDA cells executed with real Python outputs; its ten
  previous raw-audit cells retain their earlier outputs. JSON structure checks
  passed. The current runtime lacks Jupyter/nbformat, so neither a full fresh-
  kernel run nor canonical nbformat validation was performed for Step 8.
- No predictive model has been trained or evaluated.

## Resume Step 9

1. Read [HANDOFF_TO_SADAT.md](HANDOFF_TO_SADAT.md), this checkpoint, and the
   Step 8 evidence. Preserve the approved proposal wording and frozen split.
2. Implement a reusable derived-feature transformer and Notebook 03, with
   explicit formulas, source fields, missing-value rules, and justification.
3. Consider total nights/guests, prior cancellation share with a no-history
   indicator, company presence, and cyclic calendar features. Confirm the
   company-presence meaning before adoption. Do not treat proposals as selected
   features or claim performance improvements before validation.
4. Extend the strict preprocessing schema deliberately: the Step 7 cleaner
   rejects undeclared new columns. Fit learned transforms inside each training
   fold, preserve row alignment, and use development-relative CV indices.
5. Verify formulas and edge cases, save Notebook 03 outputs, and document the
   final engineered feature list for report Section 4.
6. Leave the held-out test set untouched and Step 10 selection/reduction for
   its own stage. After Step 9, both statistical selection and dimensionality
   reduction must be demonstrated in Step 10.

For modeling later, keep preprocessing and all learned feature selection/
reduction inside the pipeline passed to CV. Do not prefit on all development
data before cross-validation. Primary selection metric is mean cancellation-
class F1 over the three folds; other metrics are accuracy, precision, recall,
and ROC-AUC. Planned models are a majority baseline, logistic regression, and
random forest, with seed 42 for stochastic components. Reserve final test
evaluation for Step 13 after freezing every choice.

## Files to open

- [Executed Notebook 02](notebooks/02_preprocessing.ipynb)
- [Notebook 01: raw audit and development EDA](notebooks/01_data_audit_and_eda.ipynb)
- [Step 8 tables, figures, and reproduction](data/eda/README.md)
- [Report Sections 1–3 draft](report/report.md)
- [Step 9 handoff](HANDOFF_TO_SADAT.md)
- [Frozen split instructions](data/splits/README.md)
- [Step 7 results and pipeline reuse](data/processed/step7/README.md)
- [Step 7 report](report/step7_preprocessing.md)
- [Preprocessing factory](src/preprocessing.py)

From a local clone with dependencies installed:

```bash
python -m src.splitting
python -m src.preprocessing_audit
python -m src.development_eda
python -m unittest discover -s tests -v
```

These commands use committed processed files. Running all of Notebook 02
requires the original `data/raw/hotel_bookings.csv` for its Step 5 cells.

## Remaining reproducibility/submission items

- Add the original CSV under `data/raw/` (it is below the faculty's 50 MB limit).
  Its exact SHA-256 and source link are in data/README.md. Record the actual
  source version/download date and exact licence; do not invent missing details.
- Validate a clean dependency install and separate fresh Jupyter-kernel runs;
  the earlier IPython and current Step 8 Python execution checks do not establish
  that final gate. Canonical notebook format validation is also pending.
- Keep notebooks numbered 01–05 with saved outputs and relative paths.
- Preserve frozen data/split hashes and the original CSE437 document.
- Complete report/report.md and report/report.pdf to the supplied template,
  at most 10 pages. Current report/report.md drafts Sections 1–3; later results,
  final summary, contribution records, and PDF assembly remain pending.
- Record genuine member contributions, verified references, and AI assistance.
- Submit the one public GitHub repository link when the full project is ready.

## Suggested message to resume

“Continue my CSE437 Group 15 project from Step 9. Read PROJECT_STATUS.md and
HANDOFF_TO_SADAT.md in https://github.com/faraaz1027-cloud/cse437-hotel-cancellation-15
first. Sadat owns Step 9; preserve the original proposal and frozen evaluation split.”
