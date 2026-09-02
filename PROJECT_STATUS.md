# CSE437 Group 15 — Progress and resume checkpoint

## Current position

**Step 7 is complete and committed. Resume at Step 8 — Faraaz.**

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
| 8 | Descriptive statistics and development-only EDA | Faraaz | **Next — not started** |
| 9 | Derived features | Sadat | Pending |
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
- All 19 focused tests pass. Notebook 02's 13 code cells executed sequentially
  in a fresh Python process with IPython and real outputs saved.
- No predictive model has been trained or evaluated.

## Resume Step 8

1. Read this checkpoint, README.md, and the Step 7 report.
2. Use the frozen assignments to select development rows in the original
   Step 5 file order. Use the Step 6 loading example; never reorder one aligned
   file independently.
3. Add development-only descriptive statistics and relationship figures to
   Notebook 01's pending EDA portion. Keep its raw audit clearly separate.
4. Examine the committed factors: lead time, deposit type, and previous
   cancellations. Report observed associations, not causal effects.
5. Explain missingness and skew/outliers; save 2–3 useful relationship figures
   and 3–5 supported observations for report Section 3. Faraaz drafts Sections
   1–3 without altering the proposal wording.
6. Preserve the holdout and CV plan. Do not train/tune models, change the split
   based on results, or analyze held-out feature/target relationships.
7. Prepare the handoff to Sadat for Step 9. Potential later features include
   total nights/guests, prior-history ratios, company presence, and cyclic
   calendar terms. The first CV training window lacks several validation
   months, so calendar coverage is an explicit limitation to address.

For modeling later, keep preprocessing and all learned feature selection/
reduction inside the pipeline passed to CV. Do not prefit on all development
data before cross-validation. Primary selection metric is mean cancellation-
class F1 over the three folds; other metrics are accuracy, precision, recall,
and ROC-AUC. Planned models are a majority baseline, logistic regression, and
random forest, with seed 42 for stochastic components. Reserve final test
evaluation for Step 13 after freezing every choice.

## Files to open

- [Executed Notebook 02](notebooks/02_preprocessing.ipynb)
- [Notebook 01: raw audit; EDA pending](notebooks/01_data_audit_and_eda.ipynb)
- [Frozen split instructions](data/splits/README.md)
- [Step 7 results and pipeline reuse](data/processed/step7/README.md)
- [Step 7 report](report/step7_preprocessing.md)
- [Preprocessing factory](src/preprocessing.py)

From a local clone with dependencies installed:

```bash
python -m src.splitting
python -m src.preprocessing_audit
python -m unittest discover -s tests -v
```

These commands use committed processed files. Running all of Notebook 02
requires the original `data/raw/hotel_bookings.csv` for its Step 5 cells.

## Remaining reproducibility/submission items

- Add the original CSV under `data/raw/` (it is below the faculty's 50 MB limit).
  Its exact SHA-256 and source link are in data/README.md. Record the actual
  source version/download date and exact licence; do not invent missing details.
- Validate a clean dependency install and separate fresh Jupyter-kernel runs;
  the current IPython execution checks do not establish that final gate.
- Keep notebooks numbered 01–05 with saved outputs and relative paths.
- Preserve frozen data/split hashes and the original CSE437 document.
- Complete report/report.md and report/report.pdf to the supplied template,
  at most 10 pages. Current report/report.md is a placeholder, not the final report.
- Record genuine member contributions, verified references, and AI assistance.
- Submit the one public GitHub repository link when the full project is ready.

## Suggested message to resume

“Continue my CSE437 Group 15 project from Step 8. Read PROJECT_STATUS.md in
https://github.com/faraaz1027-cloud/cse437-hotel-cancellation-15 first. Faraaz
owns Step 8; preserve the original proposal and frozen evaluation split.”
