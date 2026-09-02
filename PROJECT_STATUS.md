# CSE437 Group 15 — Progress and resume checkpoint

## Current position

**Step 10 is complete. Resume at Step 11 — Sadat.** Sadat is the assigned owner of Steps 9–10; review and actual contributions remain for the members to record.

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
| 9 | Derived features | Sadat | Complete; eight features, fold verification, Notebook 03 outputs, and report explanation |
| 10 | Feature selection and dimensionality reduction | Sadat | Complete; F-score selection, centered numeric PCA, four-way reference comparison, retained lists/components and decision |
| 11 | Baseline and two model families | Sadat | **Next — not started** |
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
- All 35 focused tests pass, including six Step 10 checks. Notebook 02's 13 code cells executed sequentially
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
- Step 10 now has development validation results from a fixed logistic-regression
  reference (12 fits). The Step 11 baseline/two-family comparison and final
  held-out evaluation have not been completed.
- Step 9 adds eight derived fields and replaces categorical month names with
  a fixed cyclic pair: 24 retained source fields plus eight derived = 32 fields.
  The Step 7 factory remains unchanged. The Step 9 fold widths are 332/421/490;
  train/validation widths match and encoded values are finite in both variants.
- Step 9 retains 604 zero-night development bookings and four unknown guest
  totals before imputation. No target values are read by its audit; no test rows
  are fitted/transformed. Validation leaves training medians/scalers/vocabularies
  unchanged. The company flag means a code is recorded, not verified payment.
- Notebook 03's five Step 9 cells executed sequentially in a fresh Python process with
  saved outputs. Full Jupyter-kernel and canonical format checks remain pending.
- Step 10 appends five more actually executed Python cells to Notebook 03
  (ten code cells total). It compares full, selected, numeric PCA, and selected
  then numeric PCA, using the same preprocessing and forward folds.
- Current preference: `selected`, keeping the top 75% of nonconstant encoded
  training features by ANOVA F. Widths are 247/314/366. Mean reference F1 is
  0.713609 versus 0.693094 full, 0.694633 PCA and 0.701023 combined. Fold 3 is
  slightly worse than full; this is not a uniform improvement or a final score.
- Centered PCA on 23 numeric fields retains 16/15/16 components and at least
  95% numeric training variance. Categorical fields stay sparse. The combined
  mode reduces 20/21/21 selected numeric fields to 15 components in each fold.
- No held-out rows are fitted/transformed/scored; no global full-development
  model or selection mask is fitted. Complete fold masks, rankings, coefficients
  and variance evidence are saved. All 12 reference fits converged.

## Resume Step 11

1. Read this checkpoint, the Step 10 report, and Notebook 03. Preserve the
   approved wording and frozen split. Step 10's preference applies to a fixed
   logistic-regression reference and need not be optimal for random forest.
2. Implement Notebook 04's baseline/model comparison: a majority-class
   DummyClassifier, LogisticRegression, and RandomForestClassifier.
3. Compare the current selected representation with a full-feature control for
   the learned model families. Use a fresh `BookingRepresentation(mode="selected")`
   or `BookingRepresentation(mode="full")` inside each estimator pipeline.
   For trees, full/selected variants can omit scaling; PCA variants require it.
4. Keep preprocessing and selection training-fold fitted. Preserve development
   row order and use development-relative indices from `development_cv`.
   Reuse the frozen metric/threshold policy and seed 42; do not use the test set.
5. Report per-fold F1, accuracy, precision, recall and ROC-AUC, plus the unweighted
   three-fold mean F1. Explain model choices, class imbalance and temporal
   variation; do not treat the reference comparison as final model evaluation.
6. Save actual Notebook 04 outputs and report Section 5. Model hyperparameter
   search is Step 12; freeze all choices before final refit/evaluation in Step 13.

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
- [Notebook 03: derived features](notebooks/03_feature_engineering.ipynb)
- [Step 9 report](report/step9_feature_engineering.md)
- [Step 9 evidence and schemas](data/processed/step9/README.md)
- [Step 10 report and feature decision](report/step10_selection_and_reduction.md)
- [Step 10 comparisons, rankings and PCA evidence](data/processed/step10/README.md)
- [Cloneable representation wrapper](src/representation.py)
- [Sadat's handoff](HANDOFF_TO_SADAT.md)
- [Frozen split instructions](data/splits/README.md)
- [Step 7 results and pipeline reuse](data/processed/step7/README.md)
- [Step 7 report](report/step7_preprocessing.md)
- [Preprocessing factory](src/preprocessing.py)

From a local clone with dependencies installed:

```bash
python -m src.splitting
python -m src.preprocessing_audit
python -m src.development_eda
python -m src.feature_audit
python -m src.representation_audit
python -m unittest discover -s tests -v
```

These commands use committed processed files. Running all of Notebook 02
requires the original `data/raw/hotel_bookings.csv` for its Step 5 cells.

## Remaining reproducibility/submission items

- Add the original CSV under `data/raw/` (it is below the faculty's 50 MB limit).
  Its exact SHA-256 and source link are in data/README.md. Record the actual
  source version/download date and exact licence; do not invent missing details.
- Validate a clean dependency install and separate fresh Jupyter-kernel runs;
  the earlier IPython and Steps 8–10 Python execution checks do not establish
  that final gate. Canonical notebook format validation is also pending.
- Keep notebooks numbered 01–05 with saved outputs and relative paths.
- Preserve frozen data/split hashes and the original CSE437 document.
- Complete report/report.md and report/report.pdf to the supplied template,
  at most 10 pages. Current report/report.md drafts Sections 1–4; later results,
  final summary, contribution records, and PDF assembly remain pending.
- Record genuine member contributions, verified references, and AI assistance.
- Submit the one public GitHub repository link when the full project is ready.

## Suggested message to resume

“Continue my CSE437 Group 15 project from Step 11. Read PROJECT_STATUS.md and
HANDOFF_TO_SADAT.md in https://github.com/faraaz1027-cloud/cse437-hotel-cancellation-15
first. Sadat owns Step 11; preserve the original proposal and frozen evaluation split.”
