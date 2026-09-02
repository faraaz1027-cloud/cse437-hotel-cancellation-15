# CSE437 Group 15 — Progress and resume checkpoint

## Current position

**Step 15 report/PDF assembled and checked; final sign-off pending. Owner: Sadat, with both members reviewing.** Section 05, Summer 2026. The PDF has 10 pages and a 168-word summary; the approved comparison, diagnostic plots, numbered sections and disclosures are included. Frozen model/results are unchanged. Next: local fresh-kernel verification, raw CSV publication and genuine contribution statements; see [FINAL_CHECKS.md](FINAL_CHECKS.md). No Step 16.

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
| 11 | Baseline and two model families | Sadat | Complete; five candidates, 15 fits, Notebook 04 outputs, metrics, figure and Section 5 |
| 12 | Hyperparameter tuning | Sadat | Complete; 20 settings, 60 fits, frozen selection, Notebook 04 outputs and Section 6 |
| 13 | Final test evaluation and error analysis | Sadat | Complete; fitted pipeline, final metrics, predictions, errors, figures, Notebook 05 and Section 7 |
| 14 | Answer the unchanged three questions | Sadat | Complete; evidence-backed answers, source links, limitations and report Sections 7.3–8 |
| 15 | Report, reproducibility, and submission | Sadat + both reviewers | **Report/PDF and technical checks complete; fresh kernels, raw publication and member sign-off remain open** |

## What has been verified

The entries below are stage-specific checkpoints. Statements that the final
test was untouched apply to Steps 5–12; Step 13 subsequently evaluated the
frozen selected model once. Earlier test counts are historical, not the latest
suite size (58 tests).

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
- All 47 focused tests pass, including seven Step 12 checks. Notebook 02's 13 code cells executed sequentially
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
- Step 10 has development validation results from a fixed logistic-regression
  reference (12 fits). Step 11 completes the baseline/two-family comparison;
  Step 13 subsequently completes the final held-out evaluation.
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

- Step 11 compares five fixed candidates on identical forward folds (15 fits).
  Mean F1: majority 0; full LR 0.693094; selected LR 0.713609; full forest
  0.626504; selected forest 0.657994. Selected LR is the untuned leader.
- Full/selected logistic scores match Step 10 within 1e−12. No convergence
  warnings occurred. Forests show large in-sample to validation F1 gaps.
- Notebook 04 contains six actually executed Python cells with saved outputs.
  Current outputs, parameters, schemas, confusion counts and hashes are in
  `data/results/step11/`. Full fresh-kernel validation remains pending.

- Step 12 completes an exhaustive grid with 8 logistic and 12 forest settings:
  60 fits on the same three folds, selected features and threshold 0.5.
- Untuned threshold metrics match Step 11 within 1e−12; secondary forest AUC
  uses 1e−7 tolerance after a documented numerical audit. All fits finish
  with no errors/convergence warnings; no global full-development model is fitted.
- Selected family: **Logistic Regression**; C=1.0, class_weight=balanced; mean development F1
  **0.732102**. Exact settings and lineage are frozen in
  `data/results/step12/final_selection.json` for Step 13, not final test selection.
- Notebook 04 now has 11 code cells: six preserved Step 11 outputs and five new
  sequentially Python-executed Step 12 outputs. Full fresh-kernel checks remain.

- Step 13 fits the frozen selected-feature Logistic Regression (`C=1`, balanced
  weights, threshold 0.5) on 95,415 development rows and evaluates 23,795 test
  rows once. No model/threshold reselection follows test access.
- Final metrics: F1 0.750592, accuracy 0.760874, precision 0.654187, recall
  0.880321, ROC-AUC 0.875977; TN 9,543, FP 4,526, FN 1,164, TP 8,562.
- Twenty distinct errors, five subgroup dimensions, fixed probability bins and
  all 406 coefficients are published. The fitted pipeline is saved under models/.
- All 53 tests pass. Notebook 05 has six actually executed analysis/verification
  cells. Full fresh-Jupyter and canonical validation remain final gates.

## Step 14 completed synthesis

- All three exact approved questions are answered in report Sections 7.3–8
  and `report/step14_research_answers.md`, with source-table links.
- Question 1 distinguishes deposit/lead-time associations and non-monotonic,
  repetition-sensitive prior history from causal effects or global importance.
- Question 2 reports the frozen held-out metrics, confusion counts, subgroup
  weaknesses, probability overprediction and period-specific scope.
- Question 3 traces the selection/PCA, model-family and tuning comparisons to
  selected-feature balanced Logistic Regression as best evaluated by F1.
- No new fitting, predictions, calibration, threshold choice or test-driven
  model change occurs. The original document and all non-documentation
  artifacts remain unchanged from the verified Step 13 checkpoint.

## Resume Step 15

The report is now assembled as report/report.md and a visually reviewed 10-page
report/report.pdf. All five notebooks pass canonical format validation. The
installed Python 3.12 environment passes pip check, and the 58-test suite passes.
Full fresh-kernel execution is NOT certified: kernel startup is blocked by host
permissions. The user approved report publication with that check pending.
Do not retry the restricted host workflow or claim it succeeded. The local
runner and exact next actions are in FINAL_CHECKS.md. Source metadata/CC BY 4.0
are verified; raw CSV publication and member contributions remain open.

The user-approved test-comparison supplement is complete: baseline F1 0,
Logistic Regression F1 0.750591742, Random Forest F1 0.723115162. Forest uses
the development-selected `rf_12` settings, not newly tuned settings. The
selection and all Step 13 model/output hashes remain unchanged. Original
download date and acquired version are confirmed unknown; do not invent them.
Notebook 05 includes an executed supplementary evidence check (ordinary Python
execution, not a fresh Jupyter-kernel certification). The current 58-test suite
passes. See `data/results/step15/README.md`. No Step 16 is planned; submission
follows completion of the remaining gates below.

1. Read report Sections 1–8 and the Step 14 answers. Preserve the exact
   problem/questions and the frozen model, predictions and split.
2. Retain unknown original date/version and the verified source licence,
   and add the unchanged sub-50-MB original CSV under `data/raw/`.
3. Dependencies and canonical format are checked. Run all five notebooks
   top-to-bottom in fresh Jupyter kernels locally with outputs saved.
   Reproduction checks do not authorize a new model choice from test results.
4. The 168-word summary and 10-page report/PDF are assembled and visually checked.
   Rebuild and recheck the PDF after member contributions or verification status change.
5. Have Faraaz review Sections 1–3 and Sadat review Sections 4–8/assembly.
   Both record genuine contributions, commit links and AI assistance; do not
   claim assigned work was personally performed without member confirmation.
6. Verify the public repository contents and reproducibility instructions,
   then submit only its link once all gates are actually complete.

The final result is period-specific. Do not switch model families, optimize
thresholds or calibrate from the held-out outcome. Report associations and
coefficients cautiously and preserve the frozen evidence.

## Files to open

- [Executed Notebook 02](notebooks/02_preprocessing.ipynb)
- [Notebook 01: raw audit and development EDA](notebooks/01_data_audit_and_eda.ipynb)
- [Step 8 tables, figures, and reproduction](data/eda/README.md)
- [Report Sections 1–8 draft and question answers](report/report.md)
- [Step 14 research-question synthesis and evidence links](report/step14_research_answers.md)
- [Notebook 03: derived features](notebooks/03_feature_engineering.ipynb)
- [Step 9 report](report/step9_feature_engineering.md)
- [Step 9 evidence and schemas](data/processed/step9/README.md)
- [Step 10 report and feature decision](report/step10_selection_and_reduction.md)
- [Step 10 comparisons, rankings and PCA evidence](data/processed/step10/README.md)
- [Notebook 04: model comparison](notebooks/04_modeling_and_tuning.ipynb)
- [Step 11 comparison report](report/step11_model_comparison.md)
- [Step 11 results and schemas](data/results/step11/README.md)
- [Step 12 tuning report](report/step12_hyperparameter_tuning.md)
- [Step 12 evidence and frozen settings](data/results/step12/README.md)
- [Tuning and unfitted handoff factory](src/tuning.py)
- [Notebook 05: final evaluation](notebooks/05_evaluation_and_error_analysis.ipynb)
- [Step 13 evaluation report](report/step13_final_evaluation.md)
- [Step 13 metrics, predictions and errors](data/results/step13/README.md)
- [Final evaluation implementation](src/final_evaluation.py)
- [Fresh model-pipeline factory](src/modeling.py)
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
python -m src.model_comparison
python -m src.tuning
python -m src.final_evaluation
python -m unittest discover -s tests -v
```

These commands use committed processed files. Running all of Notebook 02
requires the original `data/raw/hotel_bookings.csv` for its Step 5 cells.

## Remaining reproducibility/submission items

- Add the original CSV under `data/raw/` (it is below the faculty's 50 MB limit).
  Its exact SHA-256, source and verified CC BY 4.0 terms are in data/README.md.
  Original acquisition date/version are unknown; do not invent missing details.
- Dependency consistency and canonical notebook validation pass. Complete the
  separate fresh-Jupyter-kernel check locally using FINAL_CHECKS.md; existing
  Python/IPython checks do not establish that gate. Host startup is restricted.
- Keep notebooks numbered 01–05 with saved outputs and relative paths.
- Preserve frozen data/split hashes and the original CSE437 document.
- The report/report.md and 10-page report/report.pdf are assembled and visually
  checked. Update the contribution/status records and regenerate/review the PDF
  only after the relevant member confirmations and local checks are complete.
- Record genuine member contributions, verified references, and AI assistance.
- Submit the one public GitHub repository link when the full project is ready.

## Suggested message to resume

“Continue my CSE437 Group 15 project from Step 15. Read PROJECT_STATUS.md and
HANDOFF_TO_SADAT.md in https://github.com/faraaz1027-cloud/cse437-hotel-cancellation-15
first. Sadat leads Step 15 with both members reviewing; preserve the original
proposal and frozen evaluation results. Finish and verify the remaining submission gates.”
