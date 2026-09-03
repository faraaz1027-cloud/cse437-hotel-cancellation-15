# Dataset information

The original hotel CSV is publicly committed in `data/raw/hotel_bookings.csv`. Its Git blob matches the checksum-verified source. Derived data and supporting evidence are stored under `data/processed/`.

## Source and acquisition

Source: [Hotel Booking Demand by Jesse Mostipak on Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand).

Download the dataset, extract `hotel_bookings.csv`, and place the untouched file at `data/raw/hotel_bookings.csv`. Keep original source files unchanged. For a new download, record its date/version separately: these do not establish when the original supplied file was acquired. The original acquisition date and acquired source version are unknown, as confirmed by the user. Verify exact licence terms before raw-data publication.

## Verified file record

| Field | Value |
| --- | --- |
| Original filename | hotel_bookings.csv |
| Rows / columns | 119,390 / 32 |
| Actual size | 16,855,599 bytes (16.86 MB; 16.07 MiB) |
| SHA-256 | `7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06` |
| Observed arrival period | 2015-07-01 to 2017-08-31 |
| City / resort bookings | 79,330 / 40,060 |
| Target | is_canceled |
| Not canceled / canceled | 75,166 / 44,224 |
| Original acquired source version | Unknown — user confirmed |
| Original download date | Unknown — user confirmed |
| Kaggle dataset licence | CC BY 4.0; public metadata verified 2026-09-02 |
| Current public metadata version | 1; distinct from the unknown original acquired version |

The file is below 50 MB and is included in a normal repository clone. The source link above is an alternative acquisition route; verify the checksum before using a new download.

## Attribution and licence

The [Kaggle public metadata](https://www.kaggle.com/api/v1/datasets/view/jessemostipak/hotel-booking-demand)
lists Attribution 4.0 International (CC BY 4.0). See the
[licence](https://creativecommons.org/licenses/by/4.0/) and the compact
[verified provenance record](processed/source_provenance.json). Credit Nuno Antonio,
Ana de Almeida and Luis Nunes for the original publication, Jesse Mostipak for
the Kaggle distribution, and the acknowledged prior preparation by Thomas Mock
and Antoine Bichat for TidyTuesday. No endorsement is implied.

Raw bytes remain unchanged. Project-derived data use the separately documented
eligibility, leakage exclusions, grouping and training-fitted transformations.
Retain these attribution and change notices when sharing. The current public
version/date must not be substituted for unknown original acquisition facts.

## Audit evidence

- [Human-readable audit](README.md)
- [Executed audit notebook](../notebooks/01_data_audit_and_eda.ipynb)
- [Machine-readable audit record](processed/audit_summary.json)
- [Quality figure](../figures/01_data_quality_audit.png)

The original audit is unchanged. eligibility now produces a separate cohort of **119,210 rows**, removes the two reservation-status fields from predictors, and exports 29 candidate predictors separately from the target and metadata. No retained values are imputed or otherwise changed. See [processed outputs](README.md), [eligibility summary](processed/eligibility_summary.json), and [the decision report](README.md).

## Source documentation

Antonio, N., de Almeida, A., and Nunes, L. (2019). *Hotel booking demand datasets*. Data in Brief, 22, 41-49. https://doi.org/10.1016/j.dib.2018.11.126

The publication describes extraction from hotel Property Management System SQL databases. It explains agency/company NULL semantics and observation timing. See https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/ . Document any differences between the publication and the combined Kaggle CSV.

## Handling rules

- Preserve original source bytes.
- Use repository-relative paths.
- Document unknown versus not-applicable category handling.
- Exclude both reservation-status fields from model inputs.
- Prevent overlap between retained duplicate groups in evaluation partitions.
- Fit learned preprocessing only inside training folds.
- Only the hotel dataset is used. Unrelated candidate-dataset files have been removed from the current tree and remain recoverable from Git history.


## Detailed methods and historical evidence

The following consolidated method records describe their original analyses. Historical execution and sign-off statements are not current submission certification. Current status is in the main README. Paths and administrative labels have been normalized; the original commit history remains available.

## Reproducibility and submission requirements

The workflow repair is implemented. The supplied execution verification run completed all five notebooks in fresh kernels with zero errors and passed 70 tests. Execution passed with numerical reproduction differences, not an unqualified reproduction pass. See [verification review](README.md).

### Diagnosed execution issue and repair

The original host could not start a separate Jupyter kernel. A subsequent local audit passed dependency consistency and 58 unit tests, and completed notebooks 01–04. Notebook 05 failed in the supplementary comparison with:

```text
ValueError: Frozen tuning model settings changed.
```

Notebook 04 regenerated tuning evidence and selected C=0.1, while notebook 05 expected the original C=1 model. Notebook 04 now creates a separate external development workspace. Its new comparison report records changed scores/settings without overwriting the original selection. Notebook 05 requires cached final-test evidence and retains the original settings/hash checks; missing evidence cannot trigger retraining.

The numerical cause of the earlier score differences remains unconfirmed. The runner captures environment/build details for further investigation. See [repair notes](README.md).

Retain the failed verification record, rerun tuning tables, selected configuration, Python/package versions and any locally modified runner. Diagnose the mismatch separately. Unit tests and format checks are not substitutes for end-to-end execution.

### Source data

Place the untouched `hotel_bookings.csv` at `data/raw/hotel_bookings.csv`. Expected SHA-256:

`7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06`

Raw-data publication is reported complete by the group; its independent recheck is deferred. Preserve [source attribution](README.md) and CC BY 4.0 terms. The original acquisition date/version are unknown. Keep the original private CSE437 DOCX unchanged.

### Verification commands

Use Python 3.12 and a new clone of the repair commit. Do not pull over an existing working directory or failed-run evidence. Committed bytes must be preserved; this repository supplies a `.gitattributes` rule for that purpose.

```bash
python -m pip install -r requirements.txt
python -m pip check
python -m unittest discover -s tests -q
python src/tools/verify_notebooks.py
```

The runner creates a temporary copy and executes notebooks 01–05 in fresh kernels. It also saves fresh tuning evidence separately. Save the printed verification directory, including `verification.json`, `reproduction_comparison.json`, `pip_freeze.txt`, `development_run/` and executed notebooks. If execution fails, preserve `traceback.txt` and the partial notebook outputs.

Interpret the result explicitly:

| Exit code | Meaning |
| --- | --- |
| 0 | All five notebooks executed and the recorded tuning comparison matched; not a claim of full-pipeline numerical reproduction or submission readiness. |
| 2 | All five notebooks executed, but development evidence differs. `status` is `passed_with_reproduction_differences`; review is still required. |
| 1 | Execution or integrity failed; retain diagnostics and do not mark verification complete. |

Require five completed notebooks, zero cell errors, `original_repository_unchanged: true` and `frozen_evidence_unchanged: true`. These conditions are present in the reviewed execution verification bundle. A changed winner must be reported, never promoted to the original test evaluation. Do not change tolerances, settings or assertions merely to obtain exit code 0. The supplied archive lacks exact checkout commit provenance; a final exact-commit check remains pending. The numerical warning is documented, not resolved.

`data/processed/verification/historical_verification.json` is a historical snapshot. Its earlier hashes and execution status do not certify files edited during cleanup.

### Report and authorship

- Assigned responsibilities are confirmed by the group and recorded in the report; verify attributable commits and complete joint review of the whole project.
- Review and finalize the author-supplied provisional declaration before submission, ensuring it accurately covers the assistance used.
- Rebuild the PDF after report changes with `python src/tools/build_report_pdf.py`; inspect every page and retain the 10-page limit.
- Preserve late-comparison timing, source limitations and unresolved checks.
- Confirm that raw data, five executed notebooks, code, dependencies, figures, saved model and both report formats are publicly accessible.

Submit one public repository link through the faculty's designated channel only after these requirements are satisfied:

https://github.com/faraaz1027-cloud/cse437-hotel-cancellation-15


---

## descriptive analysis — Development-data EDA

All tables use the frozen **95,415 development bookings**, covering 2015-07-01
through 2017-04-22. They contain 34,473 cancellations: **36.13%**. No test
relationships or model results are included. Source/split hashes and artifact
hashes are in `eda_summary.json`.

| Table | Contents |
| --- | --- |
| `numeric_descriptive.csv` | Valid/missing counts, mean, SD, quartiles, p95/p99, and range for the 15 initial numeric fields |
| `missingness.csv` | Original development missing counts for the 29 candidate columns |
| `categorical_descriptive.csv` | Category counts and most-frequent values after fixed domain rules |
| `lead_time_rates.csv` | Cancellation numerator, denominator, and percentage for six fixed lead-time bins |
| `deposit_rates.csv` | Rates for the three deposit types |
| `prior_cancellation_rates.csv` | Rates for 0, 1, 2–3, and 4+ previous cancellations |
| `hotel_rates.csv` | City/resort rates |
| `*_by_hotel.csv` | Lead-time, deposit, and prior-cancellation rates stratified by hotel |
| `monthly_rates.csv` | Monthly development rates/counts; April 2017 is partial through day 22 |
| `sensitivity_*.csv` | Alternative rates giving each duplicate-profile group total weight one |

Fixed preprocessing rules mark negative ADR missing and preserve unknown/no-agent
categories. Numeric EDA excludes missing values field by field rather than
fitting an imputer. Company ID and the three timing-risk fields are excluded
from the initial analyzed schema. No predictive feature is added or selected.

Main rates weight every booking equally. Sensitivity rates instead weight each
eligibility duplicate-profile group equally, retaining the mean outcome when labels
conflict. This changes the descriptive estimand, not the dataset, frozen split,
or future model weights. Do not infer independent observations or causal effects.
Group sizes matter, particularly refundable bookings and high prior counts.

Reproduce from the repository root:

```bash
python -m src.development_eda
python -m unittest discover -s tests -v
```

The module uses committed processed files; the original raw CSV is not needed
for descriptive analysis alone. Full Notebook 01 also includes the older raw audit and needs
the original CSV for those cells. The current session lacked IPython/nbformat:
five new EDA cells ran with Python and actual display outputs were captured;
the earlier ten executed audit outputs were preserved. A full fresh Jupyter
kernel run and canonical notebook validation remain pending.

See [Notebook 01](../notebooks/01_data_audit_and_eda.ipynb),
[report Sections 1–3](../report/report.md).


---

## eligibility outputs

eligibility and preprocessing are complete. The original eligibility files below are preserved. Frozen splits are in [../splits/](README.md); preprocessing evidence and reuse instructions are in [preprocessing/](README.md).

**Publication status:** all four verified row-level data files are committed
in this repository, with the aggregate summary, hashes, code, and notebook
outputs. Use these files with the fixed validation assignments, or use the command
below to regenerate them from the original CSV.

These files contain the eligible cohort **before** imputation, encoding,
scaling, or feature selection. validation partition membership is stored separately
in `data/processed/splits/`; these eligibility files retain their original row order. They are not a final
model-ready dataset. Gzip is lossless compression; pandas reads it directly.

| File | Contents |
| --- | --- |
| `eligibility_candidates.csv.gz` | 119,210 rows, 29 original candidate predictors; no target or reservation statuses |
| `eligibility_target.csv.gz` | Corresponding `is_canceled` target in exactly the same row order |
| `eligibility_metadata.csv.gz` | Row lineage, arrival dates, duplicate groups, and quality flags; never model inputs |
| `eligibility_exclusions.csv` | Original row positions and reason for the 180 excluded zero-guest records |
| `eligibility_summary.json` | Measured counts, versions, and hashes for all four data files |

`source_row_id` is a zero-based original record position after the header, not
a real booking ID. Positional alignment across candidates, target, and metadata
is mandatory. Do not sort, drop, or shuffle one file independently.

From the repository root, regenerate with:

```bash
python -m src.eligibility
```

Or run `notebooks/02_preprocessing.ipynb` from top to bottom. Both require the
unchanged original `data/raw/hotel_bookings.csv`; see `../README.md`.

Load the committed compressed outputs with:

```python
from pathlib import Path
import pandas as pd

data_dir = Path('data/processed')  # from the repository root
X = pd.read_csv(data_dir / 'eligibility_candidates.csv.gz')
y = pd.read_csv(data_dir / 'eligibility_target.csv.gz')['is_canceled']
metadata = pd.read_csv(data_dir / 'eligibility_metadata.csv.gz')
assert len(X) == len(y) == len(metadata)
```

The original eligibility candidate file retains its negative ADR, missing values,
and company codes. preprocessing applies documented domain rules and training-fitted
preprocessing through the pipeline without modifying this source copy. Its
initial schema uses 25 source fields; the three post-booking-timing fields and
company ID are excluded. Candidate availability at booking creation is still
not established by these retrospective source snapshots.

Duplicate groups are defined by the 29 candidate columns, excluding target
and outcome-status fields, with missing values grouped consistently. A group's
members all have the same arrival date. validation verifies group separation at
every split and preserves whole calendar dates. Do not use group IDs,
row IDs, dates from metadata, or review flags as predictors by accident.

Decisions and limitations: [eligibility report](README.md).


---

## representation comparison selection and reduction evidence

| File | Contents |
| --- | --- |
| `comparison_protocol.json` | Choices fixed before local comparison: four modes, 75% F ranking, 95% numeric PCA, fixed reference classifier and decision rule |
| `fold_results.csv` | Twelve fits: metrics, widths, components, numeric variance retained, convergence iterations and measured runtime |
| `representation_comparison.csv` | Mean metrics and fold F1 SD for each mode |
| `feature_rankings.csv` | Every encoded field's training variance, F-score ranking and selection flag per fold; no inferential p-values |
| `representation_schemas.json` | Complete selected/output names plus PCA inputs, coefficients, means and explained variances per fold/mode |
| `representation_summary.json` | Current preferred mode, scope, checksums, runtime and limitations |

The current preference is `selected` (mean reference-model validation F1
0.713609): 247/314/366 output features across the three frozen folds. Numeric
PCA and combined selection/PCA were demonstrated but are not the preferred
representation because their mean F1 was lower.

Inputs are the committed eligibility files and frozen assignments. Only development
labels are used: training labels for supervised selection/reference fitting,
validation labels for comparison. The held-out test is not transformed or scored.
Each fold learns its own vocabularies, scores, mask and components. Do not treat
these diagnostics as a fitted full-development model or an unbiased final score.

Run from the repository root:

```bash
python -m src.representation_audit
python -m unittest discover -s tests -v
```

Runtimes can change across executions; checksums in the summary correspond to
the evidence generated by that execution. The source and frozen split checksums
must remain unchanged. The comparison protocol rejects unversioned changes.

For subsequent CV, use `BookingRepresentation(mode="selected")` inside a model
Pipeline; retain `mode="full"` as a control. PCA requires scaled numeric inputs;
full/selected tree variants can omit scaling. All 35 tests pass. Fresh Jupyter-
kernel notebook execution and canonical format validation remain final gates.

See [the representation comparison report](README.md) and
[PCA variance figure](../figures/06_numeric_pca_variance.png).


---

## preprocessing — Preprocessing evidence

| File | Purpose |
| --- | --- |
| `preprocessing_summary.json` | Per-fold shapes, missingness before/after, domain decisions, medians, unseen-category counts, runtime versions, and source hashes |
| `feature_schemas.json` | Ordered output names for each training-fold encoder |

The diagnostic matrices were produced separately for each development
training/validation pair. No additional rows were removed and no test rows
were fitted or transformed. No predictive estimator or global preprocessor was
trained. Use the reusable factory inside later model pipelines; do not prefit
one transformer on all development rows before CV.

### Measured output dimensions

| Fold | Training rows | Validation rows | Encoded columns | Nonfinite values after preprocessing |
| --- | ---: | ---: | ---: | ---: |
| 1 | 23,797 | 23,893 | 328 | 0 |
| 2 | 47,690 | 23,776 | 422 | 0 |
| 3 | 71,466 | 23,949 | 491 | 0 |

The 29 eligibility source columns become 25 initial source fields after excluding
`company`, `assigned_room_type`, `booking_changes`, and `days_in_waiting_list`.
These 25 fields comprise 15 numeric fields and 10 categorical fields. Two
fixed missing indicators cover children and ADR. One-hot vocabularies vary
across folds because only training categories are learned. Matrix columns must
not be pooled by position between folds; use the matching schema.

### Reproduce

From the repository root using the committed eligibility/6 files:

```bash
python -m src.preprocessing_audit
python -m unittest discover -s tests -v
```

The complete Notebook 02 also runs eligibility and validation and therefore needs the original
CSV for its earlier cells. A separate fresh-kernel Jupyter run remains a final
submission verification task. The executed environment is recorded in the
summary; `requirements.txt` retains compatible starter ranges, not a certified
clean-install lockfile.

### Reuse in later modeling

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from src.preprocessing import make_preprocessor
from src.splitting import development_cv

pipeline = Pipeline([
    ('preprocess', make_preprocessor()),
    ('model', LogisticRegression(max_iter=2000, random_state=42)),
])
search = GridSearchCV(pipeline, {'model__C': [0.1, 1.0, 10.0]},
                      scoring='f1', cv=development_cv(assignments))
## Run in model comparison/12 after adding the feature engineering/10 components:
## search.fit(X_dev, y_dev)
```

Construct `assignments`, `X_dev`, and `y_dev` using
[the validation loading example](README.md). Pass raw candidate
DataFrames in that exact development order. Set `scale_numeric=False` for the
tree-compatible variant. Feature-engineering extensions require an explicit
schema update in feature engineering; unknown extra columns raise an error instead of being
silently dropped. The final fitted model/pipeline will be saved in the later
modeling stage.

descriptive analysis uses development rows for descriptive statistics and relationships.
Company-presence/derived-calendar features remain feature engineering candidates. The
policy exclusions in this stage are not a substitute for statistical feature
selection and dimensionality reduction in representation comparison.

Details: [preprocessing report](README.md).


---

## feature engineering feature evidence

- `feature_summary.json`: measured development-only diagnostics, fold checks,
  input/output SHA-256 hashes, runtime, and limitations.
- `feature_schemas.json`: full encoded feature names for each frozen fold.
- `derived_feature_statistics.csv`: descriptive values of the eight fixed-formula
  features across 95,415 development rows, before learned imputation/scaling.

No target values are read by this audit and no test row is fitted/transformed.
No source rows, split assignments, or training weights change. Aggregate evidence
does not substitute for a final selected feature set or predictive comparison.

The schema has 32 fields before encoding (24 retained source fields plus eight
derived fields). Month names are replaced by a fixed sine/cosine pair. Five
fixed missing indicators follow encoding. Folds produce 332, 421 and 490 columns.
Both scaled and unscaled variants are checked, with training-only statistics.

Run from the repository root:

```bash
python -m src.feature_audit
python -m unittest discover -s tests -v
```

The commands work with committed processed files. Notebook 03 contains five
actually executed Python cells and their saved output. A separate fresh Jupyter
kernel run and canonical nbformat validation remain final checks.

Read [the feature engineering report](README.md) for formulas,
missing-value semantics, known limitations, and integration instructions. For
later CV, use `make_feature_preprocessor()` inside the estimator pipeline and
keep fold indices relative to development rows. Do not reuse globally fitted
transformations. The original preprocessing factory is preserved for comparison.


---

## model comparison model-comparison evidence

| File | Contents |
| --- | --- |
| `comparison_protocol.json` | Five candidates, fixed settings, representations, metrics, threshold and decision rule |
| `fold_results.csv` | All 15 fits: validation metrics/confusion counts, training diagnostics, runtime, capacity and membership checks |
| `model_comparison.csv` | Unweighted three-fold mean metrics and descriptive F1 SD |
| `feature_schemas.json` | Output names for each trained pipeline; baseline uses no features |
| `estimator_parameters.json` | Full actual estimator settings, including defaults |
| `model_summary.json` | Untuned leader, scope, runtime, frozen input and output hashes |

Current leader: **Logistic regression — selected**, mean validation F1 **0.713609**.
This is a development choice for further tuning, not the final test result.
Baseline plus two learned families are implemented; each learned family compares
full and selected features. All 40 tests pass; logistic metrics match representation comparison.

The audit uses only development features and labels. Every pipeline fits its
preprocessing and selection inside its own training fold. The final test is
neither transformed nor scored. Row-level predictions and globally fitted
models are not exported. Measured timings can vary between executions; output
hashes identify the corresponding saved run.

```bash
python -m src.model_comparison
python -m unittest discover -s tests -v
```

Run from the repository root using the committed processed inputs. Notebook 04
has six sequentially executed Python cells with saved output. A fresh Jupyter-
kernel run and canonical notebook validation remain final submission checks.

See [the model comparison report](README.md) and
[comparison figure](../figures/07_model_comparison.png). Next, tune both
learned model families through fresh `make_model_pipeline` instances inside CV.


---

## tuning tuning evidence

| File | Evidence |
| --- | --- |
| `search_protocol.json` | Predeclared grids, scoring, threshold, budget, tie rule and no-refit policy |
| `candidate_parameters.json` | Search parameters and complete estimator settings for all 20 candidates |
| `candidate_results.csv` | All candidate means, sample fold SD, training F1, gap and runtime |
| `fold_results.csv` | All 60 fits: five metrics, confusion counts, training F1 and fold membership |
| `logistic_regression_cv_results.csv` | Complete sklearn CV results for eight logistic settings |
| `random_forest_cv_results.csv` | Complete sklearn CV results for twelve forest settings |
| `tuning_comparison.csv` | Best setting per family versus its model comparison untuned control |
| `control_parity.csv` | Exact model comparison/control differences and numerical-check tolerances |
| `final_selection.json` | Development-selected settings and unfitted evaluation configuration |
| `tuning_summary.json` | Scope, checks, runtime, source/input/output checksums and limitations |

Selected: **Logistic Regression**, `C=1.0, class_weight=balanced`, mean development F1 **0.732102**.
Both families use selected features and threshold 0.5. All 60 fits complete;
untuned controls reproduce model comparison. These are reused-development scores, not
final test performance. No held-out rows are fitted, transformed or scored,
and no full-development refit occurs. All 47 tests pass.

Threshold-based control metrics and logistic AUC match within 1e−12; forest
AUC uses a 1e−7 numerical tolerance. The initial complete grid's strict audit
caught a tiny forest AUC difference; the unchanged grid was rerun after fixing
only that check. See the report for the execution history.

```bash
python -m src.tuning
python -m unittest discover -s tests -v
```

Run from the repository root using the committed processed inputs. Timings
and corresponding output hashes can vary between runs. Candidate SD uses
ddof=1; raw sklearn CV-table SD uses ddof=0. Both are descriptive only.
See [the detailed report](README.md)
and [Notebook 04](../notebooks/04_modeling_and_tuning.ipynb).
Its five new cells have actual Python outputs; full fresh-Jupyter and canonical
format verification remain final gates.


---

## evaluation final-evaluation evidence

The tuning selected-feature Logistic Regression pipeline (`C=1`, balanced
class weights, threshold 0.5) was fitted on 95,415 development rows and
evaluated once on 23,795 later-arrival test rows. No test result changed the
model, representation or threshold.

| File | Evidence |
| --- | --- |
| `evaluation_protocol.json` | Frozen settings, metrics, error slices and no-reselection rule |
| `evaluation_summary.json` | Selection lineage, final metrics, checks, runtime, limits and hashes |
| `final_metrics.csv` | Held-out metrics and confusion counts |
| `test_predictions_01.csv.gz` … `04` | Four ordered parts containing aligned row IDs, dates, labels, probabilities and error types |
| `subgroup_metrics.csv` | Descriptive results by hotel, lead time, deposit, market segment and customer type |
| `probability_diagnostics.csv` | Fixed 0.1-wide probability bins and observed rates |
| `error_examples.csv` | Twenty distinct false-positive/false-negative examples selected by the frozen rule |
| `feature_coefficients.csv` | All 406 selected encoded fields and fitted logistic coefficients |

Final F1 is **0.750592**, accuracy **0.760874**, precision **0.654187**,
recall **0.880321**, and ROC-AUC **0.875977**. Confusion counts are TN 9,543,
FP 4,526, FN 1,164 and TP 8,562. Brier score is 0.160007.

The saved fitted pipeline is `models/final_logistic_regression.joblib`. Figures
09–10 show the final metrics/confusion matrix and error/probability diagnostics.
All 53 tests pass. Notebook 05 contains six executed evidence-analysis cells.
Full fresh-Jupyter execution, canonical notebook validation and clean-install
verification remain verification gates.

```bash
python -m src.final_evaluation
python -m unittest discover -s tests -v
```

Regenerating the evidence with the frozen settings is a reproducibility check,
not permission to choose a new model from the test result.


---

## verification — authorized reporting-only test comparison

### Timing and fixed choices

The user approved adding a majority baseline and Random Forest test comparison
after the evaluation Logistic Regression test outcome was known. Consequently,
this is not described as a preregistered simultaneous three-model evaluation.
The supplement protocol was saved before fitting the two added estimators.

The baseline learns the development majority (class 0). Random Forest uses
tuning's development-selected `rf_12`: selected representation, 100 trees,
minimum leaf size 10, unlimited depth, balanced weights, sqrt feature sampling,
seed 42. Both fit only 95,415 frozen development rows and score the identical
23,795 held-out bookings. All learned preprocessing and feature selection fit
development data only. Threshold stays 0.5. The final Logistic Regression model
remains selected based on development F1; its saved predictions are reused.

### Results

| Model | F1 | Accuracy | Precision | Recall | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training-majority baseline | 0.000000 | 0.591259 | 0.000000 | 0.000000 | 0.500000 |
| Logistic Regression (selected) | 0.750592 | 0.760874 | 0.654187 | 0.880321 | 0.875977 |
| Random Forest | 0.723115 | 0.789325 | 0.781239 | 0.673041 | 0.878190 |

Baseline precision is zero by the `zero_division=0` convention: it predicts
no cancellations. Logistic Regression has higher F1/recall; Random Forest has
higher precision/accuracy and slightly higher AUC. These descriptive outcomes
do not trigger retuning, threshold optimization, calibration or model switching.
A single chronological test does not establish statistical significance or
generalization to other hotels/periods. Brier scores are diagnostics, not proof
of calibrated probabilities.

### Evidence and reproduction

- [Frozen protocol](processed/results/test_comparison/comparison_protocol.json)
- [Full metrics, confusion counts and Brier scores](processed/results/test_comparison/test_comparison.csv)
- [Random Forest probabilities with aligned source row IDs and actual labels](processed/results/test_comparison/random_forest_test_probabilities.csv.gz)
- [Hash lineage and runtime](processed/results/test_comparison/comparison_summary.json)
- [Implementation](../src/test_comparison.py)
- [Tests](../src/tests/test_test_comparison.py)
- [Original Logistic Regression evidence](README.md)

From the repository root, run `python -m src.test_comparison`. If the saved
supplement exists, this verifies its hashes and returns cached results without
repeating test evaluation. To independently reproduce fitting, use an isolated
copy without its verification outputs; never remove the original evidence. Requires
the same scientific runtime recorded in `comparison_summary.json` and the
committed eligibility/6/12/13 artifacts. No raw dataset changes are made.

The current suite (`python -m unittest discover -s tests -q`) passes 58 tests.
All original evaluation output hashes and the tuning selection hash were checked
before and after this run. Notebook 05's added evidence-check cell was executed
in Python with its output saved; full clean-environment/fresh-Jupyter-kernel
verification remains a separate open submission gate.

Original acquisition date and acquired dataset version are unknown. Outstanding submission requirements are listed in [data/README.md](README.md).


---

## validation — Frozen evaluation partitions

**Publication:** the verified row-level assignment file is committed here with
the code, aggregate plan, and hashes. Load it directly using the example below.
To reproduce and check the same assignments from the committed eligibility data,
run `python -m src.splitting`.

| File | Purpose |
| --- | --- |
| `validation_assignments.csv.gz` | Verified, committed assignments; one per eligibility row, preserving row order |
| `validation_split_plan.json` | Fixed dates, counts, metrics, validation protocol, and input/output hashes |

### Final split

| Partition | Arrival dates (inclusive) | Bookings |
| --- | --- | ---: |
| Development | 2015-07-01 to 2017-04-22 | 95,415 |
| Test | 2017-04-23 to 2017-08-31 | 23,795 |

The whole-date boundary minimizes the difference from an 80% development row
count; ties choose the earlier date. No label, model result, random shuffle,
or stratification determines the boundary.

### Forward validation inside development data

| Fold | Training through | Training rows | Validation period (inclusive) | Validation rows |
| --- | --- | ---: | --- | ---: |
| 1 | 2016-01-26 | 23,797 | 2016-01-27 to 2016-06-21 | 23,893 |
| 2 | 2016-06-21 | 47,690 | 2016-06-22 to 2016-11-06 | 23,776 |
| 3 | 2016-11-06 | 71,466 | 2016-11-07 to 2017-04-22 | 23,949 |

Every training window begins on 2015-07-01. Development boundaries approximate
25%, 50%, and 75% row prefixes, preserving whole dates. Window durations differ.
Future development rows are `unused` until a later fold. Test rows always have
the role `excluded_test`. Every duplicate group has one role in each fold.

### Loading and index alignment

Run this from the repository root:

```python
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from src.splitting import check_assignments, development_cv

root = Path('.')
plan = json.loads((root / 'data/processed/splits/validation_split_plan.json').read_text())
path = root / 'data/processed/splits' / plan['assignment_file']
assert hashlib.sha256(path.read_bytes()).hexdigest() == plan['assignment_sha256']
assignments = pd.read_csv(path)
check_assignments(assignments)
X = pd.read_csv(root / 'data/processed/eligibility_candidates.csv.gz')
y = pd.read_csv(root / 'data/processed/eligibility_target.csv.gz')['is_canceled']
metadata = pd.read_csv(root / 'data/processed/eligibility_metadata.csv.gz')
assert assignments.source_row_id.tolist() == metadata.source_row_id.tolist()
assert len(X) == len(y) == len(assignments)

dev_rows = np.flatnonzero(assignments.partition.eq('development'))
X_dev = X.iloc[dev_rows].reset_index(drop=True)
y_dev = y.iloc[dev_rows].reset_index(drop=True)
cv = development_cv(assignments)
## Later: GridSearchCV(pipeline, ..., scoring='f1', cv=cv).fit(X_dev, y_dev)
## Fit all learned preprocessing/selection inside pipeline and each train fold.
```

The helper returns indices **relative to the development subset**, not the full
cohort. Preserve the demonstrated row order. `cohort_row` is the zero-based
position in eligibility files; `source_row_id` identifies the original raw row.
Neither IDs, dates from metadata, fold roles, nor partition labels are model
features. The 29 candidate inputs still need preprocessing availability/preprocessing
review.

To reproduce/verify the split from the committed eligibility files, without needing
the original raw CSV:

```bash
python -m src.splitting
python -m unittest discover -s tests -v
```

Rerunning unchanged inputs preserves the frozen artifacts. Changed membership
or plan settings cause an error; any revision must be explicitly reviewed and
versioned before evaluation. Do not revise the split to improve model scores.

### Evaluation protocol

- Select by mean cancellation-class F1 across the three development folds;
  also report per-fold results, accuracy, precision, recall, and ROC-AUC.
- Compare a majority baseline, logistic regression, and random forest using
  these same folds. Use random state 42 for stochastic models/searches later.
- Default classification threshold is 0.5; any threshold tuning uses development
  validation only. Do not tune using test results.
- Fit preprocessing, feature selection, and dimensionality reduction only on
  each training fold. Refit the selected complete pipeline on all development
  rows, then evaluate the held-out test set once in evaluation.
- No model was trained and no test class distribution or score was computed in
  the split construction. Existing full-source quality audits remain disclosed.

This is retrospective generalization to later arrival cohorts. It does not
establish booking-time feature snapshots or live availability of training
labels; no temporal embargo is applied. Consider seasonality and distribution
change when interpreting the later test period.

See [validation report](README.md) and
[timeline](../figures/02_evaluation_timeline.png).


---

## Initial Hotel Booking Data Audit

**CSE437: Data Science | Group 15**

**Status:** raw-data audit completed. Cleaning, development/test assignment, predictor-outcome EDA, and modeling remain pending.

### File verified

| Check | Measured value |
| --- | --- |
| File | hotel_bookings.csv |
| Rows | 119,390 |
| Columns | 32 |
| File size | 16,855,599 bytes (16.86 MB; 16.07 MiB) |
| Arrival period | 2015-07-01 to 2017-08-31 |
| Invalid arrival dates | 0 |
| City / resort records | 79,330 / 40,060 |
| Not canceled / canceled | 75,166 / 44,224 |
| Cancellation share | 37.0416% |

SHA-256: `7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06`

These are measurements of the original file. Class proportions are not held-out model performance.

### Parsed missing values

| Column | Count | Share of records |
| --- | --- | --- |
| `company` | 112,593 | 94.31% |
| `agent` | 16,340 | 13.69% |
| `country` | 488 | 0.41% |
| `children` | 4 | 0.0034% |

Company and agent require different decisions: agent is mostly populated. Their codes are categorical identifiers. The original dataset documentation describes NULL agency/company entries as not applicable; source semantics should guide treatment rather than generic numerical imputation.

### Repeated records

- Additional exact full-row copies after the first occurrence: **31,994 (26.80%)**.
- All rows participating in repeated groups: **40,165**.
- Distinct full rows: **87,396**.

There is no unique booking identifier in this file. Identical rows may reflect repeated extracts or indistinguishable separate bookings. No duplicates have been removed. Document the retain/remove policy and keep identical retained groups out of opposite validation partitions.

### Guest and stay checks

- Known zero-total-guest records: **180**.
- Unknown guest totals due to missing counts: **4**.
- Zero-adult records: **403**, including **223 with a known positive total**.
- Negative guest-count records: **0**.
- Zero-night records: **715**.
- Records flagged as both zero guests and zero nights: **70**.

Total guests were calculated only when all three guest counts were present. Do not replace unknown counts with zero merely to run this check. The flags overlap and should not be summed as independent exclusions.

### ADR checks

| Measure | Value |
| --- | --- |
| Minimum / median / maximum | -6.38 / 94.575 / 5,400 |
| Next largest value after 5,400 | 510 |
| Negative ADR records | 1 |
| Zero ADR records | 1,959 |
| ADR greater than 1,000 | 1 |
| Q1 / Q3 | 69.29 / 126.00 |
| Full-file diagnostic 1.5-IQR fences | -15.775 to 211.065 |
| Records outside those fences | 3,793 |

The negative ADR is **not** outside the IQR fence, so statistical flags alone miss this domain issue. The 5,400 observation deserves inspection, but an unusual value is not automatically an error. Zero rates can have valid operational explanations.

The full-file fences above are descriptive audit results, not thresholds to reuse in modeling. Any clipping, imputation, scaling, selection, or reduction fitted on data must be learned inside training folds.

### Direct outcome leakage

| reservation_status | is_canceled = 0 | is_canceled = 1 |
| --- | --- | --- |
| Check-Out | 75,166 | 0 |
| Canceled | 0 | 43,017 |
| No-Show | 0 | 1,207 |

Reservation status reproduces all observed labels. Exclude `reservation_status` and `reservation_status_date` from predictor inputs. Removing the target and these two columns leaves **29 candidate predictors**, still subject to availability review.

The original publication describes observation timing relative to the day before arrival. This does not establish that every stored value was known when the initial booking was made.

### Other categorical flags

Explicit `Undefined` values occur in market_segment (2), distribution_channel (5), and meal (1,169). Read the field definitions before merging or treating these as unknown categories.

### Recommended next decisions

| Issue | Starting approach to evaluate |
| --- | --- |
| Agent | Retain categorical information and an explicit no-agent category where the source meaning supports it. |
| Company | Compare categorical retention against a company-booking indicator; do not treat code magnitude as meaningful. |
| Country / children | Document unknown-value handling and fit any imputation on training folds. |
| Zero total guests | Investigate and justify eligibility for a guest-booking prediction population. |
| Duplicate groups | Establish a documented policy and prevent retained identical groups crossing partitions. |
| ADR | Investigate suspicious values; compare justified training-only treatments rather than deleting all statistical outliers. |
| Zero nights | Review meaning separately from zero guests. |
| Outcome columns | Exclude both status fields and audit timing of remaining predictors. |

No cleaning decision has yet been applied to the source.

### Reproduction and files

1. Obtain the source CSV and place it at `data/raw/hotel_bookings.csv`.
2. Use the repository setup instructions and run `notebooks/01_data_audit_and_eda.ipynb` from top to bottom.
3. The notebook writes `data/processed/audit_summary.json` and `figures/01_data_quality_audit.png`.

The 10 code cells were executed sequentially in a fresh Python process with an IPython shell, and actual outputs were saved. A separate Jupyter kernel could not launch in this environment; a normal fresh-kernel Jupyter run remains to be verified before submission. The source checksum was unchanged after execution.

The NYC Airbnb CSV and map belong to a different dataset. They are not inputs to this audit.

### Sources

- Source named in the project: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
- Original publication: Antonio, N., de Almeida, A., and Nunes, L. (2019). Hotel booking demand datasets. Data in Brief, 22, 41-49. https://doi.org/10.1016/j.dib.2018.11.126
- Data definitions: https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/
- Exact licence/terms for the downloaded Kaggle version: still to be recorded.


---

## Development rerun and frozen-evaluation separation

### Diagnosed failure

The earlier Windows verification completed notebooks 01–04, then failed notebook 05's frozen-selection check. Fresh tuning selected balanced Logistic Regression with C=0.1; the published final model uses C=1. Notebook 04 had written the new selection to the same location used by the historical evaluation.

The supplied evidence records mean development F1 changing from 0.7321017246390454 to 0.7313708930914117 for C=1, balanced. C=0.1, balanced remained at 0.7316967924435923 and became the rerun winner. Other C=1 and C=10 candidates also changed. This is not merely display rounding. CPU/numerical-library differences are a hypothesis, not an established cause.

### Implemented boundary

- Notebook 04 copies development inputs and relevant source files into a new external workspace. New model-comparison and tuning evidence stays there. Final-test results and saved models are not copied into that workspace.
- The final comparison cell reports selection, candidate/fold metric, protocol, recorded input/source hash and environment differences. It never promotes a new winner to the published final evaluation.
- Notebook 05 requires the existing comparison cache. The original C=1 requirement, exact selection hash, cached output hashes and model checks remain intact. Missing artifacts raise an error rather than triggering a new fit.
- The verification runner preserves the original repository, checks frozen evidence after each notebook, captures failures/partial outputs, and distinguishes execution success from tuning-evidence agreement. New development tables and diagnostic files are collected separately.
- Windows path keys are normalized for the representation comparison reference lookup without changing stored digests or numerical tolerances. Git attributes preserve committed bytes across platforms. Run-local serialization differences are reported separately from semantic protocol changes.

### Interpretation and remaining checks

Existing published notebook outputs were retained as historical reference; newly added code has no fabricated outputs. Unit regression tests exercise changed winners, independent copies, preserved evidence, missing caches, path/newline differences and status reporting. Stubbed-kernel tests check orchestration only, not actual notebook execution.

The supplied execution verification Windows run completed all five notebooks in fresh kernels with zero errors and passed 70 tests. The cached final comparison verified original evidence without training. Development scores and the winner still differed. This does not establish the cause of numerical drift, guarantee identical optimization results, refit the final model, or certify submission readiness. The archive does not establish the complete checkout's exact commit; see the [verification review](README.md).

Use the commands and exit-code interpretation in [data/README.md](README.md). In particular, `passed_with_reproduction_differences` means execution succeeded but the scientific differences still need review. It is not an unqualified pass.

report alignment aligns the Markdown report and PDF with this reviewed run and user-confirmed contributions. The historical verification snapshot remains unchanged. Final declaration review, raw-publication recheck and joint submission sign-off remain separate tasks.


---

## representation comparison — Feature selection and dimensionality reduction

Selection alone is the current preferred representation for the fixed logistic-
regression reference: mean development F1 is **0.713609**, versus **0.693094**
with all feature engineering features. Numeric PCA and selection followed by PCA were both
implemented and evaluated, but scored lower than selection alone. PCA is kept
as a documented experiment rather than included in the preferred pipeline.

### Fixed comparison protocol

Before the comparison, `comparison_protocol.json` fixes four modes: all feature engineering
features (`full`), supervised selection (`selected`), numeric PCA (`pca`), and
selection followed by numeric PCA (`selected_pca`). No feature/target definitions,
cohort membership, duplicate weights, or frozen splits change.

Selection removes columns with training variance at most 1e−12, ranks remaining
encoded features by ANOVA F against training cancellation labels, and retains
the top 75% rounded upward. Ties follow original encoded order. F statistics are
ranking heuristics; p-values are discarded because independent-row inference is
not justified. A category's one-hot column may be retained while another category
of the same source field is dropped. This is selection of encoded features, not
necessarily entire source variables.

PCA uses centered full SVD on standardized numeric inputs only, keeping the
smallest component count whose cumulative training variance exceeds the 95%
target. Categories and missing indicators stay sparse and bypass PCA. In the
combined mode, PCA operates on the numeric features retained by selection.
The largest dense numeric training block is **13,149,744 bytes (13.15 MB)**;
the full sparse categorical matrix is never densified.

All four modes use the same logistic-regression reference: C=1, lbfgs,
max_iter=2000, tol=1e−4, no class weights, random_state=42, and probability
threshold 0.5. Twelve fits compare four modes on the three frozen forward folds.
Every fit converged without a ConvergenceWarning (95–182 iterations). The mean
cancellation-class F1 is the prespecified primary comparison metric. Exact mean
ties favor fewer average output columns, then declared mode order.

This reference classifier makes the feature decision reviewable; model comparison still
must evaluate the majority baseline and two model families, and tuning still
must report model hyperparameter tuning.

### Results and current choice

| Representation | Fold 1 F1 | Fold 2 F1 | Fold 3 F1 | Mean F1 | Output columns by fold |
| --- | ---: | ---: | ---: | ---: | --- |
| All feature engineering features | 0.643231 | 0.702455 | 0.733595 | 0.693094 | 332 / 421 / 490 |
| Selection only | 0.693691 | 0.714117 | 0.733020 | **0.713609** | 247 / 314 / 366 |
| Numeric PCA | 0.692187 | 0.662830 | 0.728881 | 0.694633 | 325 / 413 / 483 |
| Selection then PCA | 0.692859 | 0.680961 | 0.729248 | 0.701023 | 242 / 308 / 360 |

Selection improves mean F1 by **0.020515** over the all-feature reference while
reducing encoded width. The improvement is not uniform: fold 3 is slightly
worse. Mean precision rises from 0.681025 to 0.783364 while mean recall falls
from 0.753386 to 0.659498, so the preference reflects the chosen F1 metric and
does not mean improvement on every objective. Secondary metrics and fold SDs
are in the comparison CSV; SD across three temporal folds is descriptive, not
a confidence interval.

The current feature rule is therefore **75% training F-score selection without
PCA**, refitted inside every later model-training fold. Complete retained names
are saved per fold. A global full-development selection mask is not fitted now;
that refit belongs after all modeling choices are frozen.

Third-training-fold top-ranked features include non-refundable/no-deposit
indicators, country PRT, prior cancellation share, lead time, prior cancellations,
history presence, and special requests. These rankings are associations, not
causal effects or final model-importance findings. See the complete rankings
for all encoded fields and the retained/discarded flags.

### PCA evidence

| Fold | Numeric PCA inputs → components | Retained numeric variance | Selected numeric inputs → components | Retained selected numeric variance |
| --- | --- | ---: | --- | ---: |
| 1 | 23 → 16 | 95.8830% | 20 → 15 | 96.4056% |
| 2 | 23 → 15 | 95.0011% | 21 → 15 | 96.4178% |
| 3 | 23 → 16 | 95.6934% | 21 → 15 | 96.4710% |

![Training variance retained by numeric PCA](../figures/06_numeric_pca_variance.png)

PCA addresses correlated numeric totals/components but preserves variance, not
necessarily cancellation signal. The threshold applies to the numeric input
block, not the entire encoded dataset or predictive information. Complete
component coefficients, centering means, input names and explained variances
are saved in `representation_schemas.json`.

### Validation and limitations

Preprocessing, supervised ranking and PCA learn only from each training prefix.
Validation transformations preserve fitted state; validation labels are used
only to score reference predictions and compare representations. No held-out
feature/target distribution, transformation or prediction is computed. No rows
are removed and no final full-development model is fitted.

All **35 tests** pass, including six new tests for known-signal selection,
constant removal, tie handling, numeric-only centered PCA, preserved categories,
variance coverage, training-state isolation, invalid inputs and cloneable pipeline
integration. Notebook 03 retains five executed feature engineering cells and appends five
actually executed representation comparison Python cells. The saved JSON passes structural checks;
canonical nbformat validation and fresh Jupyter-kernel execution remain final
submission gates because those packages are unavailable in this runtime.

The four-way comparison uses the same development folds to choose a version,
so its scores are selection estimates rather than unbiased final performance.
Univariate scores may miss nonlinear interactions or rare categories. The
preferred representation may differ for random forest; model comparison should keep an
all-feature control. Source timing, repeated bookings and temporal drift remain
limitations from earlier stages.

### Reproduce and hand off

```bash
python -m src.representation_audit
python -m unittest discover -s tests -v
```

For later modeling, place `BookingRepresentation(mode="selected")` inside the
estimator Pipeline passed to the unchanged development CV. It creates fresh
preprocessing, selection and optional PCA internally. Tree models can request
`scale_numeric=False` for full/selected modes; PCA modes require scaling.
Do not use a precomputed globally fitted matrix. Preserve the development-row
order expected by `development_cv`.


---

## model comparison — Baseline and two model families

The leading untuned candidate is **Logistic regression — selected**, with mean development
cancellation F1 **0.713609**. This is a starting point for tuning,
not the final best-model or test result. Five candidates were fitted on the same
three frozen forward folds, giving 15 model fits.

### Models, settings and validation

The majority baseline predicts the class most frequent in each training fold
and ignores predictors. Logistic regression provides a regularized linear
reference; random forest supplies a distinct nonlinear tree-ensemble family.
Both learned families compare full feature engineering features with representation comparison's selected
features (75% of nonconstant training columns by ANOVA F ranking).

- Logistic regression: C=1, lbfgs, max_iter=2000, tol=1e−4, no class weights,
  random_state=42; numeric fields are scaled inside training folds.
- Random forest: 100 trees, Gini criterion, unlimited depth, min_samples_split=2,
  min_samples_leaf=1, max_features=sqrt, bootstrap=True, no class weights,
  random_state=42, n_jobs=2; the verified unscaled numeric variant is used.
- Threshold: cancellation probability ≥0.5 predicts class 1 for every model.
  No resampling, threshold optimization or model-parameter search occurs.

The protocol is written before fitting. Each candidate receives a new pipeline
whose preprocessing and selection fit only on its training prefix. All
candidates use identical validation membership in each fold. No validation
labels fit a selector and no held-out row is fitted, transformed or scored.

The primary metric is the unweighted mean cancellation-class F1 across three
folds. Accuracy, precision, recall and ROC-AUC are secondary. Fold standard
deviations are descriptive, not confidence intervals. Logistic results match
the previously published representation comparison full/selected metrics within 1e−12.

### Measured development results

| Candidate | Mean F1 | Accuracy | Precision | Recall | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Majority baseline | 0.000000 | 0.638167 | 0.000000 | 0.000000 | 0.500000 |
| Logistic regression — full | 0.693094 | 0.751370 | 0.681025 | 0.753386 | 0.876098 |
| Logistic regression — selected | 0.713609 | 0.809314 | 0.783364 | 0.659498 | 0.882905 |
| Random forest — full | 0.626504 | 0.793498 | 0.906662 | 0.481119 | 0.886473 |
| Random forest — selected | 0.657994 | 0.803866 | 0.893717 | 0.521714 | 0.892267 |

![Untuned model comparison](../figures/07_model_comparison.png)

The majority classifier predicts no cancellations in all three folds. Its mean
accuracy is **63.82%**, but cancellation precision/recall/F1 are
zero and ROC-AUC is 0.5. Accuracy alone would conceal that failure.

For logistic regression, selection changes mean F1 by **+0.020516** versus
full features. For random forest, the change is **+0.031490**. The best
representation within each family is `lr_selected`
and `rf_selected` under these fixed settings. These
preferences may change after model tuning; they do not show that a field is
universally useful or useless.

### Training scores and temporal variation

| Candidate | Fold 1 F1 | Fold 2 F1 | Fold 3 F1 | Mean training F1 | Mean training−validation F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Majority baseline | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Logistic regression — full | 0.643231 | 0.702455 | 0.733595 | 0.817267 | 0.124173 |
| Logistic regression — selected | 0.693691 | 0.714117 | 0.733020 | 0.814529 | 0.100920 |
| Random forest — full | 0.657928 | 0.572253 | 0.649330 | 0.990300 | 0.363796 |
| Random forest — selected | 0.667216 | 0.638378 | 0.668386 | 0.990202 | 0.332209 |

Training scores are in-sample resubstitution diagnostics. The forest fits its
training data very closely while its later-period validation F1 is much lower.
This supports testing stronger regularization in tuning, but does not prove
that capacity alone causes the gap: temporal shift and repeated-profile
structure also matter. The observed fold-to-fold variation must be reported.
Increasing model complexity has not by itself established better generalization.

Selection is learned separately for the linear and unscaled tree pipelines.
Complete output schemas are saved for every candidate/fold. Comparisons follow
the original row order and immutable forward folds; no records are removed or
reweighted to improve a score.

### Verification and limits

All **40 tests** pass, including five new tests for positive-class metrics,
threshold ties, invalid/misaligned inputs, majority-baseline behavior,
single-class probability handling, pipeline cloning, leakage rejection and
prediction without refitting preprocessing. All 15 comparison fits complete;
the six logistic fits issue no convergence warnings. Confusion counts reconcile
with validation sizes, and each fold's membership hash is identical across models.

Notebook 04 contains **six actually executed Python cells** with saved outputs.
The current runtime lacks Jupyter/IPython/nbformat; separate fresh-kernel runs
and canonical format validation remain final submission gates. These Python
execution checks do not establish that final notebook gate.

The same development folds have informed representation and model choices;
their scores can be optimistic and are not untouched-test estimates. The
original source-timing, repeated-record and provenance limitations remain.
No model has been refitted on all development data. No fold model pickle or
row-level prediction file is published; aggregate evidence is sufficient here.
The final fitted model is a later deliverable after settings are frozen.

### Reproduce and proceed

```bash
python -m src.model_comparison
python -m unittest discover -s tests -v
```

Results, confusion counts, estimator parameters, feature names and checksums
are in `data/processed/results/model_comparison/`. Use `make_model_pipeline` for later searches so
all preprocessing/selection is fitted inside CV. tuning should tune both
learned families with a documented modest grid or randomized search, report
the spaces and all candidate/fold scores, and compare against this untuned
reference. Consider regularization strength/class weighting for logistic
regression and depth, leaf size, feature sampling and tree count for forest.
Any threshold change must be selected using development data only and documented.
Keep the final test for evaluation after all choices are frozen.


---

## tuning — Hyperparameter tuning

The development-selected model is **Logistic Regression**, with **C=1.0, class_weight=balanced** and mean
cancellation F1 **0.732102**. This is the best setting in the
declared grid, not proof of a globally optimal model or final test performance.
The approved dataset, target, problem statement and questions are unchanged.

### Search protocol and rationale

The complete grid was declared and persisted before the new scores were
computed. Both families keep the selected-feature representation from representation comparison and model comparison: top 75% of nonconstant encoded training fields by ANOVA F. Numeric PCA
has been demonstrated but is not retained. It is not retuned in this analysis.

| Family | Parameter | Values |
| --- | --- | --- |
| Logistic Regression | C | 0.01, 0.1, 1, 10 |
| Logistic Regression | class_weight | None, balanced |
| Random Forest | max_depth | 8, 16, None (unlimited) |
| Random Forest | min_samples_leaf | 1, 10 |
| Random Forest | class_weight | None, balanced |

This gives 8 logistic and 12 forest settings, each on three frozen expanding
forward folds: **20 candidates and 60 fits**. Both grids include the exact
model comparison selected-feature control. Logistic C tests stronger/weaker regularization;
forest depth/leaf size tests capacity controls in response to model comparison's large
training–validation gap. Balanced weighting tests the cancellation recall/
precision tradeoff without changing the target or primary metric.

Other settings remain fixed: logistic lbfgs/max_iter=2000/tol=1e−4; forest
100 trees, sqrt feature sampling and bootstrap. Seeds are 42. Logistic numerics
are scaled; forest numerics are unscaled. Search is sequential (`n_jobs=1`),
with two workers per forest. The modest grid is intentional; no adaptive grid
expansion or pursuit of a particular score occurs.

### Leakage-safe validation and scoring

`GridSearchCV` clones the complete raw-input pipeline for each candidate/fold.
Imputation, category vocabularies, scaling and supervised selection fit only
the training prefix. Balanced class weights are calculated separately from
each training fold. No oversampling, undersampling or dataset reweighting is
performed outside that explicit model option.

The primary metric is the **unweighted mean cancellation-class F1** across the
same three development folds. The probability rule remains **≥0.5 ⇒ class 1**.
A custom multimetric scorer uses that rule consistently for both models,
including exact ties. Accuracy, precision, recall, ROC-AUC and confusion counts
are also recorded. Exact mean-F1 ties follow the predeclared family and grid
order. There is no threshold search.

`refit=False` prevents an automatic full-development refit. Errors or convergence
warnings stop the experiment. The held-out test is never fitted, transformed,
scored or summarized. Source/split hashes and development row alignment are
verified before searching.

### Measured results

| Family | Selected search parameters | Untuned F1 | Tuned F1 | Change |
| --- | --- | ---: | ---: | ---: |
| Logistic Regression | C=1.0, class_weight=balanced | 0.713609 | 0.732102 | +0.018492 |
| Random Forest | class_weight=balanced, max_depth=None, min_samples_leaf=10 | 0.657994 | 0.669294 | +0.011300 |

![Tuning comparison](../figures/08_hyperparameter_tuning.png)

| Family | Mean F1 | Accuracy | Precision | Recall | ROC-AUC | Training F1 | F1 gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.732102 | 0.800244 | 0.715322 | 0.756772 | 0.882666 | 0.827779 | 0.095677 |
| Random Forest | 0.669294 | 0.805928 | 0.876136 | 0.543722 | 0.889701 | 0.872311 | 0.203017 |

The complete 20-setting rankings, all 60 fold rows, raw sklearn CV outputs and
full estimator parameters are saved in `data/processed/results/tuning/`. The reported
candidate-table SD is sample SD over the three fold F1 values; sklearn's raw
CV tables use population SD. Neither is a confidence interval. Training F1 is
an in-sample diagnostic and not an independent generalization estimate.

Both untuned control settings reproduce their model comparison per-fold F1, accuracy,
precision and recall within 1e−12, using identical validation membership.
Logistic ROC-AUC also matches within 1e−12. Forest ROC-AUC is checked to 1e−7;
exact differences and tolerances are recorded in `control_parity.csv`.
This supports attributing the measured differences to the declared model
settings under this protocol rather than an altered split or threshold.

### Limits and interpretation

The logistic winner retains C=1 and changes class weighting to balanced.
Mean cancellation recall rises from 0.659498 to 0.756772, while precision
falls from 0.783364 to 0.715322 and accuracy from 0.809314 to 0.800244.
The F1 improvement therefore reflects a measured tradeoff, not a gain on all
metrics. C=0.1 with balanced weights is close (mean F1 0.731697); the small
difference is not evidence of a statistically distinct or universally superior C.

The best forest combines balanced weights with min_samples_leaf=10 and unlimited
depth. Its training–validation F1 gap falls from 0.332209 to 0.203017, but its
mean validation F1 remains below the selected logistic model. Very shallow
forest settings underperform in this grid; regularization does not automatically
improve every setting. No additional search was triggered by these observations.

Representation and model tuning reuse development folds, so selection can make
winning scores optimistic. No nested CV or unbiased final test estimate is
claimed. The grid is finite; a winning edge value is not evidence that broader
search would improve performance. Differences in training–validation gaps
can reflect model capacity, temporal shift and repeated profiles together.

Class weighting can trade precision for recall. Evaluate that tradeoff using
the recorded secondary metrics; do not replace the agreed F1 objective because
another metric looks more favorable. All original source-timing, repetition,
partial-year coverage and provenance limitations remain. The project remains
a retrospective arrival-cohort analysis, not a validated live booking system.

### Verification and reproducibility

The first complete 60-fit run stopped during its final strict parity check:
forest ROC-AUC differed from model comparison by at most 1.14e−8, while its threshold
metrics matched exactly. This is consistent with floating-point summation near
tied probability ranks. Only the secondary forest AUC comparison tolerance was
changed to 1e−7; the grid, metric calculations, threshold and selection rule
were not changed. A focused test rejects changes to F1 or larger AUC differences.
The unchanged 60-fit grid was rerun; the published notebook/results come from
that completed rerun (120 grid fits across the two attempts, excluding unit tests).

All **47 tests pass**, including seven new checks covering grid/control counts,
probability-threshold scoring, training-fold-only fit calls, no global refit,
protocol-change rejection, deterministic ranking, numerical parity and frozen-setting validation.
All 60 fits completed with zero failed fits and zero convergence warnings.
Confusion counts reconcile with fold sizes and cancellations; aggregate mean
scores are derived from actual per-fold scores.

Notebook 04 preserves six model comparison code cells and appends five tuning code
cells executed sequentially in a fresh Python process with captured outputs.
A full 11-cell fresh Jupyter-kernel execution, canonical notebook validation
and clean dependency-install check remain final submission gates.

```bash
python -m src.tuning
python -m unittest discover -s tests -v
```

### Frozen evaluation configuration

`final_selection.json` stores the selected family, representation, complete
estimator settings, fixed threshold and frozen data lineage. It also preserves
the best setting within each family for interpretation, not to select a model
later using test scores. `build_frozen_pipeline(selection)` returns an unfitted
pipeline and rejects changed settings/defaults. **No final trained model is
claimed or saved yet.**

The selected pipeline is fitted on development rows only before final test evaluation. Test scores must not change the threshold, model family or selected settings.


---

## evaluation — Final evaluation and error analysis

### Frozen model and official evaluation

Before test-label access, the protocol fixed the selected-feature Logistic
Regression pipeline, `C=1`, balanced class weights, and probability threshold
0.5 from tuning. The complete preprocessing/selection/classifier pipeline was
fitted on all **95,415 development bookings**. It retained **406 encoded
features** and converged in 238 iterations. It was then evaluated once on the
**23,795 later-arrival bookings** from 2017-04-23 through 2017-08-31.

| Metric | Held-out result |
| --- | ---: |
| Cancellation-class F1 | **0.750592** |
| Accuracy | 0.760874 |
| Precision | 0.654187 |
| Recall | 0.880321 |
| ROC-AUC | 0.875977 |
| Brier score | 0.160007 |

The test set contains 9,726 cancellations (40.87%). Confusion counts are
**TN=9,543, FP=4,526, FN=1,164, TP=8,562**. The model detects 88.03% of
cancellations but only 65.42% of predicted cancellations are correct. Balanced
class weighting favors recall: false positives considerably exceed false
negatives. This tradeoff must accompany the F1 result.

![Final test performance](../figures/09_final_test_performance.png)

Mean development F1 was 0.732102, versus 0.750592 on the held-out period. The
later result did not decline, but the difference is not an estimated improvement:
the periods have different cancellation rates and one holdout supplies neither
a sampling distribution nor a confidence interval.

### Where the model fails

City Hotel has a 25.59% error rate (3,521 FP; 659 FN), versus 20.24% for Resort
Hotel (1,005 FP; 505 FN). Raw counts partly reflect the larger City Hotel group.
Online TA is the largest and hardest market segment: 31.95% error, precision
0.5861 and recall 0.8681. Direct bookings have F1 0.5027; Complementary has F1
0.1667 but only 105 rows and 15 cancellations.

Lead-time performance varies markedly. For 0–7 days, cancellation F1 is 0.3344
and recall 0.4054; for 8–30 days, F1 is 0.5882. The model performs strongly on
366+ days (F1 0.9481), but this can reflect strong correlated signals and does
not establish a causal lead-time effect. No Deposit bookings account for almost
all errors (4,525 FP; 1,156 FN). Non Refund is nearly perfectly separated in
this period: 2,290 of 2,291 rows cancel. Refundable has only 22 rows, so its
metrics are unstable.

![Final error diagnostics](../figures/10_final_error_analysis.png)

Predicted probabilities exceed observed cancellation rates in every populated
fixed bin. This is consistent with balanced weighting and shows that the scores
should not be presented as calibrated probabilities. Brier score is supplied
as a probability diagnostic, not as a tuning objective. Calibrating or changing
the threshold now would use test evidence and is therefore not performed.

### Concrete wrong predictions

The frozen example rule publishes five most-confident and five near-threshold
errors of each type. These are diagnostic extremes, not representative samples.

- False positive source row 14,182 (Resort Hotel, 80-day lead, Direct,
  No Deposit, one previous cancellation) received probability **0.999603** but
  was not canceled. The model can overgeneralize from cancellation-associated
  history/category patterns.
- False negative source row 94,387 (City Hotel, one-day lead, Complementary,
  Transient-Party, four previous cancellations, three special requests) received
  probability **0.006030** but canceled. Row 94,388 has the same displayed
  pattern and outcome, illustrating the retained repeated-profile limitation.
- Near-threshold errors lie within about 0.001 of 0.5; small probability changes
  would flip them, unlike the confident errors above.

All 20 examples and their safe booking fields are in `error_examples.csv`.
Reservation-status fields are excluded. Row IDs permit audit against the public
source without asserting that any individual field caused an error.

### Model inspection and limitations

The largest positive coefficient is `deposit_type_Non Refund` (+2.9297).
Several large coefficients belong to specific agent categories; required car
parking spaces has a large negative coefficient (−2.2908). Coefficients act on
encoded/scaled inputs, so magnitudes are not uniformly comparable and are not
causal feature importance. Rare categories can produce unstable estimates.

Subgroup analysis occurred only after official test access and did not alter
the model, fields, weights or threshold. A single chronological holdout remains
period-specific. Retained repeated profiles, incomplete source timing, partial
seasonal coverage, uncertain live-prediction availability and unfinished source
provenance constrain generalization. High performance for Non Refund and long
lead-time groups may reflect dominant associations or recording practices.

### Verification

All **53 tests pass**, including six evaluation tests for immutable protocol,
threshold alignment, subgroup/probability reconciliation, distinct error
examples, leakage-safe exports, coefficient shape and serialization round trips.
Input and tuning hashes match; prediction leaves the fitted representation
unchanged. The complete fitted pipeline is saved in
`models/final_logistic_regression.joblib`; aggregate evidence, row predictions,
coefficients and checksums are in `data/processed/results/evaluation/`.

Notebook 05 has six cells executed in a fresh Python process that load, inspect
and verify the completed official evaluation. A full fresh-Jupyter run,
canonical notebook validation and clean-install check remain final gates.

```bash
python -m src.final_evaluation
python -m unittest discover -s tests -v
```

research answers should answer the three unchanged questions using measured EDA,
development comparisons and this final test result. Do not claim causality,
global optimality or calibrated probabilities; do not reselect after test.


---

## research answers — Answers to the approved research questions

This synthesis uses the evidence published through evaluation at commit
[`ad4ddff`](https://github.com/faraaz1027-cloud/cse437-hotel-cancellation-15/commit/ad4ddfffcd191b479021754ad16e0f96a4dd2397).
The dataset, target (`is_canceled`), original problem and question wording are
unchanged. No model is fitted, no new predictions are made, and no threshold,
feature set, weighting or calibration decision is changed in research answers.

### 1. Which booking and customer-related factors have the biggest effect on hotel cancellations?

**Answer:** Among the three predeclared factors, deposit type shows a large
descriptive separation, longer lead time has a consistent increasing
association, and prior cancellations show a substantial but non-monotonic and
repetition-sensitive association. These are observed relationships, not
identified causal effects or an exhaustive ranking of all predictors.

The following rates use only the **95,415 development bookings**:

| Factor | Observed development result | Interpretation and qualification |
| --- | --- | --- |
| Deposit type | Non Refund: **99.25%** canceled (12,296 bookings); No Deposit: **26.82%** (82,979); Refundable: **10.71%** (140) | Large separation between the two well-populated categories. Refundable is small; the data do not establish that a deposit policy causes cancellation. |
| Lead time | Rates increase through all six fixed bins, from **9.43%** at 0–7 days (17,247 bookings) to **80.81%** at 366+ days (2,079) | The increasing pattern also occurs within both hotels. This does not mean every long-lead booking cancels or that lead time has an isolated causal effect. |
| Previous cancellations | For 0, 1, 2–3 and 4+: **32.07%, 95.65%, 31.29%, 74.77%**, with n=89,079; 5,951; 163; 222 | “More previous cancellations always means higher risk” is unsupported. The two highest-count groups are sparse. |

Evidence: [deposit table](processed/eda/deposit_rates.csv),
[lead-time table](processed/eda/lead_time_rates.csv),
[prior-cancellation table](processed/eda/prior_cancellation_rates.csv),
[lead time within hotels](processed/eda/lead_time_by_hotel.csv), and
[deposit within hotels](processed/eda/deposit_by_hotel.csv).

Repeated-profile weighting changes the interpretation. Giving each profile
group total weight one changes the overall development cancellation rate from
**36.13% to 25.26%**; for 4+ prior cancellations, **74.77% becomes 26.32%**.
The Non Refund rate remains high at **93.49%**, and the lead-time endpoints
remain ordered (**8.17% to 47.10%**). This sensitivity describes an average
profile rather than an average booking. It is not a reason to delete retained
records or change the frozen model weights.
Sources: [EDA summary](processed/eda/eda_summary.json),
[prior-history sensitivity](processed/eda/sensitivity_prior_cancellations.csv),
[deposit sensitivity](processed/eda/sensitivity_deposit.csv), and
[lead-time sensitivity](processed/eda/sensitivity_lead_time.csv).

The final fitted model also assigns the largest positive encoded coefficient
to `deposit_type_Non Refund` (**+2.929658**); parking spaces has a negative
coefficient (**−2.290799**), and several agent categories have large
coefficients. These are conditional model associations in an encoded,
regularized representation. Numeric scaling, log transforms, correlated fields
and categorical coding prevent reading coefficient magnitudes as a universal
raw-feature importance ranking. No causal or independently validated
importance ranking was estimated. Source:
[all 406 coefficients](processed/results/evaluation/feature_coefficients.csv).

### 2. How accurately can machine-learning models predict whether a hotel booking will be canceled?

**Answer:** The frozen selected-feature Logistic Regression achieved **F1
0.750592**, **76.09% accuracy**, **65.42% precision**, **88.03% recall**, and
**ROC-AUC 0.875977** on **23,795 held-out bookings**, with arrivals from
2017-04-23 through 2017-08-31. These are results for the selected model in one
later period—not test scores for every candidate or a guarantee for new hotels.

The model correctly detected **8,562 of 9,726 cancellations**, missed **1,164**,
and incorrectly flagged **4,526 noncancellations**; **9,543** noncancellations
were correctly classified. It catches most cancellations, but a substantial
number of alerts are false. Balanced weighting improved the development F1
tradeoff, not every metric. The threshold stayed at **0.5**.
Sources: [final metrics](processed/results/evaluation/final_metrics.csv),
[evaluation protocol](processed/results/evaluation/evaluation_protocol.json), and
[evaluation summary](processed/results/evaluation/evaluation_summary.json).

The **40.87%** test cancellation prevalence differs from development's
**36.13%**. Test F1 being above the **0.732102** development mean is not an
estimated improvement: the training size, booking mix and time windows differ.
Only the development-selected pipeline received the official final evaluation.

Errors are uneven: F1 is **0.3344** for lead times of 0–7 days and **0.5882**
for 8–30 days; Online TA's error rate is **31.95%**, and it contributes
**3,653 false positives**. City Hotel's error rate is **25.59%**, versus
**20.24%** for Resort Hotel. Group sizes and prevalence differ, so these
post-test diagnostics are not causal explanations or model-selection rules.
The test Non Refund group contains **2,290 cancellations among 2,291 bookings**;
this near-separation warrants caution about source timing and booking mix.
Sources: [subgroup metrics](processed/results/evaluation/subgroup_metrics.csv) and
[the detailed error analysis](README.md).

In every populated fixed probability bin, the observed cancellation rate is
below the mean predicted probability. The outputs therefore should not be
advertised as calibrated probabilities. The Brier score is **0.160007**, a
probability-error measure, not another accuracy percentage. No calibration or
threshold adjustment is made after the test. Source:
[probability diagnostics](processed/results/evaluation/probability_diagnostics.csv).

Operationally, the evidence supports further evaluation of a cancellation
screening aid, not an automatic cancellation/overbooking policy. False alerts
and missed cancellations have different costs; those costs and prospective
financial benefits were not measured.

### 3. Which machine-learning model gives the best result after data preprocessing, feature selection, dimensionality reduction, and hyperparameter tuning?

**Answer:** Selected-feature **Logistic Regression with C=1 and balanced class
weights** is the **best evaluated under this protocol**, using mean cancellation
F1 across the three frozen forward development folds as the primary metric.
It achieves **0.732102**, versus **0.669294** for the best tested Random Forest.
The final pipeline does **not** include PCA: dimensionality reduction was
tested and documented, but the selected-only representation performed better.

| Development comparison | Mean cancellation F1 | Decision supported |
| --- | ---: | --- |
| Majority baseline | 0.000000 | Accuracy alone can hide failure to detect cancellations. |
| Reference Logistic Regression, all feature engineering features | 0.693094 | Full-feature control. |
| Reference Logistic Regression, selection only | 0.713609 | Best of the four reference representations. |
| Reference Logistic Regression, numeric PCA | 0.694633 | Does not beat selection only. |
| Reference Logistic Regression, selection then PCA | 0.701023 | Does not beat selection only. |
| Untuned Random Forest, selected features | 0.657994 | Trails selected Logistic Regression on F1. |
| Tuned Random Forest, selected features | 0.669294 | Best tested forest: balanced weights, unlimited depth, leaf size 10. |
| Tuned Logistic Regression, selected features | **0.732102** | Final development-selected pipeline: C=1, balanced weights. |

Sources: [representation comparison](processed/representations/representation_comparison.csv),
[untuned model comparison](processed/results/model_comparison/model_comparison.csv),
[tuning comparison](processed/results/tuning/tuning_comparison.csv), and
[frozen settings](processed/results/tuning/final_selection.json).

Selection retains the top 75% of nonconstant encoded training features by
ANOVA F; numeric PCA retains at least 95% of training numeric variance. All
learned transformations, selection and PCA were fitted inside training folds.
The grid evaluated **20 settings × 3 folds = 60 fits**. Logistic tuning raises
mean F1 by **0.018492**, but mean accuracy falls from **0.809314 to 0.800244**
while recall rises from **0.659498 to 0.756772**. This is a precision/recall
tradeoff, not improvement on every measure.

These are staged comparisons, not an exhaustive search over every model and
representation combination. PCA variants were compared with the fixed
logistic reference, not tuned for both families. Derived features were
evaluated as a bundle; no isolated ablation establishes each feature's benefit.
Reusing development folds for several choices introduces selection optimism;
there is no nested-validation or statistical-significance claim. The test
result confirms the selected pipeline's period-specific performance, but
cannot establish a test-set ranking over models that were not tested there.

### Conclusion

The project demonstrates a complete preprocessing-to-evaluation workflow and
finds useful retrospective predictive signal. Deposit and lead-time patterns
are strong descriptive findings; prior-history interpretation is sensitive to
repetitions. The chosen logistic pipeline balances cancellation detection and
false alerts better on the agreed development F1 metric than the alternatives
tested, while remaining limited by feature timing, repeated profiles, temporal
coverage, uncalibrated probabilities and the narrow model search.

Any future calibration, feature-timing study, deposit ablation, threshold/cost
study or external validation must use a newly declared development/evaluation
design and fresh evaluation data. It must not change this frozen result.


---

## eligibility — Eligibility and direct leakage removal

### Implemented decisions

1. Exclude `reservation_status` and `reservation_status_date` from candidate
   predictors. Neither field, nor a derivative, belongs in a predictive model.
2. Store `is_canceled` separately, preserving its source values and row order.
3. Exclude a record only when adults, children, and babies are all known and
   their sum is zero. This defines the analytic population; it does not prove
   such administrative records are erroneous.
4. Keep unknown guest totals and zero-adult bookings with positive total guests.
5. Keep repeated records. Without a booking ID, exact matches do not establish
   accidental duplication. Group identical candidate predictors independently
   of the target and statuses for use in partition checks.
6. Keep and flag zero-night, zero-price, negative-price, and exceptionally high
   positive-price records. The ADR > 1,000 flag is for review only and never
   changes eligibility. No full-data IQR fence is used to delete or cap values.

The original CSV and proposal documents remain unchanged. No retained source
values were edited. No imputation, encoding, scaling, model fit, or evaluation
split occurs in this stage.

### Verified results

| Measure | Result |
| --- | ---: |
| Original rows | 119,390 |
| Excluded known-zero-guest rows | 180 |
| Eligible rows | 119,210 |
| Candidate predictors | 29 |
| Not canceled / canceled in eligible cohort | 75,011 / 44,199 |
| Retained unknown guest totals | 4 |
| Retained zero-adult bookings with positive guest totals | 223 |
| Retained zero-night bookings | 645 |
| Retained negative-ADR rows | 1 |
| Retained zero-ADR rows | 1,810 |
| Retained ADR-above-1,000 rows | 1 |
| Original full-row additional copies among retained bookings | 31,980 |
| Distinct candidate-predictor groups | 86,707 |
| Additional copies of candidate-predictor combinations | 32,503 |
| Candidate rows participating in repeated groups | 40,730 |
| Candidate groups containing both outcome labels | 265 |

Full-row duplicates and candidate-predictor duplicates are different counts:
dropping status information and target can make additional rows identical.
Conflicting outcomes are kept; selecting a label based on majority or dropping
one outcome would distort the supervised task. Every candidate group has one
arrival date, permitting whole-date splits without dividing these groups.

### Outputs and reproducibility

- [Executed notebook](../notebooks/02_preprocessing.ipynb)
- [Reusable implementation](../src/eligibility.py)
- [Processed files and loading instructions](README.md)
- [Measured summary and hashes](processed/eligibility_summary.json)
- [Boundary and leakage-invariance tests](../src/tests/test_eligibility.py)

The four verified row-level output files are committed in `data/processed/`.
Their hashes match the recorded eligibility outputs. This repository contains the
code, executed aggregate results, decisions, and file hashes needed to reproduce
them. These outputs are ready for the evaluation-split work in validation.

Run from the repository root after supplying the original CSV:

```bash
python -m src.eligibility
python -m unittest discover -s tests -v
```

Notebook 02's five code cells executed sequentially in a fresh Python process
using IPython, with real outputs saved. Original-value comparison, complete
row accounting, input/output hashes, compressed-file reloads, and metadata
alignment passed. Four focused tests check unknown guests and anomalous rows,
independence from outcome information, original-data preservation, and handling
of an unexpected negative guest count. A separate fresh Jupyter-kernel run
remains a final submission verification task.

### Decisions carried forward

**validation:** define chronological holdout and forward development validation
before predictor-outcome EDA. Freeze row assignments and verify no duplicate
group crosses partitions in any split. No dates or ratios are fixed here.

**preprocessing:** negative ADR will be treated as unavailable under a nonnegative-price
assumption, then imputed using training-fold statistics. Preserve zero and
high positive ADR; assess any learned transform or cap using development folds.
Treat `agent` as a category, with an explicit no-agent value under the source
NULL convention. Exclude the sparse `company` identifier from the primary
model and consider a company-presence indicator in feature engineering. Distinguish these
not-applicable categories from unknown country or child information. These
rules are documented but have not been applied in the eligibility candidate CSV.

**Prediction-time availability:** the 29 candidate columns are not the final
feature set. Review potentially updated fields such as `assigned_room_type`,
`booking_changes`, and `days_in_waiting_list` against the intended prediction
time. Removing direct outcome leakage does not establish booking-time validity.

**Substantive focus:** investigate longer lead time, deposit type, and prior
cancellations as named hypotheses in development data. Any findings are
associations, not causal effects. The approved research questions are unchanged.

### Limitations

The zero-guest rule narrows the population. Retaining repeated records preserves
their frequency weight and may emphasize common booking patterns; report that
choice and consider a development-only sensitivity check later. Unusual ADR,
missing values, and uncertain feature timing remain explicit later tasks.
Full-source quality summaries were inspected before splitting; report this
transparently rather than claiming that no information about the later holdout
was ever seen.

Sources: [Kaggle dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand);
Antonio, N., de Almeida, A., and Nunes, L. (2019), *Hotel booking demand datasets*,
Data in Brief 22, 41–49, https://doi.org/10.1016/j.dib.2018.11.126;
[original documentation](https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/).


---

## validation — Chronological evaluation plan

### Decision and scope

The question is whether the model generalizes to later arrival cohorts.
Use approximately 80% of eligible bookings for development and the latest 20%
for a final test. Preserve whole dates and the duplicate groups established in
eligibility. The boundary is chosen by the nearest prefix row count; ties favor the
earlier date. Cancellation outcomes and model scores do not choose the dates.

This is retrospective arrival-cohort evaluation. It is not a booking-creation
deployment simulation: the original source lacks reliable creation-time
snapshots for all fields and live label-availability records. No temporal gap
or embargo is applied. Prediction-time feature availability remains a review
item before modeling; the approved project wording is unchanged.

### Fixed holdout

| Partition | First arrival | Last arrival | Rows |
| --- | --- | --- | ---: |
| Development | 2015-07-01 | 2017-04-22 | 95,415 |
| Final test | 2017-04-23 | 2017-08-31 | 23,795 |

The partitions account for all 119,210 eligible bookings. All 67,961
development duplicate groups and 18,746 test duplicate groups remain separate.
The target is not recomputed or changed, and no records are added or removed.

### Forward cross-validation

Within development, whole-date boundaries near 25%, 50%, and 75% cumulative
row counts define three validation blocks. Training expands over time.

| Fold | Training dates | Training rows | Validation dates | Validation rows |
| --- | --- | ---: | --- | ---: |
| 1 | 2015-07-01 – 2016-01-26 | 23,797 | 2016-01-27 – 2016-06-21 | 23,893 |
| 2 | 2015-07-01 – 2016-06-21 | 47,690 | 2016-06-22 – 2016-11-06 | 23,776 |
| 3 | 2015-07-01 – 2016-11-06 | 71,466 | 2016-11-07 – 2017-04-22 | 23,949 |

Earlier validation rows become training rows in subsequent folds, as expected
for expanding forward validation. There is no row or duplicate-group overlap
between training and validation within any one fold. The initial training block
does not receive validation predictions. Every later development row validates
once, and the final test is excluded from all folds. Blocks have similar counts
but unequal calendar durations; inspect fold-level variation in later results.

![Fixed split and forward validation](../figures/02_evaluation_timeline.png)

### Metrics and model-selection commitments

- Primary: F1 for cancellation class 1, with unweighted mean across the three
  forward folds; report individual folds and variability. Set `zero_division=0`
  where precision/F1 is undefined.
- Secondary: accuracy, precision, recall, and ROC-AUC. ROC-AUC uses continuous
  probabilities or decision scores.
- Planned comparisons: majority-class dummy baseline, logistic regression,
  and random forest, all using the identical validation assignments.
- Default probability threshold: 0.5. Any threshold search, transformation
  choice, feature selection, dimensionality reduction, or hyperparameter choice
  must use development data only. Their validation scores are selection scores,
  not an independent final performance claim.
- No shuffle or split seed is needed. Use seed 42 for later stochastic
  estimators/searches and record any additional randomness.
- After selecting all settings, freeze the full pipeline and refit on all
  development data. Use the final test once in evaluation for evaluation and error
  analysis. Do not tune further in response to that result.

### Implementation and verification

The [split module](../src/splitting.py) verifies eligibility hashes, uses only
metadata for membership, exports assignments, and rejects inconsistent changes
to a frozen plan. The compressed CSV preserves original cohort row order.
The helper `development_cv` produces indices relative to the development subset
for scikit-learn; an example in [data/README.md](README.md)
documents alignment explicitly.

Checks confirm full row accounting, unique source positions, strictly earlier
training dates, whole-date boundaries, zero duplicate-group contamination,
zero test rows in CV, and expanding training periods. Both outcome classes are
present in each fixed development training/validation block. That class check
is performed after fixing the dates and does not influence membership. No
holdout class distribution or model score is calculated for validation.

Seven targeted split tests cover nonchronological source order, relative CV
indices, target independence, deterministic ties, identity/group problems,
test contamination, insufficient dates, and frozen-plan protection. The four
existing eligibility tests also pass (11 tests total).

Notebook 02 has nine code cells, executed sequentially in a fresh Python
process using IPython with actual outputs. Its end-to-end checks verify
compressed-file reloads, unchanged eligibility/source hashes, and byte-preserving
reruns of the frozen split. A separate fresh Jupyter-kernel run remains a
final submission verification task.

### Artifacts and commands

- [Verified split assignments](processed/splits/validation_assignments.csv.gz) — committed;
  the file checksum matches the frozen evaluation plan.
- [Fixed plan and hashes](processed/splits/validation_split_plan.json)
- [Executed Notebook 02](../notebooks/02_preprocessing.ipynb)
- [Split tests](../src/tests/test_splitting.py)

From the repository root:

```bash
python -m src.splitting
python -m unittest discover -s tests -v
```

The split module can run from the committed processed files. Rerunning all of
Notebook 02 additionally requires the original CSV for the earlier eligibility cells.

### Limitations and contribution disclosure

Later arrival periods may differ because of seasonality, hotel mix, or broader
changes. The source does not provide a real booking/guest identifier, so the
group guarantees apply to the recorded candidate-predictor duplicate groups.
Full-source raw-data quality summaries were inspected before the holdout was
defined; disclose this instead of claiming the holdout was never seen in any
form. No preprocessing or predictive model has been fitted in validation.


---

## preprocessing — Preprocessing inside training folds

### Applied decisions

| Item | Rule | Rationale |
| --- | --- | --- |
| Company identifier | Exclude from the initial model inputs | Sparse nominal ID; a presence indicator remains a feature engineering candidate |
| Assigned room, booking changes, waiting-list days | Exclude from initial model inputs | Potentially updated after reservation; conservative timing policy |
| Negative ADR | Mark value missing; retain booking | Nonnegative-price assumption, documented instead of deleting the row |
| Numeric missingness | Median fitted separately within each training fold | Avoid learning imputation statistics from validation/test data |
| Entirely missing training field | Keep the field with a zero fallback | Maintain schema; missing flags preserve children/ADR missingness |
| Missing country / ordinary category | Explicit `Unknown` value | Separate unknown information from a real category |
| Agent | Convert codes to nominal strings; null becomes `NoAgent` | Source NULL convention represents no agent; literal code 0 stays distinct |
| Existing `Undefined` values | Retain as their own category | Do not silently equate supplied labels with missing values |
| Skewed numeric fields | log1p for lead time, previous cancellations, previous noncanceled bookings, and ADR | Reduce numeric skew without deleting or clipping high positive values |
| Numeric scaling | Training-fold StandardScaler after imputation/log1p | Suitable numeric magnitudes for linear models; optional unscaled tree variant |
| Categories | Sparse one-hot encoding, training vocabulary only | Unseen values become an all-zero block for that field without refitting |
| Missing indicators | Always append children and ADR missing flags | Distinguish imputed from observed values, including validation-only missingness |

The target and both reservation-status columns remain excluded. Split metadata
and identifiers are rejected if passed as predictors. No rows are removed at
preprocessing, no extra duplicate policy is introduced, and source data are unchanged.

The 25 retained source fields include all three substantive factors committed
before modeling: lead time, deposit type, and previous cancellations. The four
policy exclusions do not fulfill the separate statistical selection and
dimensionality-reduction requirements; those remain representation comparison work.

### Before/after evidence

| Fold | Training rows | Validation rows | Encoded columns | Children training median | ADR training median | Nonfinite encoded values |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 23,797 | 23,893 | 328 | 0 | 76.50 | 0 |
| 2 | 47,690 | 23,776 | 422 | 0 | 80.75 | 0 |
| 3 | 71,466 | 23,949 | 491 | 0 | 90.00 | 0 |

Four child values are missing in each expanding training window. They are
imputed to its observed median of zero and retain a missing flag. There are no
missing child values in the validation windows. The single negative ADR occurs
in fold 3 validation: it becomes missing and receives the **training** median
of 90 before log transformation/scaling. No training window contains that
negative value.

| Fold | Training country unknowns | Validation country unknowns | Training no-agent values | Validation no-agent values |
| --- | ---: | ---: | ---: | ---: |
| 1 | 158 | 123 | 3,471 | 3,359 |
| 2 | 281 | 72 | 6,830 | 2,434 |
| 3 | 353 | 120 | 9,264 | 4,481 |

All zero ADR values are retained. The ADR of 5,400 occurs in fold 1 validation
and subsequently enters training in later folds; log1p(5400) is approximately
8.594. No validation-derived cap is fitted. Counts across expanding folds
overlap and must not be added as if they represented distinct bookings.

### Category changes and limitations

Train and validation use the same encoding schema within each fold. Schemas
differ between folds because the vocabularies grow with the training period.

| Fold | Validation rows with unseen month | Unseen country | Unseen market segment | Unseen agent |
| --- | ---: | ---: | ---: | ---: |
| 1 | 23,475 | 114 | 62 | 1,531 |
| 2 | 0 | 31 | 0 | 349 |
| 3 | 0 | 15 | 0 | 613 |

The first training window covers only part of a calendar year. One-hot encoding
cannot learn category-specific effects for months absent from training. The
all-zero handling is safe computationally but loses that field's category
information; other date fields remain. Record this seasonal limitation and
consider cyclic calendar features in feature engineering without changing the split.

Excluding potentially updated fields reduces an identifiable timing risk; it
does not establish that every remaining column is available at booking
creation. The dataset still supports the previously declared retrospective
arrival-cohort evaluation. No booking-time deployment validity is claimed.

Medians, log transformations, and standardization are documented initial
choices. Their predictive advantage is not established until later modeling.

### Implementation and checks

- [Preprocessing factory](../src/preprocessing.py) returns a fresh, cloneable
  sklearn pipeline with fixed domain rules and learned column transforms.
- [Development-fold audit](../src/preprocessing_audit.py) uses the frozen validation
  indices and exports aggregate results and per-fold feature schemas.
- [Summary and schemas](README.md) record versions,
  input hashes, missing counts, dimensions, and fitted medians.
- [Executed Notebook 02](../notebooks/02_preprocessing.ipynb) contains the full
  eligibility and preprocessing sequence with saved outputs.

For every fold, checks compare learned medians against the training data,
verify scaler means and fitted row counts, and compare encoder vocabularies to
training categories. Validation transformation leaves fitted state unchanged.
Both scaled and unscaled variants produce finite matrices with consistent
within-fold dimensions. The audit does not read the target file and does not
fit or transform any final test rows.

Eight new tests cover train-only statistics, unseen categories, fixed domain
rules and source preservation, excluded-field independence, accidental target/
metadata inputs, entirely missing fields, cloneable scaled/unscaled variants,
and invalid numeric codes/counts. All **19 tests** including prior stages pass.
Notebook 02's **13 code cells** execute sequentially in a fresh Python process
using IPython with real outputs. A separate fresh Jupyter-kernel run remains
part of final submission verification.

The exact run used Python 3.12.13, pandas 2.2.3, NumPy 2.3.5, SciPy 1.17.0,
and scikit-learn 1.8.0. Starter dependency ranges remain in requirements.txt;
clean-install dependency validation is a final reproducibility task.

### Reproduction

From the repository root:

```bash
python -m src.preprocessing_audit
python -m unittest discover -s tests -v
```

Keep `make_preprocessor()` inside the estimator pipeline passed to CV. Do not
export a single globally imputed/scaled matrix and then cross-validate it.
feature engineering features and representation comparison selection/reduction must also remain within the
fold-safe pipeline where they learn from data. Refit on all development rows
only after model settings are selected; reserve the test for evaluation.

descriptive analysis should now produce development-only descriptive statistics, plots, and
observations relevant to the original questions. The original CSE437 document,
approved question wording, eligibility files, and validation assignments are unchanged.

Source semantics reference: Antonio, N., de Almeida, A., and Nunes, L. (2019),
*Hotel booking demand datasets*, https://doi.org/10.1016/j.dib.2018.11.126;
https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/ .


---

## feature engineering — Derived features

feature engineering adds a reusable deterministic feature transformer and a preprocessing
factory for the unchanged hotel-cancellation project. The target, approved
questions, original document, eligible cohort, and frozen evaluation split are
preserved. The audit reads no target values and fits/transforms no test rows.

### Feature choices and justification

| Feature | Formula and missing-value rule | Reason to evaluate later |
| --- | --- | --- |
| `total_nights` | Weekend + weekday nights; missing component means missing total; zero stays retained | Reservation duration in one field |
| `total_guests` | Adults + children + babies; missing component means missing total | Booking party size; does not interpret unknown children as zero |
| `previous_bookings_total` | Previous canceled + previous noncanceled bookings | Amount of recorded history |
| `has_booking_history` | 1 when history total > 0; 0 when zero; missing when unknown | Distinguishes no history from observed history |
| `previous_cancellation_share` | Previous cancellations / history total; set to zero only for known zero total; unknown stays missing | Relative cancellation history, interpreted with history presence |
| `company_code_recorded` | 1 for a non-null code, 0 for null; literal code 0 is present | Coarse recording indicator without sparse company ID categories |
| `arrival_month_sin` | sin(2π(m−1)/12), m=1…12 | Fixed seasonal coordinate with December next to January |
| `arrival_month_cos` | cos(2π(m−1)/12), m=1…12 | Completes the cyclic representation |

The company flag does not prove corporate payment, sponsorship, or availability
at booking creation. It is explicitly a code-recording proxy. The source month
name is replaced by the cyclic pair; remaining preprocessing source predictors stay
available for later selection. Unknown month labels and invalid company codes
raise errors rather than silently producing arbitrary values.

The resulting candidate schema contains **24 retained source fields + eight
derived fields = 32 fields** before encoding. These choices are justified by
interpretability and representation; they are not claimed to improve prediction.
Statistical feature selection and dimensionality reduction remain representation comparison.

### Preprocessing and integration

`src/feature_engineering.py` supplies `BookingFeatureEngineer` and
`make_feature_preprocessor(scale_numeric=True/False)`. It explicitly replaces
the strict preprocessing domain stage while reusing its fixed cleaning rules. The
original preprocessing factory remains unchanged and available for later comparisons.

Derived values are computed before imputation. Training medians are learned
separately for each numeric field, with a zero fallback for an entirely missing
training column. Therefore an imputed guest total need not equal the sum of
separately imputed guest components. This is deliberate and disclosed; fixed
missing indicators preserve missingness for children, ADR, total nights, total
guests, and cancellation share.

Total nights, guests, and previous bookings receive log1p after imputation,
alongside preprocessing's four logarithmic fields. Ratios, binary indicators, and cyclic
coordinates are not logged. The scaled variant learns StandardScaler statistics
inside each training fold; the tree-compatible variant omits scaling. Other
categorical vocabularies are training-only and tolerate unseen categories.

For later modeling, put a **new** feature preprocessor in the estimator pipeline
passed to CV. Do not prefit a matrix on all development rows. CV positions from
`development_cv(assignments)` apply to development rows in eligibility source order.

### Measured evidence

The fixed development cohort contains **95,415 bookings**. The feature audit
retains **604 zero-night bookings**, preserves **four unknown guest totals**
before imputation, and identifies **86,585 bookings with no recorded history**
and **5,900 with a company code recorded**. These describe input features only.

| Fold | Training rows | Validation rows | Encoded columns in both | Nonfinite values | Validation rows with month absent from training |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 23,797 | 23,893 | 332 | 0 | 23,475 |
| 2 | 47,690 | 23,776 | 421 | 0 | 0 |
| 3 | 71,466 | 23,949 | 490 | 0 | 0 |

Each training fold has medians of three total nights, two total guests, and
zero prior bookings. Learned statistics and vocabularies were checked against
training data; validation transformation did not change fitted state. Both
scaled and unscaled variants passed. Feature names are unique and saved per fold.

The first training window lacks February–June. Fixed cyclic coordinates remain
defined for all those months, unlike a missing month-specific one-hot category.
This solves representation coverage only: it does not supply observed seasonal
outcomes or establish generalization to unseen seasons.

All **29 tests** pass, including seven feature engineering tests covering formulas, missing
components, zero denominators, company-code semantics, month wrap-around,
validation isolation, cloning, excluded fields, and leakage/schema rejection.
Notebook 03's **five code cells** executed sequentially in a fresh Python
process with real outputs saved. Jupyter/IPython/nbformat were unavailable;
full fresh-kernel execution and canonical notebook validation remain final gates.

### Reproduce and continue

From the repository root with the documented dependencies:

```bash
python -m src.feature_audit
python -m unittest discover -s tests -v
```

These commands use the committed processed data and frozen assignments.
Evidence is in `data/processed/features/`: summary, complete schemas, and development
feature descriptive statistics. Only aggregate evidence is exported; no fitted
full-development matrix or model is published.


---

## execution verification verification review

Reviewed on 3 September 2026 from the group-supplied `cse437_test_comparison_3_verification.zip`. This is a review of supplied evidence, not a new run performed by this document or a certification of submission readiness.

### Evidence identity

| Artifact | SHA-256 |
| --- | --- |
| Complete supplied ZIP | `18840d43aaf572b35f191aafdb6c6ff287efc93e2a52e64c14a3043f474e8e36` |
| ZIP entry verification.json | `6dc3ba55d03462208c6ee2d7d512d6828d192968e62dc6a6f3f0ee89cab546be` |
| ZIP entry reproduction_comparison.json | `f02bd0847be1dcf4d6c776c951739c422f248a13a84e8eb6b15e13091999179f` |

All 35 manifest file hashes were checked. All five executed notebooks passed canonical validation; their cell sources and IDs match the repaired notebooks. All 32 locally available frozen files listed in the run context matched, as did the included model comparison and tuning output hash chains. Full logs and executed copies remain in the supplied archive; they are not silently substituted for published historical outputs.

### Execution outcome

Dependency installation and consistency checks passed. The run passed 70 unit tests. Each notebook used a fresh kernel, had sequential code-cell execution counts and contained zero cell-error outputs.

| Notebook | Code cells | Seconds | Errors |
| --- | ---: | ---: | ---: |
| 01_data_audit_and_eda | 15 | 5.656 | 0 |
| 02_preprocessing | 13 | 12.281 | 0 |
| 03_feature_engineering | 10 | 18.032 | 0 |
| 04_modeling_and_tuning | 12 | 529.218 | 0 |
| 05_evaluation_and_error_analysis | 7 | 2.328 | 0 |

The runner reports `passed_with_reproduction_differences`: execution passed, numerical reproduction differed, and `submission_ready` is false. Exit code 2 is the documented numerical-warning outcome, not a notebook execution failure. Original repository and frozen evidence checks passed. Final evaluation verified cached evidence; it did not refit the final model or independently regenerate its predictions.

### Numerical comparison

| Balanced Logistic Regression candidate | Original mean development F1 | Rerun mean development F1 |
| --- | ---: | ---: |
| C=1, lr_06 | 0.7321017246390454 | 0.7313708930914117 |
| C=0.1, lr_04 | 0.7316967924435923 | 0.7316967924435923 |

The rerun winner is lr_04, while the original winner remains lr_06. Other C=1 and C=10 candidates changed; the largest absolute fold confusion-count difference is 41. Some Random Forest ROC-AUC values also differ slightly. These differences are not merely display rounding. The supplied prose summary understates the differences; this review follows the structured comparison and recorded tables.

Candidate parameters, semantic search protocol and recorded input/source hashes agree. Protocol byte differences include line endings. The feature representation and 0.5 threshold are unchanged. No rerun selection was promoted to frozen evaluation; the original final model and published test results remain unchanged.

### Environment and provenance limits

The supplied run used Windows 11, Python 3.12.10, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0 and joblib 1.5.3. Package-freeze records agree after decoding. CPU/numerical-library differences are hypotheses, not an established explanation for the changed optimization results.

The intended repair commit is `589ba0cc0714afc83306ec10ff0a43535163319f`. The supplied run used a downloaded repository archive and does not provide an exact checkout commit or Git status. Matching notebook sources and recorded source hashes do not establish identity of the entire tested checkout. Historical `final_verification.json` remains a record of its earlier run, not current-file certification.

### Remaining submission checks

- Recheck the raw CSV's public location and exact checksum; publication is group-reported complete.
- Bind final validation to an exact source commit and preserve executed notebook evidence with clear provenance.
- Retain the numerical reproduction limitation; do not retune from test results or weaken integrity assertions.
- Finalize the provisional author-supplied AI declaration and complete both members' whole-project review and attributable-commit checks.
- Inspect the rebuilt 10-page PDF and verify final public submission contents before declaring readiness.
