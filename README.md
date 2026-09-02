# Hotel Booking Cancellation Project

**CSE437: Data Science | Group 15**

**Status: Step 14 completed — evidence-backed answers to the three unchanged research questions.** Responsible: **Sadat**. **Next: Step 15 — final report, reproducibility and submission, led by Sadat with both members reviewing.** Final Logistic Regression F1 is 0.750592 on 23,795 held-out bookings. Report Sections 1–8 now include the question answers and limitations; final assembly and submission checks remain pending.

Resume checkpoint: [PROJECT_STATUS.md](PROJECT_STATUS.md) records completed work, remaining checks, and the next action.

## Project

### Group members

| Member | Student ID |
| --- | --- |
| Faraaz Jamil Chowdhury | 24241205 |
| Ihfaz Rashid Sadat | 23301499 |

### Problem statement

Hotel booking cancellations can cause loss of money and make room planning difficult for hotels. The goal of this project is to study the factors that affect booking cancellations and build a machine-learning model that can predict whether a hotel booking will be canceled.

### Research questions

1. Which booking and customer-related factors have the biggest effect on hotel cancellations?
2. How accurately can machine-learning models predict whether a hotel booking will be canceled?
3. Which machine-learning model gives the best result after data preprocessing, feature selection, dimensionality reduction, and hyperparameter tuning?

The problem statement and question wording above are reproduced unchanged from the user-approved project brief.

## Files and folders

| Location | Purpose |
| --- | --- |
| `data/raw/` | Original source data |
| `data/processed/` | Documented outputs of data preparation |
| `data/results/step11/` | Untuned model-comparison evidence |
| `data/results/step12/` | Tuning grids, all search results and frozen selected settings |
| `data/results/step13/` | Final metrics, predictions, subgroup diagnostics and error examples |
| `data/README.md` | Dataset acquisition and provenance |
| `notebooks/` | Numbered analysis notebooks |
| `models/` | Saved models and fitted pipelines |
| `figures/` | Exported figures |
| `report/report.md` | Sections 1–8 drafted, including question answers; final assembly pending |
| `report/step14_research_answers.md` | Detailed answers, source-table links, interpretation limits and Step 15 handoff |
| `requirements.txt` | Starter Python dependencies |

Empty `.gitkeep` files preserve folders in Git.

## Local setup

Use Python 3.11 or newer. The dependency ranges are provisional and have not been tested together for this project.

```bash
git clone https://github.com/faraaz1027-cloud/cse437-hotel-cancellation-15.git
cd cse437-hotel-cancellation-15
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the starter dependencies and launch JupyterLab:

```bash
python -m pip install -r requirements.txt
python -m jupyterlab
```

The uploaded CSV has been audited but is not yet committed here. Obtain it from the source in [data/README.md](data/README.md) and place the original at `data/raw/hotel_bookings.csv` to reproduce the audit. Preserve original data and use repository-relative paths.

## Notebook order

Notebook 01 contains the raw-data audit and development EDA. Notebook 02 implements Steps 5–7. Notebook 03 implements Step 9 derived features and Step 10 selection/reduction. Notebook 04 implements Step 11 modeling and Step 12 tuning. Notebook 05 implements Step 13 final evaluation and error analysis:

1. [Data audit and EDA](notebooks/01_data_audit_and_eda.ipynb)
2. [Preprocessing](notebooks/02_preprocessing.ipynb)
3. [Feature engineering](notebooks/03_feature_engineering.ipynb)
4. [Modeling and tuning](notebooks/04_modeling_and_tuning.ipynb)
5. [Evaluation and error analysis](notebooks/05_evaluation_and_error_analysis.ipynb)

As implementation progresses, document the evaluation design, keep learned transformations within training folds, save notebook outputs, and record the exact working dependency versions.

## Initial audit findings

The supplied `hotel_bookings.csv` has **119,390 rows and 32 columns**. Key findings include **31,994 additional exact full-row copies**, **180 known zero-total-guest bookings**, and ADR values from **-6.38 to 5,400**. Parsed null shares are **94.31% for company** and **13.69% for agent**. These figures describe the original source. Step 5 has subsequently produced a separate eligible cohort; the raw file remains unchanged.

See [the audit report](report/data_audit.md), [measured audit record](data/audit_summary.json), and [quality figure](figures/01_data_quality_audit.png).

The audit's 10 code cells were executed sequentially with real outputs in a fresh Python process using IPython. A separate fresh-kernel Jupyter run remains a final verification task.

## Step 5 eligibility results

Step 5 retained **119,210 bookings with 29 candidate predictors**, excluded **180 known-zero-guest records**, and separated the target from the predictors. Both reservation-status columns are absent from predictors. Retained values are unchanged; anomalies are flagged and duplicates are grouped. Step 7 now fits separate preprocessing components within development training folds.

The four verified row-level processed files are published in `data/processed/`, alongside the code, aggregate results, and output hashes. Step 6 supplies frozen evaluation assignments, and Step 7 supplies reusable fold-fitted imputation, encoding, and scaling.

See the [Step 5 decision report](report/step5_eligibility.md), [executed Notebook 02](notebooks/02_preprocessing.ipynb), and [processed data instructions](data/processed/README.md). To regenerate from the original CSV:

```bash
python -m src.eligibility
python -m unittest discover -s tests -v
```

## Step 6 evaluation plan

The frozen chronological split uses **95,415 development bookings** (2015-07-01 through 2017-04-22) and **23,795 final test bookings** (2017-04-23 through 2017-08-31). Three expanding forward validation folds stay within development data. Whole dates and duplicate groups remain together; test rows never enter CV.

The verified row-level split file is published at `data/splits/step6_assignments.csv.gz`. Its checksum matches the frozen evaluation plan. The split code reproduces the same assignments from the committed Step 5 files; the aggregate plan, checksums, and timeline are also published.

See the [Step 6 report](report/step6_evaluation_plan.md), [split files and loading example](data/splits/README.md), and [evaluation timeline](figures/02_evaluation_timeline.png). The primary metric is mean cancellation-class F1 across the three validation folds. Features, preprocessing, model settings, and any threshold must be selected using development data only. Final test evaluation is reserved for Step 13.

```bash
python -m src.splitting
python -m unittest discover -s tests -v
```

The split command works with the committed processed data and rejects changes to a frozen plan. The Step 6 split remains unchanged. At Step 7, nineteen focused tests and all thirteen Notebook 02 code cells passed; no predictive model had yet been fitted or evaluated.

## Step 7 preprocessing

Step 7 uses **25 initial source fields** after excluding the sparse company ID and three fields with uncertain post-booking timing. Numeric medians, scaling statistics, and one-hot vocabularies are learned separately within each training fold. Negative ADR is treated as missing; zero/high positive prices remain and ADR is log-transformed. Unknown country and no-agent values remain distinct. Missing children/ADR receive fixed indicators.

All three train/validation pairs contain finite encoded values. Output widths are **328, 422, and 491** as the training vocabularies grow; each fold's training and validation widths match. Test rows are not fitted or transformed. Both scaled and tree-compatible unscaled variants passed checks.

See the [Step 7 report](report/step7_preprocessing.md), [summary and schemas](data/processed/step7/README.md), and [preprocessing factory](src/preprocessing.py). Reproduce from the committed data:

```bash
python -m src.preprocessing_audit
python -m unittest discover -s tests -v
```

Place the factory inside the model pipeline used for CV; do not prefit a transformer on all development rows. The first fold lacks several validation months, a documented limitation for Step 9 calendar-feature work. These policy exclusions do not replace Step 10 statistical feature selection/dimensionality reduction.

## Step 8 statistical analysis and EDA

Descriptive statistics and relationships use only the **95,415 development bookings**. There are **34,473 cancellations (36.13%)**. Fourteen aggregate tables, three relationship figures, and five supported observations are saved in [the EDA evidence](data/eda/README.md) and [report Sections 1–3](report/report.md).

- Cancellation rises from **9.43%** for lead times of 0–7 days to **80.81%** for 366+ days.
- Non Refund bookings have a **99.25%** cancellation rate, compared with **26.82%** for No Deposit. These are observed associations; the data do not establish a causal effect of deposit policy.
- Previous cancellations have a non-monotonic relationship with cancellation. Repeated profiles materially affect the rates: equal total weight per profile group changes the overall development rate to **25.26%**. This sensitivity analysis does not change the dataset, split, or model-training weights.

See [lead time](figures/03_lead_time_cancellation.png), [deposit type](figures/04_deposit_cancellation.png), and [previous cancellations and repetition](figures/05_prior_cancellations_sensitivity.png). All 22 focused tests pass. The five new Notebook 01 EDA cells have actual Python-generated outputs; its earlier ten audit cells retain their previously executed outputs. This session did not run all cells in a fresh Jupyter kernel or perform canonical nbformat validation because those packages were unavailable. Those checks remain final gates.

```bash
python -m src.development_eda
python -m unittest discover -s tests -v
```

## Step 9 derived features

Eight deterministic features cover total nights/guests, total previous bookings, history presence, previous cancellation share, company-code recording, and cyclic month coordinates. Month names are replaced with the sine/cosine pair. The explicit candidate schema contains **32 fields** before encoding; feature selection remains Step 10.

The three frozen folds produce **332, 421, and 490 encoded columns**, with matching train/validation widths and zero nonfinite values. Both scaled and tree-compatible unscaled variants pass training-only statistic checks. All **29 focused tests** pass. Notebook 03's five code cells executed sequentially in a fresh Python process with saved outputs; separate fresh-kernel Jupyter execution and canonical notebook validation remain final gates.

See [Notebook 03](notebooks/03_feature_engineering.ipynb), [Step 9 decisions and results](report/step9_feature_engineering.md), and [aggregate evidence and schemas](data/processed/step9/README.md). Run:

```bash
python -m src.feature_audit
python -m unittest discover -s tests -v
```

The four unknown guest totals stay missing before fold-fitted imputation; 604 zero-night bookings remain. No-history is distinguishable from an observed zero cancellation share. The company indicator means a code is recorded, not verified corporate payment. Calendar encoding does not establish seasonal generalization. No target values are read by the Step 9 audit and no test rows are fitted/transformed.

## Step 10 feature selection and dimensionality reduction

Four fixed representations were compared with the same logistic-regression reference across the three frozen forward folds. Supervised selection retains the top 75% of nonconstant training features by ANOVA F ranking. Centered PCA reduces only scaled numeric fields, keeping at least 95% of their training variance; categories remain sparse.

| Representation | Mean development F1 | Output widths by fold |
| --- | ---: | --- |
| All Step 9 features | 0.693094 | 332 / 421 / 490 |
| Selection only — current preference | **0.713609** | 247 / 314 / 366 |
| Numeric PCA | 0.694633 | 325 / 413 / 483 |
| Selection then PCA | 0.701023 | 242 / 308 / 360 |

Selection alone has the highest mean F1, although fold 3 is slightly worse than all features. PCA is demonstrated and documented but is not retained in the preferred representation. These are development selection scores, not final test performance or proof that this representation is best for every model family.

See [the Step 10 report](report/step10_selection_and_reduction.md), [evidence and full retained lists](data/processed/step10/README.md), and [PCA variance figure](figures/06_numeric_pca_variance.png). All **35 tests** pass; all 12 reference fits converged. Notebook 03 now contains ten executed Python cells (five Step 9 outputs preserved, five new Step 10 outputs). A full fresh Jupyter-kernel run and canonical format validation remain final gates.

```bash
python -m src.representation_audit
python -m unittest discover -s tests -v
```

Use a new `BookingRepresentation(mode="selected")` inside the model pipeline passed to CV. Fit preprocessing, ranking, selection and optional PCA only on training folds. Do not reuse a globally fitted matrix. No final full-development model is fitted and no held-out row is transformed or scored.

The Step 7 and Step 9 factories are preserved. Step 10 wraps the 32-field Step 9 schema with training-fold representation choices. Prediction-time validity is not guaranteed; the project retains its retrospective arrival-cohort scope.

## Step 11 baseline and two model families

Five candidates were compared on the same three frozen forward folds: a training-majority baseline and full/selected versions of logistic regression and random forest. Preprocessing and selection fit only within each training fold; threshold remains 0.5. This is an untuned development comparison.

| Candidate | Mean cancellation F1 |
| --- | ---: |
| Majority baseline | 0.000000 |
| Logistic regression — full | 0.693094 |
| Logistic regression — selected | **0.713609** |
| Random forest — full | 0.626504 |
| Random forest — selected | 0.657994 |

Selected-feature logistic regression is the current leader. Selection improves mean F1 for both learned families under these settings. The selected forest has higher ROC-AUC (0.892267) than selected logistic regression (0.882905), but lower cancellation recall (0.521714 versus 0.659498) at the fixed threshold. Its large training–validation gap warrants regularization experiments; these results do not establish a final best model.

All **40 tests pass** and all **15 model fits complete**. Logistic scores reproduce Step 10's published results. Notebook 04 has six sequentially executed Python cells with real outputs. Separate fresh Jupyter-kernel execution and canonical notebook validation remain final submission gates. No held-out row is fitted, transformed or scored; no model is refitted on all development rows.

See [Notebook 04](notebooks/04_modeling_and_tuning.ipynb), [the Step 11 comparison report](report/step11_model_comparison.md), [aggregate evidence](data/results/step11/README.md), and [comparison figure](figures/07_model_comparison.png). Reproduce with:

```bash
python -m src.model_comparison
python -m unittest discover -s tests -v
```

**Step 12 (Sadat)** has tuned both learned families using fresh pipelines from `make_model_pipeline`, the frozen forward CV indices, and mean cancellation F1. Search spaces and results are recorded below. The final test remains reserved for Step 13. Read [PROJECT_STATUS.md](PROJECT_STATUS.md) and [HANDOFF_TO_SADAT.md](HANDOFF_TO_SADAT.md).

Faraaz owns Steps 1–8; Sadat owns Steps 9–15. Assigned ownership is not a record of work already performed. Both members must review their work and record actual contributions.

## Step 12 hyperparameter tuning

An exhaustive grid evaluates **20 settings across three frozen folds (60 fits)**: logistic C/class weighting and forest depth/leaf size/class weighting. Selected features, threshold 0.5, seed 42 and the original split remain fixed. Every candidate uses a fresh train-fold pipeline; `refit=False` prevents a global refit.

| Family | Selected search parameters | Untuned F1 | Tuned F1 | Change |
| --- | --- | ---: | ---: | ---: |
| Logistic Regression | C=1.0, class_weight=balanced | 0.713609 | 0.732102 | +0.018492 |
| Random Forest | class_weight=balanced, max_depth=None, min_samples_leaf=10 | 0.657994 | 0.669294 | +0.011300 |

The chosen family is **Logistic Regression**, with **C=1.0, class_weight=balanced**, mean development F1 **0.732102**. This is the best setting in the declared grid, not a final test result. Both untuned controls match Step 11 under the documented numerical tolerances; all **47 tests pass** and all 60 fits finish without convergence warnings. Notebook 04 preserves six Step 11 code cells and appends five executed Step 12 cells. Full fresh-kernel/format verification remains a final gate.

See [the tuning report](report/step12_hyperparameter_tuning.md), [all results and frozen settings](data/results/step12/README.md), and [tuning figure](figures/08_hyperparameter_tuning.png).

```bash
python -m src.tuning
python -m unittest discover -s tests -v
```

## Step 13 final evaluation and error analysis

The frozen selected-feature Logistic Regression (`C=1`, balanced class weights,
threshold 0.5) was fitted on all 95,415 development rows and evaluated once on
23,795 later-arrival test bookings.

| F1 | Accuracy | Precision | Recall | ROC-AUC |
| ---: | ---: | ---: | ---: | ---: |
| **0.750592** | 0.760874 | 0.654187 | 0.880321 | 0.875977 |

Confusion counts are TN 9,543, FP 4,526, FN 1,164 and TP 8,562. The model
detects most cancellations but overpredicts them, so its high recall comes with
lower precision. City Hotel and Online TA contain many errors; short lead-time
bookings have substantially lower F1. Fixed-bin predicted probabilities exceed
observed rates, so they are not claimed to be calibrated. No test result changed
the model or threshold.

See [Notebook 05](notebooks/05_evaluation_and_error_analysis.ipynb), [the detailed
evaluation](report/step13_final_evaluation.md), [all evidence and error examples](data/results/step13/README.md),
[final performance](figures/09_final_test_performance.png), and [error diagnostics](figures/10_final_error_analysis.png).

```bash
python -m src.final_evaluation
python -m unittest discover -s tests -v
```

All **53 tests pass**. The complete fitted pipeline is saved at
`models/final_logistic_regression.joblib`. Notebook 05 has six executed
evidence-analysis cells. Full fresh-Jupyter, canonical format and clean-install
verification remain final submission gates.

Step 14 now synthesizes this frozen evidence below. Do not change the model,
threshold, dataset, target, problem statement or questions.

## Step 14 research-question answers

1. **Factors:** deposit type shows a large development separation (Non Refund
   99.25% canceled versus No Deposit 26.82%); lead-time rates increase from
   9.43% at 0–7 days to 80.81% at 366+ days. Prior cancellations are
   non-monotonic and sensitive to repeated-profile weighting. These are
   associations, not causal effects or an exhaustive feature ranking.
2. **Prediction:** the selected pipeline achieves held-out F1 0.750592,
   accuracy 76.09%, precision 65.42%, recall 88.03% and ROC-AUC 0.875977.
   It misses 1,164 cancellations and falsely flags 4,526 noncancellations.
   Performance is period-specific, and the probabilities are not calibrated.
3. **Best tested model:** selected-feature Logistic Regression with C=1 and
   balanced weights leads on mean development F1 (0.732102 versus the best
   tested forest's 0.669294). PCA was evaluated but not retained because
   selection alone performed better. This is best evaluated under this
   protocol, not universal superiority or a test-set ranking of all models.

See the [full answers with source-table links](report/step14_research_answers.md)
and [report Sections 7.3–8](report/report.md). Step 14 changes documentation only;
datasets, model settings, saved predictions, figures and notebook outputs stay
unchanged. No new model fitting or official test evaluation is performed.

**Next: Step 15 — Sadat, with both members reviewing.** Complete the template-
compliant report/PDF, raw-data/provenance requirements, clean-environment and
fresh-kernel checks, references and genuine contribution records.

## Remaining setup

- [x] Add approved group members, problem statement, and unchanged research questions.
- [x] Audit the supplied CSV and record source identity, dimensions, and quality findings.
- [x] Complete Step 5 eligibility, direct leakage removal, grouping, and reproducible output generation.
- [x] Complete Step 6 holdout, forward CV, metric commitments, and overlap checks.
- [x] Complete Step 7 fold-fitted preprocessing and before/after verification.
- [x] Complete Step 8 development-only EDA, three figures, report Sections 1–3 draft, and Sadat's handoff.
- [x] Complete Step 9 derived features, fold verification, Notebook 03 outputs, and report explanation.
- [x] Complete Step 10 supervised selection, centered numeric PCA, reference-model comparison and justified current representation.
- [x] Complete Step 11 majority baseline, two model families, full/selected controls, Notebook 04 outputs and report Section 5.
- [x] Complete Step 12 exhaustive tuning, all candidate/fold results, frozen settings, Notebook 04 outputs and report Section 6.
- [x] Complete Step 13 final test evaluation, saved model, predictions, error analysis, Notebook 05 outputs and report Section 7.
- [x] Complete Step 14 answers to all three unchanged questions, evidence links, limitations and Step 15 handoff.
- [ ] Confirm exact source terms and add the raw dataset to the repository.
- [ ] Record collaborators' actual contributions as work is completed.
- [ ] Verify all implemented notebooks in a fresh environment and fresh Jupyter kernels.
- [ ] Complete the report and export the final PDF.

## Assistance

OpenAI ChatGPT/Codex assisted with repository scaffolding, transcription of user-approved project details, audit/EDA, Steps 5–10 preprocessing and feature work, Steps 11–13 implementation, execution, tests, modeling/tuning/final evaluation and notebook outputs, and Step 14 evidence checking and research-question synthesis. Final submission work remains pending. Assigned ownership does not establish personal contributions; members must review and record their actual work.
