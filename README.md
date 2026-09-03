# Hotel Booking Cancellation Project

**CSE437: Data Science | Section 05 | Summer 2026 | Group 15**

Read the [technical report](report/report.pdf), its [Markdown source](report/report.md), and [reproducibility notes](FINAL_CHECKS.md).

## Group members

| Member | Student ID |
| --- | --- |
| Faraaz Jamil Chowdhury | 24241205 |
| Ihfaz Rashid Sadat | 23301499 |

## Problem statement

Hotel booking cancellations can cause loss of money and make room planning difficult for hotels. The goal of this project is to study the factors that affect booking cancellations and build a machine-learning model that can predict whether a hotel booking will be canceled.

## Research questions

1. Which booking and customer-related factors have the biggest effect on hotel cancellations?
2. How accurately can machine-learning models predict whether a hotel booking will be canceled?
3. Which machine-learning model gives the best result after data preprocessing, feature selection, dimensionality reduction, and hyperparameter tuning?

## Dataset and methods

The source contains 119,390 rows and 32 columns. Excluding 180 known zero-total-guest bookings leaves 119,210 observations: 95,415 development rows and 23,795 later-arrival test rows. The target is `is_canceled`. See [data provenance and acquisition](data/README.md).

The analysis preserves raw data, excludes direct outcome-leakage fields, keeps duplicate-profile groups together and fits learned transformations within training folds. It compares a majority-class baseline, Logistic Regression and Random Forest. Feature selection and numeric PCA are evaluated on development data; hyperparameters are selected using three forward validation folds.

## Setup

Use Python 3.12:

```bash
git clone https://github.com/faraaz1027-cloud/cse437-hotel-cancellation-15.git
cd cse437-hotel-cancellation-15
python -m venv .venv
```

Activate with `source .venv/bin/activate` on macOS/Linux or `.venv\Scripts\Activate.ps1` on Windows PowerShell.

```bash
python -m pip install -r requirements.txt
python -m jupyterlab
```

Place the untouched source at `data/raw/hotel_bookings.csv`, following [data/README.md](data/README.md). Raw-data publication is reported complete; its independent recheck is deferred. Use repository-relative paths. The lock file records the original Linux/Python 3.12 environment.

## Notebook order

| Notebook | Contents |
| --- | --- |
| [01 — Data audit and EDA](notebooks/01_data_audit_and_eda.ipynb) | Raw quality audit; development-only descriptive analysis |
| [02 — Preprocessing](notebooks/02_preprocessing.ipynb) | Eligibility, frozen splits, fold-fitted preprocessing |
| [03 — Feature engineering](notebooks/03_feature_engineering.ipynb) | Derived features, selection and PCA comparison |
| [04 — Modeling and tuning](notebooks/04_modeling_and_tuning.ipynb) | Model comparisons and hyperparameter search |
| [05 — Evaluation and error analysis](notebooks/05_evaluation_and_error_analysis.ipynb) | Frozen test results, subgroup errors and supplementary comparison |

Use `python scripts/verify_notebooks.py` for isolated reproduction. Notebook 04 also creates its own external development workspace, so new comparison/tuning outputs cannot overwrite the original frozen selection. Notebook 05 verifies cached final evaluation only. Do not select settings from test outcomes.

## Recorded results

The development-selected Logistic Regression uses `C=1`, balanced class weights and threshold 0.5. It achieves test F1 **0.750592**, accuracy **0.760874**, precision **0.654187**, recall **0.880321** and ROC-AUC **0.875977**.

Random Forest achieves test F1 **0.723115**; the majority baseline has F1 **0**. The baseline/Random Forest test comparison was added after the Logistic Regression test result was known. It did not change model selection, parameters or threshold. See the report for all metrics, errors and limitations.

## Repository contents

- `src/` and `tests/`: reusable implementations and unit tests.
- `data/processed/`, `data/splits/`, `data/eda/`, `data/results/`: documented preparation and measured evidence.
- `models/` and `figures/`: saved pipeline and analysis figures.
- `report/`: technical report and detailed method records.
- `scripts/`: report rendering and verification utilities.

## Reproducibility status

The supplied Step 15.3 Windows run passed dependency checks, 70 unit tests and all five fresh-kernel notebooks with zero cell errors. Existing published notebook outputs retain their original execution provenance; the supplied executed copies are separate verification evidence.

A previous audit failed notebook 05 with **`Frozen Step 12 model settings changed.`** The repaired workflow passed execution by isolating new development evidence from frozen evaluation, without bypassing integrity checks. The rerun still selected balanced LR with C=0.1 instead of C=1; the numerical cause is unconfirmed. Final-test evidence was checked from cache, not retrained. The supplied archive does not establish the full checkout's exact commit. See the [verification review](report/verification_review.md) and [repair notes](report/reproducibility_repair.md). Historical verification records describe their original runs, not the currently edited files.

Both members' assigned responsibilities are user-confirmed and recorded in the report. The author-supplied AI declaration is provisional. Final declaration review, raw-publication recheck, attributable commit checks and joint submission review remain pending. See [FINAL_CHECKS.md](FINAL_CHECKS.md). This repository is not yet declared submission-ready.
