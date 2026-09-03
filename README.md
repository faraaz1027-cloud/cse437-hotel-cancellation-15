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

| Notebook | Contents | Google Colab |
| --- | --- | --- |
| [01 — Data audit and EDA](notebooks/01_data_audit_and_eda.ipynb) | Raw quality audit; development-only descriptive analysis | [Open in Colab](https://colab.research.google.com/github/faraaz1027-cloud/cse437-hotel-cancellation-15/blob/main/notebooks/01_data_audit_and_eda.ipynb) |
| [02 — Preprocessing](notebooks/02_preprocessing.ipynb) | Eligibility, frozen splits, fold-fitted preprocessing | [Open in Colab](https://colab.research.google.com/github/faraaz1027-cloud/cse437-hotel-cancellation-15/blob/main/notebooks/02_preprocessing.ipynb) |
| [03 — Feature engineering](notebooks/03_feature_engineering.ipynb) | Derived features, selection and PCA comparison | [Open in Colab](https://colab.research.google.com/github/faraaz1027-cloud/cse437-hotel-cancellation-15/blob/main/notebooks/03_feature_engineering.ipynb) |
| [04 — Modeling and tuning](notebooks/04_modeling_and_tuning.ipynb) | Model comparisons and hyperparameter search | [Open in Colab](https://colab.research.google.com/github/faraaz1027-cloud/cse437-hotel-cancellation-15/blob/main/notebooks/04_modeling_and_tuning.ipynb) |
| [05 — Evaluation and error analysis](notebooks/05_evaluation_and_error_analysis.ipynb) | Frozen test results, subgroup errors and supplementary comparison | [Open in Colab](https://colab.research.google.com/github/faraaz1027-cloud/cse437-hotel-cancellation-15/blob/main/notebooks/05_evaluation_and_error_analysis.ipynb) |

### Run in Google Colab

1. Open a link above, connect to a CPU runtime, and choose **Runtime > Run all**.
2. The first code cell installs the seven pinned analysis packages, downloads the project with its data/model, verifies protected checksums and sets the working directory. No manual data upload or Google Drive mount is needed.
3. If it reports **SETUP PAUSED**, choose **Runtime > Restart session**, then **Run all** again. This is needed only when incompatible packages were already loaded in memory. Do not delete the runtime. Installation failures stop execution; do not skip the setup cell.
4. Use Python 3.12 or 3.13; the recorded reference environment is Python 3.12. If Colab offers neither, use the local Python 3.12 setup above. GPU is unnecessary. Notebook 04 may take several minutes.
5. Use **File > Download > Download .ipynb** to retain your executed outputs. Opening the public notebook does not automatically publish any changes to GitHub.

The automatic download is pinned to analysis snapshot `e01e785b78f2849b423c5be4a3fe5221a96f3e66`, not whatever happens to be on main later. The setup wrapper was added after that snapshot; the original analysis cells are unchanged. Each notebook can access its committed inputs independently. Setup reuses an existing project directory without resetting it and is a no-op outside Colab. It does not install or replace Colab's Jupyter/IPython software.

Bootstrap logic has local regression tests; a live Colab run of these updated notebooks remains to be verified. Setup does not guarantee identical numerical scores or remove the documented reproduction warning. Existing outputs retain historical provenance; new setup cells have no fabricated execution results. The shared bootstrap source is [scripts/colab_setup.py](scripts/colab_setup.py), embedded identically in all five notebooks.

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
