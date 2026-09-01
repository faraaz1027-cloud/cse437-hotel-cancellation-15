# Hotel Booking Cancellation Project

**CSE437: Data Science | Group 15**

**Status: Step 6 completed — chronological holdout and forward validation frozen.** Responsible: **Faraaz**. **Next: Step 7 — fitted preprocessing.** Notebook 01 contains the raw audit; Notebook 02 contains executed Steps 5–6. Fitted preprocessing, development-only EDA, modeling, and the final report remain pending.

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
| `data/README.md` | Dataset acquisition and provenance |
| `notebooks/` | Numbered analysis notebooks |
| `models/` | Saved models and fitted pipelines |
| `figures/` | Exported figures |
| `report/report.md` | Draft report placeholder |
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

Notebook 01 contains the completed raw-data audit; its development-only relationship analysis is still pending. Notebook 02 implements Steps 5–6; fitted preprocessing is pending. Notebooks 03-05 remain starter outlines:

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

Step 5 retained **119,210 bookings with 29 candidate predictors**, excluded **180 known-zero-guest records**, and separated the target from the predictors. Both reservation-status columns are absent from predictors. Retained values are unchanged; anomalies are flagged and duplicates are grouped. No learned transformations have been fitted.

The four verified row-level processed files are published in `data/processed/`, alongside the code, aggregate results, and output hashes. Step 6 now supplies separate frozen evaluation assignments; imputation, encoding, and scaling remain pending.

See the [Step 5 decision report](report/step5_eligibility.md), [executed Notebook 02](notebooks/02_preprocessing.ipynb), and [processed data instructions](data/processed/README.md). To regenerate from the original CSV:

```bash
python -m src.eligibility
python -m unittest discover -s tests -v
```

## Step 6 evaluation plan

The frozen chronological split uses **95,415 development bookings** (2015-07-01 through 2017-04-22) and **23,795 final test bookings** (2017-04-23 through 2017-08-31). Three expanding forward validation folds stay within development data. Whole dates and duplicate groups remain together; test rows never enter CV.

The row-level split file has been generated and verified locally; its public upload awaits explicit permission. The published split code reproduces the exact assignments from the already committed Step 5 files. The aggregate plan, checksums, and timeline are published.

See the [Step 6 report](report/step6_evaluation_plan.md), [split files and loading example](data/splits/README.md), and [evaluation timeline](figures/02_evaluation_timeline.png). The primary metric is mean cancellation-class F1 across the three validation folds. Features, preprocessing, model settings, and any threshold must be selected using development data only. Final test evaluation is reserved for Step 13.

```bash
python -m src.splitting
python -m unittest discover -s tests -v
```

The split command works with the committed processed data and rejects changes to a frozen plan. Eleven focused tests and all nine Notebook 02 code cells passed. No predictive model has been fitted or evaluated.

**Next: Step 7 (Faraaz)** implements preprocessing inside each training fold. Faraaz owns Steps 1–8; Sadat owns Steps 9–15. Assigned ownership is not a record of work already performed. Both members must review their work and record actual contributions.

The 29 fields remain candidate features; prediction-time availability and the final feature set still require review. A fresh separate Jupyter-kernel run remains part of final submission verification.

## Remaining setup

- [x] Add approved group members, problem statement, and unchanged research questions.
- [x] Audit the supplied CSV and record source identity, dimensions, and quality findings.
- [x] Complete Step 5 eligibility, direct leakage removal, grouping, and reproducible output generation.
- [x] Complete Step 6 holdout, forward CV, metric commitments, and overlap checks.
- [ ] Confirm exact source terms and add the raw dataset to the repository.
- [ ] Record collaborators' actual contributions as work is completed.
- [ ] Implement the notebooks and verify a fresh-environment run.
- [ ] Complete the report and export the final PDF.

## Assistance

OpenAI ChatGPT/Codex assisted with repository scaffolding, transcription of user-approved project details, the initial data-audit code, execution, figures, and interpretation, and Steps 5–6 policy documentation, implementation, execution, and verification. The audit reports computed data-quality findings; predictive model results have not been produced.
