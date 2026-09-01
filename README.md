# Hotel Booking Cancellation Project

**CSE437: Data Science | Group 15**

**Status: initial raw-data audit completed.** Notebook 01 contains executed quality checks and saved outputs. Development-only EDA, cleaning, final splits, model training, and the final report remain pending.

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

Notebook 01 contains the completed raw-data audit; its development-only relationship analysis is still pending. Notebooks 02-05 remain starter outlines:

1. [Data audit and EDA](notebooks/01_data_audit_and_eda.ipynb)
2. [Preprocessing](notebooks/02_preprocessing.ipynb)
3. [Feature engineering](notebooks/03_feature_engineering.ipynb)
4. [Modeling and tuning](notebooks/04_modeling_and_tuning.ipynb)
5. [Evaluation and error analysis](notebooks/05_evaluation_and_error_analysis.ipynb)

As implementation progresses, document the evaluation design, keep learned transformations within training folds, save notebook outputs, and record the exact working dependency versions.

## Initial audit findings

The supplied `hotel_bookings.csv` has **119,390 rows and 32 columns**. Key findings include **31,994 additional exact full-row copies**, **180 known zero-total-guest bookings**, and ADR values from **-6.38 to 5,400**. Parsed null shares are **94.31% for company** and **13.69% for agent**. No cleaning has been applied.

See [the audit report](report/data_audit.md), [measured audit record](data/audit_summary.json), and [quality figure](figures/01_data_quality_audit.png).

The audit's 10 code cells were executed sequentially with real outputs in a fresh Python process using IPython. A separate fresh-kernel Jupyter run remains a final verification task.

## Remaining setup

- [x] Add approved group members, problem statement, and unchanged research questions.
- [x] Audit the supplied CSV and record source identity, dimensions, and quality findings.
- [ ] Confirm exact source terms and add the raw dataset to the repository.
- [ ] Record collaborators' actual contributions as work is completed.
- [ ] Implement the notebooks and verify a fresh-environment run.
- [ ] Complete the report and export the final PDF.

## Assistance

OpenAI ChatGPT/Codex assisted with repository scaffolding, transcription of user-approved project details, and the initial data-audit code, execution, figures, and interpretation. The audit reports computed data-quality findings; predictive model results have not been produced.

