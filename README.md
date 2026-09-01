# Hotel Booking Cancellation Project

**CSE437: Data Science | Group 15**

**Status: initial project structure.** The notebooks and report are placeholders; data analysis and model training have not been implemented.

## Project

This repository is being prepared for a hotel-booking cancellation data science project. The team details, project brief, research questions, and dataset metadata will be added after their public inclusion is confirmed.

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

The CSV has not been added to this repository. Record its source, licence, and acquisition instructions before adding data. Preserve original data and use repository-relative paths.

## Notebook order

These files are starter outlines, not completed analyses:

1. [Data audit and EDA](notebooks/01_data_audit_and_eda.ipynb)
2. [Preprocessing](notebooks/02_preprocessing.ipynb)
3. [Feature engineering](notebooks/03_feature_engineering.ipynb)
4. [Modeling and tuning](notebooks/04_modeling_and_tuning.ipynb)
5. [Evaluation and error analysis](notebooks/05_evaluation_and_error_analysis.ipynb)

As implementation progresses, document the evaluation design, keep learned transformations within training folds, save notebook outputs, and record the exact working dependency versions.

## Remaining setup

- [ ] Add confirmed public project details and research questions.
- [ ] Document and add the dataset as permitted by its source terms.
- [ ] Record collaborators' actual contributions as work is completed.
- [ ] Implement the notebooks and verify a fresh-environment run.
- [ ] Complete the report and export the final PDF.

## Assistance

OpenAI ChatGPT/Codex assisted with this repository scaffold. No analysis results are claimed by these starter files.
