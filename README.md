# Hotel Booking Cancellation Project

**CSE437: Data Science | Section 05 | Summer 2026 | Group 15**

[Report PDF](report/report.pdf) · [Report Markdown](report/report.md) · [Data and detailed methods](data/README.md)

## Group members

| Member | Student ID | Contribution |
| --- | --- | --- |
| Faraaz Jamil Chowdhury | 24241205 | Planning, provenance, audit, preprocessing, validation design and descriptive analysis |
| Ihfaz Rashid Sadat | 23301499 | Features, selection/PCA, modeling, tuning, evaluation, verification and report assembly |

## Problem statement

Hotel booking cancellations can cause loss of money and make room planning difficult for hotels. The goal of this project is to study the factors that affect booking cancellations and build a machine-learning model that can predict whether a hotel booking will be canceled.

## Research questions

1. Which booking and customer-related factors have the biggest effect on hotel cancellations?
2. How accurately can machine-learning models predict whether a hotel booking will be canceled?
3. Which machine-learning model gives the best result after data preprocessing, feature selection, dimensionality reduction, and hyperparameter tuning?

## Dataset and approach

Hotel Booking Demand contains 119,390 rows and 32 columns; the binary target is `is_canceled`. Excluding 180 confirmed zero-guest bookings leaves 119,210 observations: 95,415 development records and 23,795 later-arrival test records. The untouched CSV is committed in `data/raw/`; source, licence, acquisition instructions and checksum are in [data/README.md](data/README.md).

Chronological, duplicate-profile-separated validation and training-fitted preprocessing guard against leakage. A majority baseline, Logistic Regression and Random Forest are compared. Feature selection and numeric PCA are demonstrated; tuning uses three forward development folds. No model or threshold was selected from test results.

## Repository layout

| Location | Contents |
| --- | --- |
| `data/raw/` | Untouched hotel-booking CSV |
| `data/processed/` | Eligible data, split membership, statistics, model evidence, environment and verification records |
| `data/README.md` | Source information and consolidated detailed methods |
| `notebooks/` | Five notebooks numbered in execution order |
| `src/` | Reusable modules, `utils.py`, support tools and regression tests |
| `models/` | Original saved model |
| `figures/` | Report and notebook figures |
| `report/` | `report.md` and the 10-page `report.pdf` |

## Notebooks and Google Colab

| Order | Notebook | Google Colab |
| --- | --- | --- |
| 01 | [Data audit and EDA](notebooks/01_data_audit_and_eda.ipynb) | [Open in Colab](https://colab.research.google.com/github/faraaz1027-cloud/cse437-hotel-cancellation-15/blob/main/notebooks/01_data_audit_and_eda.ipynb) |
| 02 | [Preprocessing](notebooks/02_preprocessing.ipynb) | [Open in Colab](https://colab.research.google.com/github/faraaz1027-cloud/cse437-hotel-cancellation-15/blob/main/notebooks/02_preprocessing.ipynb) |
| 03 | [Feature engineering](notebooks/03_feature_engineering.ipynb) | [Open in Colab](https://colab.research.google.com/github/faraaz1027-cloud/cse437-hotel-cancellation-15/blob/main/notebooks/03_feature_engineering.ipynb) |
| 04 | [Modeling and tuning](notebooks/04_modeling_and_tuning.ipynb) | [Open in Colab](https://colab.research.google.com/github/faraaz1027-cloud/cse437-hotel-cancellation-15/blob/main/notebooks/04_modeling_and_tuning.ipynb) |
| 05 | [Evaluation and error analysis](notebooks/05_evaluation_and_error_analysis.ipynb) | [Open in Colab](https://colab.research.google.com/github/faraaz1027-cloud/cse437-hotel-cancellation-15/blob/main/notebooks/05_evaluation_and_error_analysis.ipynb) |

In Colab choose a CPU runtime and **Runtime > Run all**. Setup downloads the full project, verifies protected inputs and installs pinned analysis packages. If it reports **SETUP PAUSED**, choose **Runtime > Restart session**, then **Run all** again; do not delete the runtime. No Drive mount or manual upload is needed. Setup supports Python 3.12/3.13; the recorded reference uses 3.12. Complete live Colab execution of the reorganized notebooks is not yet verified.

Setup prints the exact source revision used and does not overwrite a different checkout. Each notebook can access committed inputs independently. Notebook 04 isolates new development results; notebook 05 verifies cached final-test evidence without retraining.

## Recorded results

| Model | Test F1 | Accuracy | Precision | Recall | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Majority baseline | 0.000000 | 0.591259 | 0.000000 | 0.000000 | 0.500000 |
| Logistic Regression | 0.750592 | 0.760874 | 0.654187 | 0.880321 | 0.875977 |
| Random Forest | 0.723115 | 0.789325 | 0.781239 | 0.673041 | 0.878190 |

Selected Logistic Regression uses C=1, balanced weights and threshold 0.5. The baseline/Random Forest test comparison was added after Logistic Regression test results were known. Its timing is disclosed and did not change model selection. See the report for all metrics and limitations.
