# Step 15 — authorized reporting-only test comparison

Owner: Sadat, with both members reviewing. This supplement is complete; overall
Step 15 and final submission are not yet complete.

## Timing and fixed choices

The user approved adding a majority baseline and Random Forest test comparison
after the Step 13 Logistic Regression test outcome was known. Consequently,
this is not described as a preregistered simultaneous three-model evaluation.
The supplement protocol was saved before fitting the two added estimators.

The baseline learns the development majority (class 0). Random Forest uses
Step 12's development-selected `rf_12`: selected representation, 100 trees,
minimum leaf size 10, unlimited depth, balanced weights, sqrt feature sampling,
seed 42. Both fit only 95,415 frozen development rows and score the identical
23,795 held-out bookings. All learned preprocessing and feature selection fit
development data only. Threshold stays 0.5. The final Logistic Regression model
remains selected based on development F1; its saved predictions are reused.

## Results

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

## Evidence and reproduction

- [Frozen protocol](comparison_protocol.json)
- [Full metrics, confusion counts and Brier scores](test_comparison.csv)
- [Random Forest probabilities with aligned source row IDs and actual labels](random_forest_test_probabilities.csv.gz)
- [Hash lineage and runtime](comparison_summary.json)
- [Implementation](../../../src/test_comparison.py)
- [Tests](../../../tests/test_test_comparison.py)
- [Original Logistic Regression evidence](../step13/README.md)

From the repository root, run `python -m src.test_comparison`. If the saved
supplement exists, this verifies its hashes and returns cached results without
repeating test evaluation. To independently reproduce fitting, use an isolated
copy without its Step 15 outputs; never remove the original evidence. Requires
the same scientific runtime recorded in `comparison_summary.json` and the
committed Step 5/6/12/13 artifacts. No raw dataset changes are made.

The current suite (`python -m unittest discover -s tests -q`) passes 58 tests.
All original Step 13 output hashes and the Step 12 selection hash were checked
before and after this run. Notebook 05's added evidence-check cell was executed
in Python with its output saved; full clean-environment/fresh-Jupyter-kernel
verification remains a separate open submission gate.

Original acquisition date and acquired dataset version are unknown (user
confirmed). Final report/PDF, raw-data publication/licence evidence,
reproducibility checks and genuine member contribution records remain open.
ChatGPT/Codex assisted in implementation, execution, checking and documentation.
