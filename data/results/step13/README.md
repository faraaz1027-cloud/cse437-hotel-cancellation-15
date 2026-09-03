# Step 13 final-evaluation evidence

The Step 12 selected-feature Logistic Regression pipeline (`C=1`, balanced
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
verification remain Step 15 gates.

```bash
python -m src.final_evaluation
python -m unittest discover -s tests -v
```

Regenerating the evidence with the frozen settings is a reproducibility check,
not permission to choose a new model from the test result.
