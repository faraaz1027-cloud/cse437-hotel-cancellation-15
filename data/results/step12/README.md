# Step 12 tuning evidence

**Owner: Sadat | Step 12 complete | Next: Step 13**

| File | Evidence |
| --- | --- |
| `search_protocol.json` | Predeclared grids, scoring, threshold, budget, tie rule and no-refit policy |
| `candidate_parameters.json` | Search parameters and complete estimator settings for all 20 candidates |
| `candidate_results.csv` | All candidate means, sample fold SD, training F1, gap and runtime |
| `fold_results.csv` | All 60 fits: five metrics, confusion counts, training F1 and fold membership |
| `logistic_regression_cv_results.csv` | Complete sklearn CV results for eight logistic settings |
| `random_forest_cv_results.csv` | Complete sklearn CV results for twelve forest settings |
| `tuning_comparison.csv` | Best setting per family versus its Step 11 untuned control |
| `control_parity.csv` | Exact Step 11/control differences and numerical-check tolerances |
| `final_selection.json` | Development-selected settings and unfitted Step 13 handoff |
| `tuning_summary.json` | Scope, checks, runtime, source/input/output checksums and limitations |

Selected: **Logistic Regression**, `C=1.0, class_weight=balanced`, mean development F1 **0.732102**.
Both families use selected features and threshold 0.5. All 60 fits complete;
untuned controls reproduce Step 11. These are reused-development scores, not
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
See [the detailed report](../../../report/step12_hyperparameter_tuning.md)
and [Notebook 04](../../../notebooks/04_modeling_and_tuning.ipynb).
Its five new cells have actual Python outputs; full fresh-Jupyter and canonical
format verification remain final gates.
