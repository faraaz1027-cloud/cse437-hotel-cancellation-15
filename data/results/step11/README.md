# Step 11 model-comparison evidence

| File | Contents |
| --- | --- |
| `comparison_protocol.json` | Five candidates, fixed settings, representations, metrics, threshold and decision rule |
| `fold_results.csv` | All 15 fits: validation metrics/confusion counts, training diagnostics, runtime, capacity and membership checks |
| `model_comparison.csv` | Unweighted three-fold mean metrics and descriptive F1 SD |
| `feature_schemas.json` | Output names for each trained pipeline; baseline uses no features |
| `estimator_parameters.json` | Full actual estimator settings, including defaults |
| `model_summary.json` | Untuned leader, scope, runtime, frozen input and output hashes |

Current leader: **Logistic regression — selected**, mean validation F1 **0.713609**.
This is a development choice for further tuning, not the final test result.
Baseline plus two learned families are implemented; each learned family compares
full and selected features. All 40 tests pass; logistic metrics match Step 10.

The audit uses only development features and labels. Every pipeline fits its
preprocessing and selection inside its own training fold. The final test is
neither transformed nor scored. Row-level predictions and globally fitted
models are not exported. Measured timings can vary between executions; output
hashes identify the corresponding saved run.

```bash
python -m src.model_comparison
python -m unittest discover -s tests -v
```

Run from the repository root using the committed processed inputs. Notebook 04
has six sequentially executed Python cells with saved output. A fresh Jupyter-
kernel run and canonical notebook validation remain final submission checks.

See [the Step 11 report](../../../report/step11_model_comparison.md) and
[comparison figure](../../../figures/07_model_comparison.png). Next, tune both
learned model families through fresh `make_model_pipeline` instances inside CV.
