# Step 9 feature evidence

**Owner: Sadat. Step 9 complete; Step 10 is next.**

- `feature_summary.json`: measured development-only diagnostics, fold checks,
  input/output SHA-256 hashes, runtime, and limitations.
- `feature_schemas.json`: full encoded feature names for each frozen fold.
- `derived_feature_statistics.csv`: descriptive values of the eight fixed-formula
  features across 95,415 development rows, before learned imputation/scaling.

No target values are read by this audit and no test row is fitted/transformed.
No source rows, split assignments, or training weights change. Aggregate evidence
does not substitute for a final selected feature set or predictive comparison.

The schema has 32 fields before encoding (24 retained source fields plus eight
derived fields). Month names are replaced by a fixed sine/cosine pair. Five
fixed missing indicators follow encoding. Folds produce 332, 421 and 490 columns.
Both scaled and unscaled variants are checked, with training-only statistics.

Run from the repository root:

```bash
python -m src.feature_audit
python -m unittest discover -s tests -v
```

The commands work with committed processed files. Notebook 03 contains five
actually executed Python cells and their saved output. A separate fresh Jupyter
kernel run and canonical nbformat validation remain final checks.

Read [the Step 9 report](../../../report/step9_feature_engineering.md) for formulas,
missing-value semantics, known limitations, and integration instructions. For
later CV, use `make_feature_preprocessor()` inside the estimator pipeline and
keep fold indices relative to development rows. Do not reuse globally fitted
transformations. The original Step 7 factory is preserved for comparison.
