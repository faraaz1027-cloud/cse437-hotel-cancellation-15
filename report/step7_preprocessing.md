# Step 7 — Preprocessing inside training folds

## Applied decisions

| Item | Rule | Rationale |
| --- | --- | --- |
| Company identifier | Exclude from the initial model inputs | Sparse nominal ID; a presence indicator remains a Step 9 candidate |
| Assigned room, booking changes, waiting-list days | Exclude from initial model inputs | Potentially updated after reservation; conservative timing policy |
| Negative ADR | Mark value missing; retain booking | Nonnegative-price assumption, documented instead of deleting the row |
| Numeric missingness | Median fitted separately within each training fold | Avoid learning imputation statistics from validation/test data |
| Entirely missing training field | Keep the field with a zero fallback | Maintain schema; missing flags preserve children/ADR missingness |
| Missing country / ordinary category | Explicit `Unknown` value | Separate unknown information from a real category |
| Agent | Convert codes to nominal strings; null becomes `NoAgent` | Source NULL convention represents no agent; literal code 0 stays distinct |
| Existing `Undefined` values | Retain as their own category | Do not silently equate supplied labels with missing values |
| Skewed numeric fields | log1p for lead time, previous cancellations, previous noncanceled bookings, and ADR | Reduce numeric skew without deleting or clipping high positive values |
| Numeric scaling | Training-fold StandardScaler after imputation/log1p | Suitable numeric magnitudes for linear models; optional unscaled tree variant |
| Categories | Sparse one-hot encoding, training vocabulary only | Unseen values become an all-zero block for that field without refitting |
| Missing indicators | Always append children and ADR missing flags | Distinguish imputed from observed values, including validation-only missingness |

The target and both reservation-status columns remain excluded. Split metadata
and identifiers are rejected if passed as predictors. No rows are removed at
Step 7, no extra duplicate policy is introduced, and source data are unchanged.

The 25 retained source fields include all three substantive factors committed
before modeling: lead time, deposit type, and previous cancellations. The four
policy exclusions do not fulfill the separate statistical selection and
dimensionality-reduction requirements; those remain Step 10 work.

## Before/after evidence

| Fold | Training rows | Validation rows | Encoded columns | Children training median | ADR training median | Nonfinite encoded values |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 23,797 | 23,893 | 328 | 0 | 76.50 | 0 |
| 2 | 47,690 | 23,776 | 422 | 0 | 80.75 | 0 |
| 3 | 71,466 | 23,949 | 491 | 0 | 90.00 | 0 |

Four child values are missing in each expanding training window. They are
imputed to its observed median of zero and retain a missing flag. There are no
missing child values in the validation windows. The single negative ADR occurs
in fold 3 validation: it becomes missing and receives the **training** median
of 90 before log transformation/scaling. No training window contains that
negative value.

| Fold | Training country unknowns | Validation country unknowns | Training no-agent values | Validation no-agent values |
| --- | ---: | ---: | ---: | ---: |
| 1 | 158 | 123 | 3,471 | 3,359 |
| 2 | 281 | 72 | 6,830 | 2,434 |
| 3 | 353 | 120 | 9,264 | 4,481 |

All zero ADR values are retained. The ADR of 5,400 occurs in fold 1 validation
and subsequently enters training in later folds; log1p(5400) is approximately
8.594. No validation-derived cap is fitted. Counts across expanding folds
overlap and must not be added as if they represented distinct bookings.

## Category changes and limitations

Train and validation use the same encoding schema within each fold. Schemas
differ between folds because the vocabularies grow with the training period.

| Fold | Validation rows with unseen month | Unseen country | Unseen market segment | Unseen agent |
| --- | ---: | ---: | ---: | ---: |
| 1 | 23,475 | 114 | 62 | 1,531 |
| 2 | 0 | 31 | 0 | 349 |
| 3 | 0 | 15 | 0 | 613 |

The first training window covers only part of a calendar year. One-hot encoding
cannot learn category-specific effects for months absent from training. The
all-zero handling is safe computationally but loses that field's category
information; other date fields remain. Record this seasonal limitation and
consider cyclic calendar features in Step 9 without changing the split.

Excluding potentially updated fields reduces an identifiable timing risk; it
does not establish that every remaining column is available at booking
creation. The dataset still supports the previously declared retrospective
arrival-cohort evaluation. No booking-time deployment validity is claimed.

Medians, log transformations, and standardization are documented initial
choices. Their predictive advantage is not established until later modeling.

## Implementation and checks

- [Preprocessing factory](../src/preprocessing.py) returns a fresh, cloneable
  sklearn pipeline with fixed domain rules and learned column transforms.
- [Development-fold audit](../src/preprocessing_audit.py) uses the frozen Step 6
  indices and exports aggregate results and per-fold feature schemas.
- [Summary and schemas](../data/processed/step7/README.md) record versions,
  input hashes, missing counts, dimensions, and fitted medians.
- [Executed Notebook 02](../notebooks/02_preprocessing.ipynb) contains the full
  Steps 5–7 sequence with saved outputs.

For every fold, checks compare learned medians against the training data,
verify scaler means and fitted row counts, and compare encoder vocabularies to
training categories. Validation transformation leaves fitted state unchanged.
Both scaled and unscaled variants produce finite matrices with consistent
within-fold dimensions. The audit does not read the target file and does not
fit or transform any final test rows.

Eight new tests cover train-only statistics, unseen categories, fixed domain
rules and source preservation, excluded-field independence, accidental target/
metadata inputs, entirely missing fields, cloneable scaled/unscaled variants,
and invalid numeric codes/counts. All **19 tests** including prior stages pass.
Notebook 02's **13 code cells** execute sequentially in a fresh Python process
using IPython with real outputs. A separate fresh Jupyter-kernel run remains
part of final submission verification.

The exact run used Python 3.12.13, pandas 2.2.3, NumPy 2.3.5, SciPy 1.17.0,
and scikit-learn 1.8.0. Starter dependency ranges remain in requirements.txt;
clean-install dependency validation is a final reproducibility task.

## Reproduction

From the repository root:

```bash
python -m src.preprocessing_audit
python -m unittest discover -s tests -v
```

Keep `make_preprocessor()` inside the estimator pipeline passed to CV. Do not
export a single globally imputed/scaled matrix and then cross-validate it.
Step 9 features and Step 10 selection/reduction must also remain within the
fold-safe pipeline where they learn from data. Refit on all development rows
only after model settings are selected; reserve the test for Step 13.

Step 8 should now produce development-only descriptive statistics, plots, and
observations relevant to the original questions. The original CSE437 document,
approved question wording, Step 5 files, and Step 6 assignments are unchanged.

Source semantics reference: Antonio, N., de Almeida, A., and Nunes, L. (2019),
*Hotel booking demand datasets*, https://doi.org/10.1016/j.dib.2018.11.126;
https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/ .
