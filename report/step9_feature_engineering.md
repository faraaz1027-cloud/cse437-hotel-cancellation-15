# Step 9 — Derived features

Step 9 adds a reusable deterministic feature transformer and a preprocessing
factory for the unchanged hotel-cancellation project. The target, approved
questions, original document, eligible cohort, and frozen evaluation split are
preserved. The audit reads no target values and fits/transforms no test rows.

## Feature choices and justification

| Feature | Formula and missing-value rule | Reason to evaluate later |
| --- | --- | --- |
| `total_nights` | Weekend + weekday nights; missing component means missing total; zero stays retained | Reservation duration in one field |
| `total_guests` | Adults + children + babies; missing component means missing total | Booking party size; does not interpret unknown children as zero |
| `previous_bookings_total` | Previous canceled + previous noncanceled bookings | Amount of recorded history |
| `has_booking_history` | 1 when history total > 0; 0 when zero; missing when unknown | Distinguishes no history from observed history |
| `previous_cancellation_share` | Previous cancellations / history total; set to zero only for known zero total; unknown stays missing | Relative cancellation history, interpreted with history presence |
| `company_code_recorded` | 1 for a non-null code, 0 for null; literal code 0 is present | Coarse recording indicator without sparse company ID categories |
| `arrival_month_sin` | sin(2π(m−1)/12), m=1…12 | Fixed seasonal coordinate with December next to January |
| `arrival_month_cos` | cos(2π(m−1)/12), m=1…12 | Completes the cyclic representation |

The company flag does not prove corporate payment, sponsorship, or availability
at booking creation. It is explicitly a code-recording proxy. The source month
name is replaced by the cyclic pair; remaining Step 7 source predictors stay
available for later selection. Unknown month labels and invalid company codes
raise errors rather than silently producing arbitrary values.

The resulting candidate schema contains **24 retained source fields + eight
derived fields = 32 fields** before encoding. These choices are justified by
interpretability and representation; they are not claimed to improve prediction.
Statistical feature selection and dimensionality reduction remain Step 10.

## Preprocessing and integration

`src/feature_engineering.py` supplies `BookingFeatureEngineer` and
`make_feature_preprocessor(scale_numeric=True/False)`. It explicitly replaces
the strict Step 7 domain stage while reusing its fixed cleaning rules. The
original Step 7 factory remains unchanged and available for later comparisons.

Derived values are computed before imputation. Training medians are learned
separately for each numeric field, with a zero fallback for an entirely missing
training column. Therefore an imputed guest total need not equal the sum of
separately imputed guest components. This is deliberate and disclosed; fixed
missing indicators preserve missingness for children, ADR, total nights, total
guests, and cancellation share.

Total nights, guests, and previous bookings receive log1p after imputation,
alongside Step 7's four logarithmic fields. Ratios, binary indicators, and cyclic
coordinates are not logged. The scaled variant learns StandardScaler statistics
inside each training fold; the tree-compatible variant omits scaling. Other
categorical vocabularies are training-only and tolerate unseen categories.

For later modeling, put a **new** feature preprocessor in the estimator pipeline
passed to CV. Do not prefit a matrix on all development rows. CV positions from
`development_cv(assignments)` apply to development rows in Step 5 source order.

## Measured evidence

The fixed development cohort contains **95,415 bookings**. The feature audit
retains **604 zero-night bookings**, preserves **four unknown guest totals**
before imputation, and identifies **86,585 bookings with no recorded history**
and **5,900 with a company code recorded**. These describe input features only.

| Fold | Training rows | Validation rows | Encoded columns in both | Nonfinite values | Validation rows with month absent from training |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 23,797 | 23,893 | 332 | 0 | 23,475 |
| 2 | 47,690 | 23,776 | 421 | 0 | 0 |
| 3 | 71,466 | 23,949 | 490 | 0 | 0 |

Each training fold has medians of three total nights, two total guests, and
zero prior bookings. Learned statistics and vocabularies were checked against
training data; validation transformation did not change fitted state. Both
scaled and unscaled variants passed. Feature names are unique and saved per fold.

The first training window lacks February–June. Fixed cyclic coordinates remain
defined for all those months, unlike a missing month-specific one-hot category.
This solves representation coverage only: it does not supply observed seasonal
outcomes or establish generalization to unseen seasons.

All **29 tests** pass, including seven Step 9 tests covering formulas, missing
components, zero denominators, company-code semantics, month wrap-around,
validation isolation, cloning, excluded fields, and leakage/schema rejection.
Notebook 03's **five code cells** executed sequentially in a fresh Python
process with real outputs saved. Jupyter/IPython/nbformat were unavailable;
full fresh-kernel execution and canonical notebook validation remain final gates.

## Reproduce and continue

From the repository root with the documented dependencies:

```bash
python -m src.feature_audit
python -m unittest discover -s tests -v
```

These commands use the committed processed data and frozen assignments.
Evidence is in `data/processed/step9/`: summary, complete schemas, and development
feature descriptive statistics. Only aggregate evidence is exported; no fitted
full-development matrix or model is published.
