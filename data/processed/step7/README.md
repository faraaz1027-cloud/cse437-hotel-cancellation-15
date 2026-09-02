# Step 7 — Preprocessing evidence

**Completed: Step 7 | Responsible: Faraaz | Next: Step 8, development-data EDA.**

| File | Purpose |
| --- | --- |
| `preprocessing_summary.json` | Per-fold shapes, missingness before/after, domain decisions, medians, unseen-category counts, runtime versions, and source hashes |
| `feature_schemas.json` | Ordered output names for each training-fold encoder |

The diagnostic matrices were produced separately for each development
training/validation pair. No additional rows were removed and no test rows
were fitted or transformed. No predictive estimator or global preprocessor was
trained. Use the reusable factory inside later model pipelines; do not prefit
one transformer on all development rows before CV.

## Measured output dimensions

| Fold | Training rows | Validation rows | Encoded columns | Nonfinite values after preprocessing |
| --- | ---: | ---: | ---: | ---: |
| 1 | 23,797 | 23,893 | 328 | 0 |
| 2 | 47,690 | 23,776 | 422 | 0 |
| 3 | 71,466 | 23,949 | 491 | 0 |

The 29 Step 5 source columns become 25 initial source fields after excluding
`company`, `assigned_room_type`, `booking_changes`, and `days_in_waiting_list`.
These 25 fields comprise 15 numeric fields and 10 categorical fields. Two
fixed missing indicators cover children and ADR. One-hot vocabularies vary
across folds because only training categories are learned. Matrix columns must
not be pooled by position between folds; use the matching schema.

## Reproduce

From the repository root using the committed Step 5/6 files:

```bash
python -m src.preprocessing_audit
python -m unittest discover -s tests -v
```

The complete Notebook 02 also runs Steps 5–6 and therefore needs the original
CSV for its earlier cells. A separate fresh-kernel Jupyter run remains a final
submission verification task. The executed environment is recorded in the
summary; `requirements.txt` retains compatible starter ranges, not a certified
clean-install lockfile.

## Reuse in later modeling

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from src.preprocessing import make_preprocessor
from src.splitting import development_cv

pipeline = Pipeline([
    ('preprocess', make_preprocessor()),
    ('model', LogisticRegression(max_iter=2000, random_state=42)),
])
search = GridSearchCV(pipeline, {'model__C': [0.1, 1.0, 10.0]},
                      scoring='f1', cv=development_cv(assignments))
# Run in Step 11/12 after adding the Step 9/10 components:
# search.fit(X_dev, y_dev)
```

Construct `assignments`, `X_dev`, and `y_dev` using
[the Step 6 loading example](../../splits/README.md). Pass raw candidate
DataFrames in that exact development order. Set `scale_numeric=False` for the
tree-compatible variant. Feature-engineering extensions require an explicit
schema update in Step 9; unknown extra columns raise an error instead of being
silently dropped. The final fitted model/pipeline will be saved in the later
modeling stage.

Step 8 uses development rows for descriptive statistics and relationships.
Company-presence/derived-calendar features remain Step 9 candidates. The
policy exclusions in this stage are not a substitute for statistical feature
selection and dimensionality reduction in Step 10.

Details: [Step 7 report](../../../report/step7_preprocessing.md).
