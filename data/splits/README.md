# Step 6 — Frozen evaluation partitions

**Responsible: Faraaz. Completed: Step 6. Next: Step 7, fitted preprocessing.**

**Publication:** the row-level assignment file has been generated and verified
locally. Its public upload awaits explicit permission after automatic approval
review. The code, aggregate plan, and hashes are published. Generate the exact
assignments from the already committed Step 5 data with `python -m src.splitting`
before running the loading example below.

| File | Purpose |
| --- | --- |
| `step6_assignments.csv.gz` | Generated output; one assignment per Step 5 row, preserving row order; public upload pending |
| `step6_split_plan.json` | Fixed dates, counts, metrics, validation protocol, and input/output hashes |

## Final split

| Partition | Arrival dates (inclusive) | Bookings |
| --- | --- | ---: |
| Development | 2015-07-01 to 2017-04-22 | 95,415 |
| Test | 2017-04-23 to 2017-08-31 | 23,795 |

The whole-date boundary minimizes the difference from an 80% development row
count; ties choose the earlier date. No label, model result, random shuffle,
or stratification determines the boundary.

## Forward validation inside development data

| Fold | Training through | Training rows | Validation period (inclusive) | Validation rows |
| --- | --- | ---: | --- | ---: |
| 1 | 2016-01-26 | 23,797 | 2016-01-27 to 2016-06-21 | 23,893 |
| 2 | 2016-06-21 | 47,690 | 2016-06-22 to 2016-11-06 | 23,776 |
| 3 | 2016-11-06 | 71,466 | 2016-11-07 to 2017-04-22 | 23,949 |

Every training window begins on 2015-07-01. Development boundaries approximate
25%, 50%, and 75% row prefixes, preserving whole dates. Window durations differ.
Future development rows are `unused` until a later fold. Test rows always have
the role `excluded_test`. Every duplicate group has one role in each fold.

## Loading and index alignment

Run this from the repository root:

```python
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from src.splitting import check_assignments, development_cv

root = Path('.')
plan = json.loads((root / 'data/splits/step6_split_plan.json').read_text())
path = root / 'data/splits' / plan['assignment_file']
assert hashlib.sha256(path.read_bytes()).hexdigest() == plan['assignment_sha256']
assignments = pd.read_csv(path)
check_assignments(assignments)
X = pd.read_csv(root / 'data/processed/step5_candidates.csv.gz')
y = pd.read_csv(root / 'data/processed/step5_target.csv.gz')['is_canceled']
metadata = pd.read_csv(root / 'data/processed/step5_metadata.csv.gz')
assert assignments.source_row_id.tolist() == metadata.source_row_id.tolist()
assert len(X) == len(y) == len(assignments)

dev_rows = np.flatnonzero(assignments.partition.eq('development'))
X_dev = X.iloc[dev_rows].reset_index(drop=True)
y_dev = y.iloc[dev_rows].reset_index(drop=True)
cv = development_cv(assignments)
# Later: GridSearchCV(pipeline, ..., scoring='f1', cv=cv).fit(X_dev, y_dev)
# Fit all learned preprocessing/selection inside pipeline and each train fold.
```

The helper returns indices **relative to the development subset**, not the full
cohort. Preserve the demonstrated row order. `cohort_row` is the zero-based
position in Step 5 files; `source_row_id` identifies the original raw row.
Neither IDs, dates from metadata, fold roles, nor partition labels are model
features. The 29 candidate inputs still need Step 7 availability/preprocessing
review.

To reproduce/verify the split from the committed Step 5 files, without needing
the original raw CSV:

```bash
python -m src.splitting
python -m unittest discover -s tests -v
```

Rerunning unchanged inputs preserves the frozen artifacts. Changed membership
or plan settings cause an error; any revision must be explicitly reviewed and
versioned before evaluation. Do not revise the split to improve model scores.

## Evaluation protocol

- Select by mean cancellation-class F1 across the three development folds;
  also report per-fold results, accuracy, precision, recall, and ROC-AUC.
- Compare a majority baseline, logistic regression, and random forest using
  these same folds. Use random state 42 for stochastic models/searches later.
- Default classification threshold is 0.5; any threshold tuning uses development
  validation only. Do not tune using test results.
- Fit preprocessing, feature selection, and dimensionality reduction only on
  each training fold. Refit the selected complete pipeline on all development
  rows, then evaluate the held-out test set once in Step 13.
- No model was trained and no test class distribution or score was computed in
  the split construction. Existing full-source quality audits remain disclosed.

This is retrospective generalization to later arrival cohorts. It does not
establish booking-time feature snapshots or live availability of training
labels; no temporal embargo is applied. Consider seasonality and distribution
change when interpreting the later test period.

See [Step 6 report](../../report/step6_evaluation_plan.md) and
[timeline](../../figures/02_evaluation_timeline.png).
