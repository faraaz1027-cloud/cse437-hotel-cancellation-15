# Step 5 outputs

Steps 5–7 are complete. The original Step 5 files below are preserved. Frozen splits are in [../splits/](../splits/README.md); Step 7 preprocessing evidence and reuse instructions are in [step7/](step7/README.md).

**Publication status:** all four verified row-level data files are committed
in this repository, with the aggregate summary, hashes, code, and notebook
outputs. Use these files with the fixed Step 6 assignments, or use the command
below to regenerate them from the original CSV.

These files contain the eligible cohort **before** imputation, encoding,
scaling, or feature selection. Step 6 partition membership is stored separately
in `data/splits/`; these Step 5 files retain their original row order. They are not a final
model-ready dataset. Gzip is lossless compression; pandas reads it directly.

| File | Contents |
| --- | --- |
| `step5_candidates.csv.gz` | 119,210 rows, 29 original candidate predictors; no target or reservation statuses |
| `step5_target.csv.gz` | Corresponding `is_canceled` target in exactly the same row order |
| `step5_metadata.csv.gz` | Row lineage, arrival dates, duplicate groups, and quality flags; never model inputs |
| `step5_exclusions.csv` | Original row positions and reason for the 180 excluded zero-guest records |
| `step5_summary.json` | Measured counts, versions, and hashes for all four data files |

`source_row_id` is a zero-based original record position after the header, not
a real booking ID. Positional alignment across candidates, target, and metadata
is mandatory. Do not sort, drop, or shuffle one file independently.

From the repository root, regenerate with:

```bash
python -m src.eligibility
```

Or run `notebooks/02_preprocessing.ipynb` from top to bottom. Both require the
unchanged original `data/raw/hotel_bookings.csv`; see `../README.md`.

Load the committed compressed outputs with:

```python
from pathlib import Path
import pandas as pd

data_dir = Path('data/processed')  # from the repository root
X = pd.read_csv(data_dir / 'step5_candidates.csv.gz')
y = pd.read_csv(data_dir / 'step5_target.csv.gz')['is_canceled']
metadata = pd.read_csv(data_dir / 'step5_metadata.csv.gz')
assert len(X) == len(y) == len(metadata)
```

The original Step 5 candidate file retains its negative ADR, missing values,
and company codes. Step 7 applies documented domain rules and training-fitted
preprocessing through the pipeline without modifying this source copy. Its
initial schema uses 25 source fields; the three post-booking-timing fields and
company ID are excluded. Candidate availability at booking creation is still
not established by these retrospective source snapshots.

Duplicate groups are defined by the 29 candidate columns, excluding target
and outcome-status fields, with missing values grouped consistently. A group's
members all have the same arrival date. Step 6 verifies group separation at
every split and preserves whole calendar dates. Do not use group IDs,
row IDs, dates from metadata, or review flags as predictors by accident.

Decisions and limitations: [Step 5 report](../../report/step5_eligibility.md).
