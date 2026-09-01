# Step 5 outputs

Step 5 implementation (Faraaz) is complete. Next: Step 6, freeze the evaluation partitions.

**Publication status:** the four row-level data files listed below were generated
and verified locally. They are not committed to this public repository pending
explicit permission to publish the booking records. The aggregate summary,
hashes, code, and notebook outputs are available here. Run the command below
to regenerate the four data files from the original CSV.

These files contain the eligible cohort **before** imputation, encoding,
scaling, feature selection, or train/test splitting. They are not a final
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

After generating the files, load the compressed outputs with:

```python
from pathlib import Path
import pandas as pd

data_dir = Path('data/processed')  # from the repository root
X = pd.read_csv(data_dir / 'step5_candidates.csv.gz')
y = pd.read_csv(data_dir / 'step5_target.csv.gz')['is_canceled']
metadata = pd.read_csv(data_dir / 'step5_metadata.csv.gz')
assert len(X) == len(y) == len(metadata)
```

The negative ADR remains negative here and is flagged for the documented
Step 7 treatment. Missing fields and sparse company codes also remain.
Drop or transform these only through the later documented preprocessing.
Candidate availability at the intended prediction time still needs review.

Duplicate groups are defined by the 29 candidate columns, excluding target
and outcome-status fields, with missing values grouped consistently. A group's
members all have the same arrival date. Step 6 must enforce group separation
at every split and avoid splitting a calendar date. Do not use group IDs,
row IDs, dates from metadata, or review flags as predictors by accident.

Decisions and limitations: [Step 5 report](../../report/step5_eligibility.md).
