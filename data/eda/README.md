# Step 8 — Development-data EDA

**Complete: Step 8 (Faraaz). Next: Step 9, feature engineering (Sadat).**

All tables use the frozen **95,415 development bookings**, covering 2015-07-01
through 2017-04-22. They contain 34,473 cancellations: **36.13%**. No test
relationships or model results are included. Source/split hashes and artifact
hashes are in `eda_summary.json`.

| Table | Contents |
| --- | --- |
| `numeric_descriptive.csv` | Valid/missing counts, mean, SD, quartiles, p95/p99, and range for the 15 initial numeric fields |
| `missingness.csv` | Original development missing counts for the 29 candidate columns |
| `categorical_descriptive.csv` | Category counts and most-frequent values after fixed domain rules |
| `lead_time_rates.csv` | Cancellation numerator, denominator, and percentage for six fixed lead-time bins |
| `deposit_rates.csv` | Rates for the three deposit types |
| `prior_cancellation_rates.csv` | Rates for 0, 1, 2–3, and 4+ previous cancellations |
| `hotel_rates.csv` | City/resort rates |
| `*_by_hotel.csv` | Lead-time, deposit, and prior-cancellation rates stratified by hotel |
| `monthly_rates.csv` | Monthly development rates/counts; April 2017 is partial through day 22 |
| `sensitivity_*.csv` | Alternative rates giving each duplicate-profile group total weight one |

Fixed Step 7 rules mark negative ADR missing and preserve unknown/no-agent
categories. Numeric EDA excludes missing values field by field rather than
fitting an imputer. Company ID and the three timing-risk fields are excluded
from the initial analyzed schema. No predictive feature is added or selected.

Main rates weight every booking equally. Sensitivity rates instead weight each
Step 5 duplicate-profile group equally, retaining the mean outcome when labels
conflict. This changes the descriptive estimand, not the dataset, frozen split,
or future model weights. Do not infer independent observations or causal effects.
Group sizes matter, particularly refundable bookings and high prior counts.

Reproduce from the repository root:

```bash
python -m src.development_eda
python -m unittest discover -s tests -v
```

The module uses committed processed files; the original raw CSV is not needed
for Step 8 alone. Full Notebook 01 also includes the older raw audit and needs
the original CSV for those cells. The current session lacked IPython/nbformat:
five new EDA cells ran with Python and actual display outputs were captured;
the earlier ten executed audit outputs were preserved. A full fresh Jupyter
kernel run and canonical notebook validation remain pending.

See [Notebook 01](../../notebooks/01_data_audit_and_eda.ipynb),
[report Sections 1–3](../../report/report.md), and
[Sadat's handoff](../../HANDOFF_TO_SADAT.md).
