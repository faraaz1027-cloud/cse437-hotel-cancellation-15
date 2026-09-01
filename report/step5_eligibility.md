# Step 5 — Eligibility and direct leakage removal

**CSE437, Group 15 | Responsible member: Faraaz | Status: complete**

Next: **Step 6 — Faraaz freezes the development/test split and validation plan.**
This report documents executed preparation, not model performance.

## Implemented decisions

1. Exclude `reservation_status` and `reservation_status_date` from candidate
   predictors. Neither field, nor a derivative, belongs in a predictive model.
2. Store `is_canceled` separately, preserving its source values and row order.
3. Exclude a record only when adults, children, and babies are all known and
   their sum is zero. This defines the analytic population; it does not prove
   such administrative records are erroneous.
4. Keep unknown guest totals and zero-adult bookings with positive total guests.
5. Keep repeated records. Without a booking ID, exact matches do not establish
   accidental duplication. Group identical candidate predictors independently
   of the target and statuses for use in partition checks.
6. Keep and flag zero-night, zero-price, negative-price, and exceptionally high
   positive-price records. The ADR > 1,000 flag is for review only and never
   changes eligibility. No full-data IQR fence is used to delete or cap values.

The original CSV and proposal documents remain unchanged. No retained source
values were edited. No imputation, encoding, scaling, model fit, or evaluation
split occurs in this stage.

## Verified results

| Measure | Result |
| --- | ---: |
| Original rows | 119,390 |
| Excluded known-zero-guest rows | 180 |
| Eligible rows | 119,210 |
| Candidate predictors | 29 |
| Not canceled / canceled in eligible cohort | 75,011 / 44,199 |
| Retained unknown guest totals | 4 |
| Retained zero-adult bookings with positive guest totals | 223 |
| Retained zero-night bookings | 645 |
| Retained negative-ADR rows | 1 |
| Retained zero-ADR rows | 1,810 |
| Retained ADR-above-1,000 rows | 1 |
| Original full-row additional copies among retained bookings | 31,980 |
| Distinct candidate-predictor groups | 86,707 |
| Additional copies of candidate-predictor combinations | 32,503 |
| Candidate rows participating in repeated groups | 40,730 |
| Candidate groups containing both outcome labels | 265 |

Full-row duplicates and candidate-predictor duplicates are different counts:
dropping status information and target can make additional rows identical.
Conflicting outcomes are kept; selecting a label based on majority or dropping
one outcome would distort the supervised task. Every candidate group has one
arrival date, permitting whole-date splits without dividing these groups.

## Outputs and reproducibility

- [Executed notebook](../notebooks/02_preprocessing.ipynb)
- [Reusable implementation](../src/eligibility.py)
- [Processed files and loading instructions](../data/processed/README.md)
- [Measured summary and hashes](../data/processed/step5_summary.json)
- [Boundary and leakage-invariance tests](../tests/test_eligibility.py)

The four row-level output files have been generated and verified locally.
Their public upload awaits explicit permission after automatic approval review
blocked disclosure of the booking records. This repository contains the code,
executed aggregate results, decisions, and file hashes needed to reproduce them.

Run from the repository root after supplying the original CSV:

```bash
python -m src.eligibility
python -m unittest discover -s tests -v
```

Notebook 02's five code cells executed sequentially in a fresh Python process
using IPython, with real outputs saved. Original-value comparison, complete
row accounting, input/output hashes, compressed-file reloads, and metadata
alignment passed. Four focused tests check unknown guests and anomalous rows,
independence from outcome information, original-data preservation, and handling
of an unexpected negative guest count. A separate fresh Jupyter-kernel run
remains a final submission verification task.

## Decisions carried forward

**Step 6:** define chronological holdout and forward development validation
before predictor-outcome EDA. Freeze row assignments and verify no duplicate
group crosses partitions in any split. No dates or ratios are fixed here.

**Step 7:** negative ADR will be treated as unavailable under a nonnegative-price
assumption, then imputed using training-fold statistics. Preserve zero and
high positive ADR; assess any learned transform or cap using development folds.
Treat `agent` as a category, with an explicit no-agent value under the source
NULL convention. Exclude the sparse `company` identifier from the primary
model and consider a company-presence indicator in Step 9. Distinguish these
not-applicable categories from unknown country or child information. These
rules are documented but have not been applied in the Step 5 candidate CSV.

**Prediction-time availability:** the 29 candidate columns are not the final
feature set. Review potentially updated fields such as `assigned_room_type`,
`booking_changes`, and `days_in_waiting_list` against the intended prediction
time. Removing direct outcome leakage does not establish booking-time validity.

**Substantive focus:** investigate longer lead time, deposit type, and prior
cancellations as named hypotheses in development data. Any findings are
associations, not causal effects. The approved research questions are unchanged.

## Limitations and assistance

The zero-guest rule narrows the population. Retaining repeated records preserves
their frequency weight and may emphasize common booking patterns; report that
choice and consider a development-only sensitivity check later. Unusual ADR,
missing values, and uncertain feature timing remain explicit later tasks.
Full-source quality summaries were inspected before splitting; report this
transparently rather than claiming that no information about the later holdout
was ever seen.

OpenAI ChatGPT/Codex assisted with the implementation, policy documentation,
execution, and verification. Faraaz owns reviewing and explaining Step 5;
the final contribution statement must describe actual member work.

Sources: [Kaggle dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand);
Antonio, N., de Almeida, A., and Nunes, L. (2019), *Hotel booking demand datasets*,
Data in Brief 22, 41–49, https://doi.org/10.1016/j.dib.2018.11.126;
[original documentation](https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/).
