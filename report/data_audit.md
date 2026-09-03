# Initial Hotel Booking Data Audit

**CSE437: Data Science | Group 15**

**Status:** raw-data audit completed. Cleaning, development/test assignment, predictor-outcome EDA, and modeling remain pending.

## File verified

| Check | Measured value |
| --- | --- |
| File | hotel_bookings.csv |
| Rows | 119,390 |
| Columns | 32 |
| File size | 16,855,599 bytes (16.86 MB; 16.07 MiB) |
| Arrival period | 2015-07-01 to 2017-08-31 |
| Invalid arrival dates | 0 |
| City / resort records | 79,330 / 40,060 |
| Not canceled / canceled | 75,166 / 44,224 |
| Cancellation share | 37.0416% |

SHA-256: `7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06`

These are measurements of the original file. Class proportions are not held-out model performance.

## Parsed missing values

| Column | Count | Share of records |
| --- | --- | --- |
| `company` | 112,593 | 94.31% |
| `agent` | 16,340 | 13.69% |
| `country` | 488 | 0.41% |
| `children` | 4 | 0.0034% |

Company and agent require different decisions: agent is mostly populated. Their codes are categorical identifiers. The original dataset documentation describes NULL agency/company entries as not applicable; source semantics should guide treatment rather than generic numerical imputation.

## Repeated records

- Additional exact full-row copies after the first occurrence: **31,994 (26.80%)**.
- All rows participating in repeated groups: **40,165**.
- Distinct full rows: **87,396**.

There is no unique booking identifier in this file. Identical rows may reflect repeated extracts or indistinguishable separate bookings. No duplicates have been removed. Document the retain/remove policy and keep identical retained groups out of opposite validation partitions.

## Guest and stay checks

- Known zero-total-guest records: **180**.
- Unknown guest totals due to missing counts: **4**.
- Zero-adult records: **403**, including **223 with a known positive total**.
- Negative guest-count records: **0**.
- Zero-night records: **715**.
- Records flagged as both zero guests and zero nights: **70**.

Total guests were calculated only when all three guest counts were present. Do not replace unknown counts with zero merely to run this check. The flags overlap and should not be summed as independent exclusions.

## ADR checks

| Measure | Value |
| --- | --- |
| Minimum / median / maximum | -6.38 / 94.575 / 5,400 |
| Next largest value after 5,400 | 510 |
| Negative ADR records | 1 |
| Zero ADR records | 1,959 |
| ADR greater than 1,000 | 1 |
| Q1 / Q3 | 69.29 / 126.00 |
| Full-file diagnostic 1.5-IQR fences | -15.775 to 211.065 |
| Records outside those fences | 3,793 |

The negative ADR is **not** outside the IQR fence, so statistical flags alone miss this domain issue. The 5,400 observation deserves inspection, but an unusual value is not automatically an error. Zero rates can have valid operational explanations.

The full-file fences above are descriptive audit results, not thresholds to reuse in modeling. Any clipping, imputation, scaling, selection, or reduction fitted on data must be learned inside training folds.

## Direct outcome leakage

| reservation_status | is_canceled = 0 | is_canceled = 1 |
| --- | --- | --- |
| Check-Out | 75,166 | 0 |
| Canceled | 0 | 43,017 |
| No-Show | 0 | 1,207 |

Reservation status reproduces all observed labels. Exclude `reservation_status` and `reservation_status_date` from predictor inputs. Removing the target and these two columns leaves **29 candidate predictors**, still subject to availability review.

The original publication describes observation timing relative to the day before arrival. This does not establish that every stored value was known when the initial booking was made.

## Other categorical flags

Explicit `Undefined` values occur in market_segment (2), distribution_channel (5), and meal (1,169). Read the field definitions before merging or treating these as unknown categories.

## Recommended next decisions

| Issue | Starting approach to evaluate |
| --- | --- |
| Agent | Retain categorical information and an explicit no-agent category where the source meaning supports it. |
| Company | Compare categorical retention against a company-booking indicator; do not treat code magnitude as meaningful. |
| Country / children | Document unknown-value handling and fit any imputation on training folds. |
| Zero total guests | Investigate and justify eligibility for a guest-booking prediction population. |
| Duplicate groups | Establish a documented policy and prevent retained identical groups crossing partitions. |
| ADR | Investigate suspicious values; compare justified training-only treatments rather than deleting all statistical outliers. |
| Zero nights | Review meaning separately from zero guests. |
| Outcome columns | Exclude both status fields and audit timing of remaining predictors. |

No cleaning decision has yet been applied to the source.

## Reproduction and files

1. Obtain the source CSV and place it at `data/raw/hotel_bookings.csv`.
2. Use the repository setup instructions and run `notebooks/01_data_audit_and_eda.ipynb` from top to bottom.
3. The notebook writes `data/audit_summary.json` and `figures/01_data_quality_audit.png`.

The 10 code cells were executed sequentially in a fresh Python process with an IPython shell, and actual outputs were saved. A separate Jupyter kernel could not launch in this environment; a normal fresh-kernel Jupyter run remains to be verified before submission. The source checksum was unchanged after execution.

The NYC Airbnb CSV and map belong to a different dataset. They are not inputs to this audit.

## Sources

- Source named in the project: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
- Original publication: Antonio, N., de Almeida, A., and Nunes, L. (2019). Hotel booking demand datasets. Data in Brief, 22, 41-49. https://doi.org/10.1016/j.dib.2018.11.126
- Data definitions: https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/
- Exact licence/terms for the downloaded Kaggle version: still to be recorded.
