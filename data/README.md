# Dataset information

## Source and acquisition

Source: [Hotel Booking Demand by Jesse Mostipak on Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand).

Download the dataset, extract `hotel_bookings.csv`, and place the untouched file at `data/raw/hotel_bookings.csv`. Keep original source files unchanged. For a new download, record its date/version separately: these do not establish when the original supplied file was acquired. The original acquisition date and acquired source version are unknown, as confirmed by the user. 

## Verified file record

| Field | Value |
| --- | --- |
| Original filename | hotel_bookings.csv |
| Rows / columns | 119,390 / 32 |
| Actual size | 16,855,599 bytes (16.86 MB; 16.07 MiB) |
| SHA-256 | `7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06` |
| Observed arrival period | 2015-07-01 to 2017-08-31 |
| City / resort bookings | 79,330 / 40,060 |
| Target | is_canceled |
| Not canceled / canceled | 75,166 / 44,224 |
| Original acquired source version | Unknown — user confirmed |
| Original download date | Unknown — user confirmed |
| Kaggle dataset licence | CC BY 4.0; public metadata verified 2026-09-02 |
| Current public metadata version | 1; distinct from the unknown original acquired version |

The file is below 50 MB and is included in a normal repository clone. 

## Attribution and licence

The [Kaggle public metadata](https://www.kaggle.com/api/v1/datasets/view/jessemostipak/hotel-booking-demand)
lists Attribution 4.0 International (CC BY 4.0). See the
[licence](https://creativecommons.org/licenses/by/4.0/) and the compact
[verified provenance record](processed/source_provenance.json). Credit Nuno Antonio,
Ana de Almeida and Luis Nunes for the original publication, Jesse Mostipak for
the Kaggle distribution, and the acknowledged prior preparation by Thomas Mock
and Antoine Bichat for TidyTuesday. No endorsement is implied.

Raw bytes remain unchanged. Project-derived data use the separately documented
eligibility, leakage exclusions, grouping and training-fitted transformations.
Retain these attribution and change notices when sharing. The current public
version/date must not be substituted for unknown original acquisition facts.

## Audit evidence

- [Human-readable audit](README.md)
- [Executed audit notebook](../notebooks/01_data_audit_and_eda.ipynb)
- [Machine-readable audit record](processed/audit_summary.json)
- [Quality figure](../figures/01_data_quality_audit.png)

The original audit is unchanged. eligibility now produces a separate cohort of **119,210 rows**, removes the two reservation-status fields from predictors, and exports 29 candidate predictors separately from the target and metadata. No retained values are imputed or otherwise changed. See [processed outputs](README.md), [eligibility summary](processed/eligibility_summary.json), and [the decision report](README.md).

## Source documentation

Antonio, N., de Almeida, A., and Nunes, L. (2019). *Hotel booking demand datasets*. Data in Brief, 22, 41-49. https://doi.org/10.1016/j.dib.2018.11.126

The publication describes extraction from hotel Property Management System SQL databases. It explains agency/company NULL semantics and observation timing. See https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/ . Document any differences between the publication and the combined Kaggle CSV.

## Handling rules

- Preserve original source bytes.
- Use repository-relative paths.
- Document unknown versus not-applicable category handling.
- Exclude both reservation-status fields from model inputs.
- Prevent overlap between retained duplicate groups in evaluation partitions.
- Fit learned preprocessing only inside training folds.
- Only the hotel dataset is used. Unrelated candidate-dataset files have been removed from the current tree and remain recoverable from Git history.

## How to Obtain the Data

1. Open the Kaggle dataset link above.
2. Download the dataset.
3. Extract `hotel_bookings.csv`.
4. Place the unchanged file at:

   `data/raw/hotel_bookings.csv`

The raw dataset must remain unchanged. Files generated after cleaning and preprocessing are stored in `data/processed/`.
