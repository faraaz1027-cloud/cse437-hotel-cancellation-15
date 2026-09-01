# Dataset information

**Status:** the supplied hotel CSV has been audited. The raw CSV is not yet committed in this repository; download/place it as described below to rerun the notebook.

## Source and acquisition

Source: [Hotel Booking Demand by Jesse Mostipak on Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand).

Download the dataset, extract `hotel_bookings.csv`, and place the untouched file at `data/raw/hotel_bookings.csv`. Keep original source files unchanged. Record the source version, download date, and exact licence terms.

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
| Source version / actual download date | Not supplied; record from the original download |
| Exact Kaggle dataset licence | Pending confirmation at the source |

The file is below 50 MB. Adding the raw data remains a submission task. The source download link above supports obtaining the input in the meantime.

## Audit evidence

- [Human-readable audit](../report/data_audit.md)
- [Executed audit notebook](../notebooks/01_data_audit_and_eda.ipynb)
- [Machine-readable audit record](audit_summary.json)
- [Quality figure](../figures/01_data_quality_audit.png)

The audit has not removed rows, imputed values, or produced a cleaned dataset. Save subsequent cleaning outputs in `data/processed/` and document how they were generated.

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
- The separate NYC files are outside this hotel's cancellation-prediction dataset.

