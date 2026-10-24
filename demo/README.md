# Hotel Booking Cancellation — Interactive Demo

**[Open the live Streamlit demo](https://cse437-hotel-cancellation-15-tbps3rptce7gx2xnyemwqt.streamlit.app/)**

This educational portfolio app loads the **existing frozen Logistic Regression
pipeline**, displays the **original** reported test metrics, and offers a form
for hypothetical hotel-booking scenarios. It does **not** retrain the model,
change reported evaluation results, or provide production-grade forecasts.

This is a collaborative CSE437 project. Read the [main README](../README.md)
for individual contributions, dataset attribution, and limitations.

## Run locally

Use Python **3.12**. From the repository root:

```bash
python -m pip install -r demo/requirements.txt
python -m streamlit run demo/app.py
```

The saved pipeline is at `models/final_logistic_regression.joblib`.
The recorded metrics come from
`data/processed/results/evaluation/final_metrics.csv`.

## Deploy your own Streamlit copy (free)

1. Sign in at <https://share.streamlit.io/> using GitHub.
2. Create a new app from `faraaz1027-cloud/cse437-hotel-cancellation-15`.
3. Select the `main` branch and `demo/app.py` as the entry point.
4. In Advanced settings, select Python **3.12** (important for pinned dependencies).
5. Deploy, then submit a test booking to confirm the model loads.

The initially deployed portfolio app was configured on the
`portfolio-streamlit-demo` branch. That existing deployment does not
automatically switch branches when this change is merged into `main`.
The demonstration code is kept on both branches.

## Limitations

- Inputs represent synthetic scenarios within the dataset's historical
  timeframe (2015–2017), not live reservations.
- Some less prominent input variables are fixed or use disclosed assumptions.
  Expand **See all values sent to the model** for details.
- Scores are for an educational demonstration and have not been validated
  for individual or production hotel decisions.
- The original evaluation protocol, fitted model and metrics remain unchanged.
