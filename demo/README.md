# Hotel Booking Cancellation — Interactive Demo

This is an **educational Streamlit portfolio demo** built around the repository's
existing, frozen Logistic Regression pipeline. It does not retrain the model or
change any recorded evaluation results.

## Run locally

Use **Python 3.12**, then from the repository root:

```bash
python -m pip install -r demo/requirements.txt
python -m streamlit run demo/app.py
```

The saved model is in `models/final_logistic_regression.joblib`.
The original metrics are read from
`data/processed/results/evaluation/final_metrics.csv`.

## Free deployment (Streamlit Community Cloud)

1. Review and merge this demo into the repository's `main` branch.
2. Sign in to <https://share.streamlit.io/> with your GitHub account.
3. Select **Create app**, then **Yup, I have an app**.
4. Choose repository `faraaz1027-cloud/cse437-hotel-cancellation-15`,
   branch `main`, entrypoint `demo/app.py`.
5. In **Advanced settings**, select **Python 3.12**.
6. Select **Deploy**. Dependencies in `demo/requirements.txt` will be used.
7. Try multiple hypothetical bookings and check that model inference works.
   If there is an error, view the app's deployment logs.

Only add your real `streamlit.app` URL to your README and CV after the
form loads and can produce predictions.

## Important context

- This is a **team project**. Consult the main README for individual contributions.
- Inputs are hypothetical scenarios, not real-time hotel booking data.
- The dataset is historical (2015–2017), and predictions are not suitable for
  actual booking or financial decisions.
- The demonstration contains fixed input assumptions (inspect them in the app).
- Never change the frozen evaluation results when developing UI features.
