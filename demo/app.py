"""Portfolio demo for the recorded CSE437 hotel cancellation classifier.

Run from the repository root:
    streamlit run demo/app.py

This application LOADS the frozen model; it never refits, retunes, or
changes the published evaluation.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Required so the fitted custom pipeline can be imported when joblib loads it.
from src.modeling import cancellation_probability  # noqa: E402
from src.preprocessing import MODEL_COLUMNS  # noqa: E402


MODEL_FILE = ROOT / "models" / "final_logistic_regression.joblib"
METRICS_FILE = (
    ROOT / "data" / "processed" / "results" / "evaluation" / "final_metrics.csv"
)


@st.cache_resource(show_spinner="Loading the recorded model...")
def load_model():
    """Load the existing, frozen fitted pipeline; do not retrain it."""
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_metrics():
    return pd.read_csv(METRICS_FILE).iloc[0]


def booking_input(
    arrival: date,
    hotel: str,
    lead_time: int,
    adr: float,
    weekend: int,
    weekdays: int,
    adults: int,
    children: int,
    cancellations: int,
    past_non_cancellations: int,
    requests: int,
    deposit: str,
    market_segment: str,
    customer_type: str,
    country: str,
    meal: str,
    repeat_guest: bool,
    parking: int,
) -> pd.DataFrame:
    """Create a complete, transparent synthetic scenario in the training schema."""
    values = {
        "lead_time": lead_time,
        "previous_cancellations": cancellations,
        "previous_bookings_not_canceled": past_non_cancellations,
        "adr": adr,
        "arrival_date_year": arrival.year,
        "arrival_date_week_number": arrival.isocalendar().week,
        "arrival_date_day_of_month": arrival.day,
        "stays_in_weekend_nights": weekend,
        "stays_in_week_nights": weekdays,
        "adults": adults,
        "children": children,
        "babies": 0,
        "is_repeated_guest": int(repeat_guest),
        "required_car_parking_spaces": parking,
        "total_of_special_requests": requests,
        "hotel": hotel,
        "arrival_date_month": arrival.strftime("%B"),
        "meal": meal,
        "country": country,
        "market_segment": market_segment,
        "distribution_channel": "TA/TO",
        "reserved_room_type": "A",
        "deposit_type": deposit,
        "agent": float("nan"),
        "customer_type": customer_type,
        "company": float("nan"),
    }
    # Feature engineering uses 'company' as an allowed additional raw field.
    required = set(MODEL_COLUMNS) | {"company"}
    if set(values) != required:
        raise ValueError("The demonstration input schema does not match the model.")
    return pd.DataFrame([values])


st.set_page_config(
    page_title="Hotel Booking Cancellation | ML Portfolio",
    page_icon="🏨",
    layout="wide",
)

st.title("🏨 Hotel Booking Cancellation Prediction")
st.write(
    "Explore a university machine-learning project using an existing "
    "Logistic Regression pipeline trained on historical hotel booking data."
)
st.caption(
    "CSE437 team project · Faraaz Jamil Chowdhury and Ihfaz Rashid Sadat · "
    "Historical booking records (2015–2017)"
)

st.info(
    "Educational demonstration only. Predictions apply to hypothetical "
    "historical scenarios, not actual hotel guests or modern bookings. "
    "The model was not independently validated for production decisions."
)

try:
    metrics = load_metrics()
except Exception:
    st.error("Recorded evaluation metrics are unavailable in this checkout.")
    st.stop()

st.subheader("Recorded held-out test results")
col1, col2, col3, col4 = st.columns(4)
col1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
col2.metric("Accuracy", f"{100 * metrics['accuracy']:.1f}%")
col3.metric("F1", f"{metrics['f1']:.3f}")
col4.metric("Test records", f"{int(metrics['test_rows']):,}")
st.caption(
    "These are the original published results on the chronological holdout. "
    "This app does not retrain or re-evaluate the model."
)

st.divider()
st.subheader("Try a hypothetical booking")

with st.form("booking_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        hotel = st.selectbox("Hotel type", ["City Hotel", "Resort Hotel"])
        arrival = st.date_input(
            "Arrival date (historical range)",
            value=date(2016, 7, 15),
            min_value=date(2015, 7, 1),
            max_value=date(2017, 8, 31),
        )
        lead_time = st.number_input("Lead time (days)", 0, 737, 45)
        adr = st.number_input("Average daily rate", 0.0, 1000.0, 95.0, 5.0)
    with c2:
        weekend = st.number_input("Weekend nights", 0, 30, 1)
        weekdays = st.number_input("Weekday nights", 0, 30, 2)
        adults = st.number_input("Adults", 1, 10, 2)
        children = st.number_input("Children", 0, 10, 0)
        requests = st.number_input("Special requests", 0, 5, 1)
    with c3:
        deposit = st.selectbox(
            "Deposit type", ["No Deposit", "Non Refund", "Refundable"]
        )
        market_segment = st.selectbox(
            "Market segment",
            ["Online TA", "Offline TA/TO", "Direct", "Groups", "Corporate"],
        )
        customer_type = st.selectbox(
            "Customer type", ["Transient", "Transient-Party", "Contract", "Group"]
        )
        cancellations = st.number_input("Previous cancellations", 0, 30, 0)
        past_non_cancellations = st.number_input(
            "Previous non-cancelled bookings", 0, 30, 0
        )

    with st.expander("Additional booking assumptions"):
        a1, a2 = st.columns(2)
        with a1:
            country = st.selectbox(
                "Country code",
                ["PRT", "GBR", "FRA", "ESP", "DEU", "USA", "BRA"],
            )
            meal = st.selectbox("Meal plan", ["BB", "HB", "FB", "SC", "Undefined"])
        with a2:
            repeat_guest = st.checkbox("Repeated guest", value=False)
            parking = st.number_input("Parking spaces requested", 0, 5, 0)
        st.caption(
            "Remaining assumed inputs: zero babies, room type A, "
            "TA/TO distribution channel, and missing agent/company codes. "
            "All input values are shown after submitting."
        )

    submitted = st.form_submit_button(
        "Estimate cancellation probability", type="primary"
    )

if submitted:
    try:
        record = booking_input(
            arrival, hotel, lead_time, adr, weekend, weekdays,
            adults, children, cancellations, past_non_cancellations,
            requests, deposit, market_segment, customer_type,
            country, meal, repeat_guest, parking,
        )
        model = load_model()
        probability = float(cancellation_probability(model, record)[0])
    except Exception as exc:
        st.error("Prediction could not be produced from the saved pipeline.")
        st.exception(exc)
    else:
        st.subheader("Model output")
        st.metric("Estimated cancellation score", f"{100 * probability:.1f}%")
        st.progress(probability)
        st.write(
            "Classification using the project's frozen 0.50 threshold: "
            + ("**Cancellation**" if probability >= 0.5 else "**Not cancellation**")
        )
        st.caption(
            "This probability is a model score on synthetic input, not a "
            "guarantee or an estimate validated for individual guests."
        )
        with st.expander("See all values sent to the model"):
            st.dataframe(record.T.rename(columns={0: "Value"}))

st.divider()
st.markdown(
    "[Source code and methodology]"
    "(https://github.com/faraaz1027-cloud/cse437-hotel-cancellation-15)"
    " · [Original report]"
    "(https://github.com/faraaz1027-cloud/cse437-hotel-cancellation-15/blob/main/report/report.pdf)"
)
st.caption(
    "Team credit and individual contributions are documented in the repository "
    "README. See the report for evaluation limits and data provenance."
)
