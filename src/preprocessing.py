"""preprocessing: deterministic domain rules and train-fitted preprocessing factories.

Never fit these components on the full cohort before cross-validation. Put a
new preprocessor inside the estimator Pipeline passed to development CV.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

LEAKAGE_COLUMNS = ("is_canceled", "reservation_status", "reservation_status_date")
METADATA_COLUMNS = ("source_row_id", "cohort_row", "arrival_date", "duplicate_group_id",
    "duplicate_group_size", "partition", "cv_fold_1", "cv_fold_2", "cv_fold_3",
    "unknown_guest_total", "zero_adults_positive_guests", "zero_total_nights",
    "negative_adr", "zero_adr", "adr_above_1000_review_only")
POLICY_EXCLUSIONS = {
    "company": "Sparse company identifier; possible presence feature deferred to feature engineering.",
    "assigned_room_type": "Assignment can change after reservation; exclude from primary inputs.",
    "booking_changes": "Accumulated post-reservation changes; timing is uncertain.",
    "days_in_waiting_list": "Accumulated waiting time; timing is uncertain.",
}
LOG_COLUMNS = ("lead_time", "previous_cancellations", "previous_bookings_not_canceled", "adr")
NUMERIC_COLUMNS = ("arrival_date_year", "arrival_date_week_number", "arrival_date_day_of_month",
    "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies",
    "is_repeated_guest", "required_car_parking_spaces", "total_of_special_requests")
CATEGORICAL_COLUMNS = ("hotel", "arrival_date_month", "meal", "country", "market_segment",
    "distribution_channel", "reserved_room_type", "deposit_type", "agent", "customer_type")
MISSING_INDICATOR_COLUMNS = ("children", "adr")
MODEL_COLUMNS = LOG_COLUMNS + NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


class BookingDomainCleaner(TransformerMixin, BaseEstimator):
    """Preserve rows, select the 25 declared fields, and apply fixed domain rules.

    This transformer learns no medians, thresholds, or category vocabularies.
    Other eligibility fields are excluded by policy. Unexpected extra columns raise
    an error so later feature engineering cannot be silently discarded.
    """

    def _clean(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Pass a DataFrame with named eligibility candidate columns.")
        if not X.columns.is_unique:
            raise ValueError("Duplicate input column names are not permitted.")
        forbidden = set(X.columns).intersection(LEAKAGE_COLUMNS + METADATA_COLUMNS)
        if forbidden:
            raise ValueError(f"Target, leakage, or metadata passed as predictors: {sorted(forbidden)}")
        missing = set(MODEL_COLUMNS).difference(X.columns)
        if missing:
            raise ValueError(f"Required model fields missing: {sorted(missing)}")
        extras = set(X.columns).difference(MODEL_COLUMNS).difference(POLICY_EXCLUSIONS)
        if extras:
            raise ValueError(f"Undeclared features: {sorted(extras)}. Extend the schema explicitly in feature engineering.")
        out = X.loc[:, list(MODEL_COLUMNS)].copy(deep=True)
        for column in LOG_COLUMNS + NUMERIC_COLUMNS:
            values = pd.to_numeric(out[column], errors="raise").astype(float)
            values = values.replace([np.inf, -np.inf], np.nan)
            if column == "adr":
                values = values.mask(values.lt(0))
            elif values.lt(0).any():
                raise ValueError(f"Unexpected negative {column}; review data rather than silently change it.")
            out[column] = values
        # Agent IDs are nominal codes. A null means no agent under the source
        # documentation's convention; a literal code 0 remains a separate code.
        agent = pd.to_numeric(out["agent"], errors="raise")
        present = agent.dropna()
        if (not np.isfinite(present).all() or present.lt(0).any()
                or present.mod(1).ne(0).any()):
            raise ValueError("Agent IDs must be nonnegative integer codes or null.")
        out["agent"] = agent.astype("Int64").astype("string").fillna("NoAgent").astype(object)
        for column in CATEGORICAL_COLUMNS:
            if column == "agent":
                continue
            out[column] = (out[column].astype("string").str.strip().replace("", pd.NA)
                           .fillna("Unknown").astype(object))
        return out

    def fit(self, X, y=None):
        self._clean(X)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        return self._clean(X)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        return np.asarray(MODEL_COLUMNS, dtype=object)


def make_preprocessor(*, scale_numeric: bool = True) -> Pipeline:
    """Return a NEW unfitted sparse preprocessor; cloning works with sklearn CV.

    A median is learned separately for each numeric field within each training
    fold. If a field is entirely missing in a training fold, sklearn retains it
    with a zero fallback; children/ADR indicators still identify missingness.
    Scaled numeric branches suit logistic regression. Tree models can request
    scale_numeric=False; all other rules and frozen folds remain identical.
    """
    scaling = StandardScaler() if scale_numeric else "passthrough"
    logarithmic = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", scaling),
    ])
    numerical = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler() if scale_numeric else "passthrough"),
    ])
    columns = ColumnTransformer([
        ("log_numeric", logarithmic, list(LOG_COLUMNS)),
        ("numeric", numerical, list(NUMERIC_COLUMNS)),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True,
                                       dtype=np.float64), list(CATEGORICAL_COLUMNS)),
        ("missing", MissingIndicator(features="all", error_on_new=False),
                   list(MISSING_INDICATOR_COLUMNS)),
    ], remainder="drop", sparse_threshold=1.0, verbose_feature_names_out=True)
    return Pipeline([("domain", BookingDomainCleaner()), ("columns", columns)])
