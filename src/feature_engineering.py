"""Step 9 deterministic derived features and a fold-fitted encoding factory.

Use make_feature_preprocessor() inside a modeling pipeline passed to the frozen
development CV. No target, group identity, fitted full-cohort matrix, or held-out
summary is needed to define these features.
"""
from __future__ import annotations

import calendar
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .preprocessing import (BookingDomainCleaner, LOG_COLUMNS, NUMERIC_COLUMNS,
                            CATEGORICAL_COLUMNS, MISSING_INDICATOR_COLUMNS)

MONTH_NUMBERS = {name: number for number, name in enumerate(calendar.month_name) if name}
DERIVED_COLUMNS = ("total_nights", "total_guests", "previous_bookings_total",
                   "has_booking_history", "previous_cancellation_share",
                   "company_code_recorded", "arrival_month_sin", "arrival_month_cos")
FEATURE_LOG_COLUMNS = LOG_COLUMNS + ("total_nights", "total_guests", "previous_bookings_total")
FEATURE_NUMERIC_COLUMNS = NUMERIC_COLUMNS + DERIVED_COLUMNS[3:]
FEATURE_CATEGORICAL_COLUMNS = tuple(c for c in CATEGORICAL_COLUMNS if c != "arrival_date_month")
FEATURE_MISSING_COLUMNS = MISSING_INDICATOR_COLUMNS + (
    "total_nights", "total_guests", "previous_cancellation_share")
FEATURE_COLUMNS = FEATURE_LOG_COLUMNS + FEATURE_NUMERIC_COLUMNS + FEATURE_CATEGORICAL_COLUMNS


class BookingFeatureEngineer(TransformerMixin, BaseEstimator):
    """Apply existing fixed domain rules, then derive eight explicit fields.

    Retain 24 Step 7 source fields and replace month names with a sine/cosine
    pair. Aggregate missing values propagate before training-only imputation.
    Company presence means only a recorded code, not verified corporate payment.
    """

    def _derive(self, X, cleaner):
        out = cleaner.transform(X)
        if "company" not in X:
            raise ValueError("Step 9 requires the original company column to derive code presence.")
        company = pd.to_numeric(X["company"], errors="raise")
        present = company.dropna()
        if not np.isfinite(present).all() or present.lt(0).any() or present.mod(1).ne(0).any():
            raise ValueError("Company codes must be nonnegative integers or null.")
        month = out["arrival_date_month"].map(MONTH_NUMBERS)
        if month.isna().any():
            raise ValueError("Arrival month must be a recognized English calendar month.")
        # Direct addition propagates missing components; zero nights remain zero.
        out["total_nights"] = out.stays_in_weekend_nights + out.stays_in_week_nights
        out["total_guests"] = out.adults + out.children + out.babies
        history = out.previous_cancellations + out.previous_bookings_not_canceled
        out["previous_bookings_total"] = history
        out["has_booking_history"] = history.gt(0).astype(float).mask(history.isna())
        out["previous_cancellation_share"] = (out.previous_cancellations / history.where(history.gt(0))).mask(history.eq(0), 0.0)
        out["company_code_recorded"] = company.notna().astype(float)
        angle = 2 * np.pi * (month - 1) / 12
        out["arrival_month_sin"] = np.sin(angle)
        out["arrival_month_cos"] = np.cos(angle)
        return out.loc[:, list(FEATURE_COLUMNS)]

    def fit(self, X, y=None):
        # y is deliberately unused, as required by sklearn's transformer API.
        cleaner = BookingDomainCleaner().fit(X)
        self._derive(X, cleaner)
        self.cleaner_ = cleaner
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "cleaner_")
        return self._derive(X, self.cleaner_)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "cleaner_")
        return np.asarray(FEATURE_COLUMNS, dtype=object)


def make_feature_preprocessor(*, scale_numeric: bool = True) -> Pipeline:
    """New unfitted Step 9 pipeline, explicitly extending Step 7's schema.

    Missing numeric values use training medians (zero if entirely missing).
    Sine/cosine and ratios are never logged. The unscaled variant suits trees.
    All learned transforms must be refitted within each training fold.
    """
    def numeric_branch(log=False):
        steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
        if log:
            steps.append(("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
        steps.append(("scale", StandardScaler() if scale_numeric else "passthrough"))
        return Pipeline(steps)
    columns = ColumnTransformer([
        ("log_numeric", numeric_branch(log=True), list(FEATURE_LOG_COLUMNS)),
        ("numeric", numeric_branch(), list(FEATURE_NUMERIC_COLUMNS)),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True,
                                       dtype=np.float64), list(FEATURE_CATEGORICAL_COLUMNS)),
        ("missing", MissingIndicator(features="all", error_on_new=False), list(FEATURE_MISSING_COLUMNS)),
    ], remainder="drop", sparse_threshold=1.0, verbose_feature_names_out=True)
    return Pipeline([("features", BookingFeatureEngineer()), ("columns", columns)])
