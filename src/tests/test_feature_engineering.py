"""feature engineering formulas, missingness, leakage rejection and fold-state isolation."""
import unittest
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone

from src.feature_engineering import (BookingFeatureEngineer, make_feature_preprocessor,
    FEATURE_COLUMNS, FEATURE_LOG_COLUMNS, FEATURE_NUMERIC_COLUMNS)
from src.preprocessing import LOG_COLUMNS, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS
from src.preprocessing_audit import fitted_state


def fixture():
    row = {c: 1 for c in LOG_COLUMNS + NUMERIC_COLUMNS}
    row.update({c: "A" for c in CATEGORICAL_COLUMNS})
    row.update(arrival_date_month="January", agent=1, company=np.nan,
               assigned_room_type="A", booking_changes=0, days_in_waiting_list=0,
               children=0, babies=0, adults=2, previous_cancellations=0,
               previous_bookings_not_canceled=0, adr=100)
    return pd.DataFrame([row.copy() for _ in range(4)], index=[11, 7, 4, 99])


def dense(matrix):
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


class FeatureEngineeringTests(unittest.TestCase):
    def test_exact_totals_ratio_and_zero_history_distinction(self):
        X = fixture()
        X["previous_cancellations"] = [0, 0, 2, 3]
        X["previous_bookings_not_canceled"] = [0, 3, 6, 0]
        X["stays_in_week_nights"] = [0, 3, 4, 1]
        X["stays_in_weekend_nights"] = [0, 2, 0, 1]
        before = X.copy(deep=True)
        Z = BookingFeatureEngineer().fit_transform(X)
        np.testing.assert_allclose(Z.total_nights, [0, 5, 4, 2])
        np.testing.assert_allclose(Z.total_guests, 2)
        np.testing.assert_allclose(Z.previous_bookings_total, [0, 3, 8, 3])
        np.testing.assert_allclose(Z.has_booking_history, [0, 1, 1, 1])
        np.testing.assert_allclose(Z.previous_cancellation_share, [0, 0, .25, 1])
        pd.testing.assert_frame_equal(X, before)
        self.assertEqual(Z.index.tolist(), X.index.tolist())
        self.assertEqual(Z.columns.tolist(), list(FEATURE_COLUMNS))

    def test_missing_components_remain_unknown_until_fold_imputation(self):
        X = fixture()
        X.loc[7, "children"] = np.nan
        X.loc[4, "previous_cancellations"] = np.nan
        X.loc[99, "stays_in_week_nights"] = np.nan
        Z = BookingFeatureEngineer().fit_transform(X)
        self.assertTrue(np.isnan(Z.loc[7, "total_guests"]))
        self.assertTrue(Z.loc[4, ["previous_bookings_total", "has_booking_history", "previous_cancellation_share"]].isna().all())
        self.assertTrue(np.isnan(Z.loc[99, "total_nights"]))

    def test_company_codes_are_presence_only_and_month_wrap_is_continuous(self):
        X = fixture()
        X["company"] = [np.nan, 0, 42, 999]
        X["arrival_date_month"] = ["December", "January", "February", "July"]
        Z = BookingFeatureEngineer().fit_transform(X)
        np.testing.assert_allclose(Z.company_code_recorded, [0, 1, 1, 1])
        xy = Z[["arrival_month_sin", "arrival_month_cos"]].to_numpy()
        np.testing.assert_allclose((xy**2).sum(axis=1), 1)
        self.assertAlmostEqual(np.linalg.norm(xy[0]-xy[1]), np.linalg.norm(xy[1]-xy[2]))
        changed = X.copy(); changed.loc[4, "company"] = 123456
        pd.testing.assert_frame_equal(Z, BookingFeatureEngineer().fit_transform(changed))

    def test_validation_extremes_and_unseen_month_do_not_change_training_state(self):
        X = fixture(); X["children"] = [0, 2, 4, np.nan]
        pipe = make_feature_preprocessor().fit(X)
        state = fitted_state(pipe)
        self.assertEqual(state["log_numeric"]["medians"][FEATURE_LOG_COLUMNS.index("total_guests")], 4)
        V = fixture().iloc[:1].copy()
        V["arrival_date_month"] = "June"; V["children"] = np.nan
        V["adr"] = 1e12; V["country"] = "UNSEEN"
        matrix = pipe.transform(V)
        self.assertTrue(np.isfinite(dense(matrix)).all())
        self.assertEqual(state, fitted_state(pipe))
        names = pipe.get_feature_names_out().tolist()
        self.assertFalse(any("categorical__arrival_date_month" in n for n in names))
        self.assertEqual(dense(matrix)[0, names.index("missing__missingindicator_total_guests")], 1)
        month = pipe.named_steps["features"].transform(V)
        self.assertAlmostEqual(month.arrival_month_sin.iloc[0], .5)

    def test_clone_variants_and_all_missing_totals_are_finite(self):
        X = fixture(); X["children"] = np.nan
        X["previous_cancellations"] = np.nan
        scaled = clone(make_feature_preprocessor()).fit(X)
        tree = clone(make_feature_preprocessor(scale_numeric=False)).fit(X)
        np.testing.assert_array_equal(scaled.get_feature_names_out(), tree.get_feature_names_out())
        for pipe in [scaled, tree]:
            self.assertTrue(np.isfinite(dense(pipe.transform(X))).all())
        names = tree.get_feature_names_out().tolist()
        self.assertTrue((dense(tree.transform(X))[:, names.index("log_numeric__total_guests")] == 0).all())

    def test_leakage_extras_invalid_month_and_invalid_company_are_rejected(self):
        for column in ["is_canceled", "reservation_status", "reservation_status_date",
                       "source_row_id", "duplicate_group_id", "total_guests", "unexpected"]:
            with self.subTest(column=column), self.assertRaises(ValueError):
                BookingFeatureEngineer().fit(fixture().assign(**{column: 0}))
        for column, value in [("arrival_date_month", "Unknown"), ("company", -1), ("company", 1.5), ("company", np.inf)]:
            X = fixture(); X.loc[11, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                BookingFeatureEngineer().fit(X)
        with self.assertRaises(ValueError):
            BookingFeatureEngineer().fit(fixture().drop(columns="company"))

    def test_outcomes_and_excluded_late_fields_do_not_determine_features(self):
        X = fixture()
        first = BookingFeatureEngineer().fit(X, [0, 0, 0, 0]).transform(X)
        X[["assigned_room_type", "booking_changes", "days_in_waiting_list"]] = "changed"
        second = BookingFeatureEngineer().fit(X, [1, 1, 1, 1]).transform(X)
        pd.testing.assert_frame_equal(first, second)


if __name__ == "__main__":
    unittest.main()
