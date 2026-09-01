"""Boundary and leakage invariants for eligibility; no model tests yet."""
import unittest

import numpy as np
import pandas as pd

from src.eligibility import prepare_records


class EligibilityTests(unittest.TestCase):
    def fixture(self):
        base = dict(is_canceled=0, reservation_status="Check-Out",
                    reservation_status_date="2016-08-02", adults=2,
                    children=0.0, babies=0, adr=100.0,
                    arrival_date_year=2016, arrival_date_month="August",
                    arrival_date_day_of_month=1,
                    stays_in_weekend_nights=0, stays_in_week_nights=1)
        rows = [base.copy() for _ in range(6)]
        rows[0].update(adults=0)  # known zero guests: exclude
        rows[1].update(adults=0, children=np.nan)  # unknown total: retain
        rows[2].update(adults=0, children=1)  # child-only: retain and flag
        rows[3].update(adr=-6.38, stays_in_week_nights=0)  # retain and flag
        rows[5].update(is_canceled=1, reservation_status="Canceled",
                       reservation_status_date="2016-06-01")
        return pd.DataFrame(rows)

    def test_unknown_guests_and_anomalies_are_not_silently_deleted(self):
        X, y, meta, excluded, summary = prepare_records(self.fixture())
        self.assertEqual(excluded.source_row_id.tolist(), [0])
        self.assertEqual(meta.source_row_id.tolist(), [1, 2, 3, 4, 5])
        self.assertTrue(meta.loc[0, "unknown_guest_total"])
        self.assertTrue(meta.loc[1, "zero_adults_positive_guests"])
        self.assertTrue(meta.loc[2, "negative_adr"])
        self.assertTrue(meta.loc[2, "zero_total_nights"])
        self.assertEqual(X.loc[2, "adr"], -6.38)

    def test_outcome_changes_do_not_change_selection_or_grouping(self):
        raw = self.fixture()
        X, _, meta, excluded, summary = prepare_records(raw)
        changed = raw.copy(deep=True)
        changed["is_canceled"] = 1 - changed["is_canceled"]
        changed["reservation_status"] = "outcome replaced"
        changed["reservation_status_date"] = "2099-01-01"
        X2, _, meta2, excluded2, _ = prepare_records(changed)
        pd.testing.assert_frame_equal(X, X2)
        pd.testing.assert_frame_equal(meta, meta2)
        pd.testing.assert_frame_equal(excluded, excluded2)
        self.assertEqual(meta.loc[3, "duplicate_group_id"], meta.loc[4, "duplicate_group_id"])
        self.assertEqual(summary["candidate_groups_with_conflicting_labels"], 1)

    def test_input_is_unchanged_and_outcome_columns_are_absent(self):
        raw = self.fixture()
        original = raw.copy(deep=True)
        X, y, *_ = prepare_records(raw)
        pd.testing.assert_frame_equal(raw, original)
        self.assertFalse(set(["is_canceled", "reservation_status", "reservation_status_date"]) & set(X))
        self.assertEqual(y.columns.tolist(), ["is_canceled"])

    def test_unexpected_negative_guest_count_fails_for_review(self):
        raw = self.fixture()
        raw.loc[0, "adults"] = -1
        with self.assertRaisesRegex(ValueError, "Negative guest count"):
            prepare_records(raw)


if __name__ == "__main__":
    unittest.main()
