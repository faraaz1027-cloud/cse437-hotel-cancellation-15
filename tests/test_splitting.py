"""Temporal boundaries, grouping, and sklearn index-alignment checks."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.splitting import make_assignments, check_assignments, development_cv, nearest_date_boundary, run_step6


class SplittingTests(unittest.TestCase):
    def fixture(self):
        # Unequal date counts and nonchronological source order expose row/date
        # confusion. Repeated groups are genuine copies within their date.
        dates = pd.date_range("2020-01-01", periods=24)
        rows = []
        for i, date in enumerate(dates):
            for j in range(1 + i % 4):
                rows.append({"source_row_id": len(rows), "arrival_date": date.strftime("%Y-%m-%d"),
                             "duplicate_group_id": i})
        return pd.DataFrame(rows).sample(frac=1, random_state=9).reset_index(drop=True)

    def test_temporal_and_group_invariants_with_unsorted_input(self):
        a, plan = make_assignments(self.fixture())
        self.assertEqual(plan["checks"]["duplicate_group_overlap"], 0)
        self.assertEqual(plan["checks"]["test_rows_in_cv"], 0)
        for column in ["partition", "cv_fold_1", "cv_fold_2", "cv_fold_3"]:
            self.assertEqual(a.groupby("duplicate_group_id")[column].nunique().max(), 1)
        dev = a.partition.eq("development")
        for train, val in development_cv(a):
            X_dev = a.loc[dev].reset_index(drop=True)
            self.assertLess(X_dev.iloc[train].arrival_date.max(), X_dev.iloc[val].arrival_date.min())
            self.assertFalse(set(train) & set(val))
            self.assertTrue((X_dev.iloc[val].partition == "development").all())

    def test_labels_and_input_order_cannot_change_row_membership(self):
        meta = self.fixture()
        first, _ = make_assignments(meta.assign(is_canceled=0))
        changed = meta.sample(frac=1, random_state=4).assign(is_canceled=1)
        second, _ = make_assignments(changed)
        cols = ["partition", "cv_fold_1", "cv_fold_2", "cv_fold_3"]
        pd.testing.assert_frame_equal(first.set_index("source_row_id")[cols].sort_index(),
                                      second.set_index("source_row_id")[cols].sort_index())

    def test_cross_date_group_and_duplicate_source_identity_are_rejected(self):
        meta = self.fixture()
        meta["duplicate_group_id"] = 0
        with self.assertRaisesRegex(ValueError, "spans arrival dates"):
            make_assignments(meta)
        meta = self.fixture()
        meta["source_row_id"] = 0
        with self.assertRaisesRegex(ValueError, "unique"):
            make_assignments(meta)

    def test_boundary_ties_choose_earlier_date(self):
        dates = pd.Series(pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]))
        self.assertEqual(nearest_date_boundary(dates, 0.5), pd.Timestamp("2020-01-02"))

    def test_test_contamination_is_rejected(self):
        a, _ = make_assignments(self.fixture())
        a.loc[a.partition.eq("test").idxmax(), "cv_fold_1"] = "train"
        with self.assertRaisesRegex(ValueError, "Test rows"):
            check_assignments(a)

    def test_insufficient_distinct_dates_are_rejected(self):
        meta = self.fixture().iloc[:2].copy()
        meta["arrival_date"] = "2020-01-01"
        with self.assertRaisesRegex(ValueError, "two distinct dates"):
            make_assignments(meta)

    def test_frozen_artifacts_reject_plan_changes_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed, output = root / 'processed', root / 'splits'
            processed.mkdir()
            meta = self.fixture()
            name = 'step5_metadata.csv.gz'
            meta.to_csv(processed / name, index=False, compression='gzip')
            digest = hashlib.sha256((processed / name).read_bytes()).hexdigest()
            (processed / 'step5_summary.json').write_text(json.dumps({
                'outputs': {name: {'sha256': digest}}, 'retained_rows': len(meta),
                'source_sha256': 'synthetic-fixture'}))
            with patch('src.splitting.METADATA_SHA256', digest):
                _, plan = run_step6(processed, output)
                plan_path = output / 'step6_split_plan.json'
                plan['primary_metric']['name'] = 'changed without review'
                plan_path.write_text(json.dumps(plan))
                before = {p.name: p.read_bytes() for p in output.iterdir()}
                with self.assertRaisesRegex(ValueError, 'Frozen plan differs'):
                    run_step6(processed, output)
                self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})


if __name__ == "__main__":
    unittest.main()
