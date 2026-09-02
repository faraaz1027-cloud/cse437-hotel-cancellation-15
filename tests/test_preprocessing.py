"""Step 7 leakage, schema, and missing/outlier behavior on synthetic rows."""
import unittest

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone

from src.preprocessing import (BookingDomainCleaner, make_preprocessor, MODEL_COLUMNS,
                               NUMERIC_COLUMNS, LOG_COLUMNS, CATEGORICAL_COLUMNS,
                               POLICY_EXCLUSIONS)
from src.preprocessing_audit import fitted_state


class PreprocessingTests(unittest.TestCase):
    def fixture(self):
        row = {col: 1 for col in LOG_COLUMNS + NUMERIC_COLUMNS}
        row.update({col: "category_a" for col in CATEGORICAL_COLUMNS})
        row.update(agent=7.0, country="PRT", company=np.nan, assigned_room_type="A",
                   booking_changes=0, days_in_waiting_list=0)
        result = pd.DataFrame([row.copy() for _ in range(3)])
        result["adr"] = [100.0, -1.0, 300.0]
        result["children"] = [0.0, np.nan, 4.0]
        result["country"] = ["PRT", None, "Undefined"]
        result["agent"] = [7.0, None, 8.0]
        return result

    def test_training_medians_and_scalers_ignore_validation_extremes(self):
        train = self.fixture()
        pipe = make_preprocessor().fit(train)
        state = fitted_state(pipe)
        self.assertEqual(state["numeric"]["medians"][NUMERIC_COLUMNS.index("children")], 2)
        self.assertEqual(state["log_numeric"]["medians"][LOG_COLUMNS.index("adr")], 200)
        self.assertAlmostEqual(state["log_numeric"]["means"][LOG_COLUMNS.index("adr")],
                               np.log1p([100, 200, 300]).mean())
        validation = train.iloc[:1].copy()
        validation["adr"] = 1e12
        validation["children"] = 10000
        validation["country"] = "UNSEEN"
        validation["agent"] = 999999
        matrix = pipe.transform(validation)
        self.assertTrue(sparse.issparse(matrix))
        self.assertTrue(np.isfinite(matrix.data).all())
        self.assertEqual(state, fitted_state(pipe))

    def test_unknown_categories_are_safe_and_not_learned(self):
        pipe = make_preprocessor().fit(self.fixture())
        validation = self.fixture().iloc[:1].copy()
        validation["country"] = "UNSEEN"
        clean = pipe.named_steps["domain"].transform(validation)
        encoder = pipe.named_steps["columns"].named_transformers_["categorical"]
        matrix = encoder.transform(clean[list(CATEGORICAL_COLUMNS)]).toarray()
        i = CATEGORICAL_COLUMNS.index("country")
        start = sum(len(c) for c in encoder.categories_[:i])
        self.assertEqual(matrix[0, start:start+len(encoder.categories_[i])].sum(), 0)
        self.assertNotIn("UNSEEN", encoder.categories_[i])

    def test_fixed_domain_rules_preserve_rows_source_and_nonnegative_prices(self):
        train = self.fixture()
        train.loc[0, "adr"] = 0
        train.loc[2, "adr"] = 5400
        original = train.copy(deep=True)
        clean = BookingDomainCleaner().fit_transform(train)
        pd.testing.assert_frame_equal(train, original)
        self.assertEqual(clean.index.tolist(), train.index.tolist())
        self.assertEqual(clean.columns.tolist(), list(MODEL_COLUMNS))
        self.assertEqual(clean.loc[0, "adr"], 0)
        self.assertTrue(np.isnan(clean.loc[1, "adr"]))
        self.assertEqual(clean.loc[2, "adr"], 5400)
        self.assertEqual(clean.loc[1, "country"], "Unknown")
        self.assertEqual(clean.loc[2, "country"], "Undefined")
        self.assertEqual(clean.loc[1, "agent"], "NoAgent")
        self.assertEqual(clean.loc[0, "agent"], "7")

    def test_excluded_fields_cannot_influence_encoded_output(self):
        train = self.fixture()
        pipe = make_preprocessor().fit(train)
        changed = train.copy()
        for col in POLICY_EXCLUSIONS:
            changed[col] = "altered excluded field"
        np.testing.assert_allclose(pipe.transform(train).toarray(), pipe.transform(changed).toarray())

    def test_target_metadata_and_undeclared_features_are_rejected(self):
        for col in ["is_canceled", "reservation_status", "reservation_status_date", "source_row_id", "new_feature"]:
            with self.subTest(column=col), self.assertRaises(ValueError):
                make_preprocessor().fit(self.fixture().assign(**{col: 0}))

    def test_all_missing_fields_keep_schema_and_indicators(self):
        train = self.fixture()
        train["children"] = np.nan
        train["adr"] = np.nan
        pipe = make_preprocessor(scale_numeric=False)
        matrix = pipe.fit_transform(train).toarray()
        names = pipe.get_feature_names_out().tolist()
        self.assertTrue(np.isfinite(matrix).all())
        np.testing.assert_array_equal(matrix[:, names.index("numeric__children")], 0)
        np.testing.assert_array_equal(matrix[:, names.index("log_numeric__adr")], 0)
        np.testing.assert_array_equal(matrix[:, -2:], np.ones((3, 2)))

    def test_scaled_and_tree_factories_are_cloneable_and_have_identical_schema(self):
        train = self.fixture()
        scaled = clone(make_preprocessor()).fit(train)
        unscaled = clone(make_preprocessor(scale_numeric=False)).fit(train)
        np.testing.assert_array_equal(scaled.get_feature_names_out(), unscaled.get_feature_names_out())
        names = unscaled.get_feature_names_out().tolist()
        np.testing.assert_allclose(unscaled.transform(train).toarray()[:, names.index("log_numeric__adr")],
                                   np.log1p([100, 200, 300]))

    def test_fractional_agent_and_negative_guest_values_fail(self):
        for col, value in [("agent", 1.5), ("children", -1)]:
            invalid = self.fixture()
            invalid.loc[0, col] = value
            with self.subTest(column=col), self.assertRaises(ValueError):
                make_preprocessor().fit(invalid)


if __name__ == "__main__":
    unittest.main()
