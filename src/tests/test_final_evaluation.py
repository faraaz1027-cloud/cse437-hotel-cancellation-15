"""evaluation frozen protocol, diagnostics, error examples and serialization."""
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from src.final_evaluation import (PROTOCOL, add_diagnostic_columns, coefficient_table,
    grouped_metrics, probability_diagnostics, select_error_examples, write_protocol)
from src.modeling import cancellation_probability, make_model_pipeline
from test_feature_engineering import fixture


class FinalEvaluationTests(unittest.TestCase):
    def diagnostic_fixture(self):
        n=24;X=pd.concat([fixture()]*6,ignore_index=True)
        X['hotel']=['Resort Hotel']*12+['City Hotel']*12
        X['lead_time']=np.arange(n)*20;X['deposit_type']=np.tile(['No Deposit','Non Refund'],12)
        X['market_segment']=np.tile(['Direct','Online TA','Groups'],8)
        X['customer_type']=np.tile(['Transient','Contract'],12)
        meta=pd.DataFrame({'cohort_row':np.arange(n),'source_row_id':np.arange(100,100+n),
                           'arrival_date':['2017-05-01']*n})
        actual=np.tile([0,1],12);prob=np.linspace(.01,.99,n)
        return add_diagnostic_columns(X,meta,actual,prob)

    def test_protocol_rejects_post_access_change(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'protocol.json';write_protocol(p);write_protocol(p)
            changed=json.loads(p.read_text());changed['threshold']=.6;p.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):write_protocol(p)

    def test_diagnostics_align_threshold_and_cover_subgroups(self):
        d=self.diagnostic_fixture();self.assertEqual(len(d),24)
        self.assertTrue((d.predicted.eq(d.cancellation_probability.ge(.5))).all())
        groups=grouped_metrics(d)
        for dimension in PROTOCOL['subgroups']:
            self.assertEqual(groups.loc[groups.dimension.eq(dimension),'rows'].sum(),24)
        self.assertTrue((groups[['rows','tn','fp','fn','tp']].notna()).all().all())
        self.assertTrue(groups.loc[groups.roc_auc.notna(),'roc_auc'].between(0,1).all())

    def test_probability_bins_reconcile_without_data_dependent_edges(self):
        result=probability_diagnostics(self.diagnostic_fixture())
        self.assertEqual(result.probability_bin.tolist(),[f'{i/10:.1f}–{(i+1)/10:.1f}' for i in range(10)])
        self.assertEqual(result.rows.sum(),24);self.assertEqual(result.errors.sum(),12)

    def test_error_examples_are_distinct_and_follow_frozen_extremes(self):
        d=self.diagnostic_fixture();examples=select_error_examples(d,per_slice=2)
        self.assertEqual(len(examples),8);self.assertFalse(examples.source_row_id.duplicated().any())
        self.assertEqual(set(examples.error_type),{'FP','FN'})
        for error in ['FP','FN']:
            self.assertEqual((examples.error_type==error).sum(),4)
            self.assertIn('most_confident_'+error,set(examples.example_reason))
            self.assertIn('closest_to_threshold_'+error,set(examples.example_reason))

    def test_model_roundtrip_coefficients_and_probabilities(self):
        X=fixture();X['adr']=[10,20,30,40];X['children']=[0,1,2,3];y=np.array([0,0,1,1])
        pipe=make_model_pipeline('logistic_regression','selected').set_params(model__class_weight='balanced').fit(X,y)
        coefficients=coefficient_table(pipe)
        self.assertEqual(len(coefficients),pipe.named_steps['model'].coef_.shape[1])
        self.assertFalse(coefficients.feature.duplicated().any())
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'model.joblib';joblib.dump(pipe,p);restored=joblib.load(p)
            np.testing.assert_allclose(cancellation_probability(pipe,X),cancellation_probability(restored,X))

    def test_diagnostics_reject_misalignment_and_do_not_export_status_fields(self):
        X=fixture();meta=pd.DataFrame({'cohort_row':range(4),'source_row_id':range(4),'arrival_date':['x']*4})
        with self.assertRaises(ValueError):add_diagnostic_columns(X,meta.iloc[:3],[0]*4,[.1]*4)
        result=add_diagnostic_columns(X,meta,[0,1,0,1],[.1,.9,.6,.4])
        self.assertNotIn('reservation_status',result.columns);self.assertNotIn('reservation_status_date',result.columns)


if __name__=='__main__':unittest.main()
