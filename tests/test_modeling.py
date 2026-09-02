"""Step 11 positive-class metrics, train-only pipelines, and baseline controls."""
import unittest
import joblib
import numpy as np
from sklearn.base import clone
from src.modeling import make_model_pipeline,classification_metrics,cancellation_probability
from test_feature_engineering import fixture


class ModelingTests(unittest.TestCase):
    def test_positive_class_threshold_confusion_counts_and_auc(self):
        values=classification_metrics([0,0,1,1],[.1,.5,.4,.9])
        self.assertEqual([values[k] for k in ['tn','fp','fn','tp']],[1,1,1,1])
        for metric in ['f1','accuracy','precision','recall']: self.assertEqual(values[metric],.5)
        self.assertEqual(values['roc_auc'],.75)

    def test_majority_baseline_ignores_features_and_does_not_learn_validation_labels(self):
        X=fixture(); y=np.array([0,0,0,1])
        baseline=make_model_pipeline('majority','none').fit(X,y)
        changed=X.astype(object).copy(); changed[:]='unseen'
        np.testing.assert_array_equal(cancellation_probability(baseline,changed),np.zeros(4))
        values=classification_metrics([0,1,1,1],cancellation_probability(baseline,changed))
        self.assertEqual(values['f1'],0); self.assertEqual(values['roc_auc'],.5)

    def test_single_class_baselines_resolve_class_one_without_assuming_column_order(self):
        X=fixture()
        for label in [0,1]:
            pipe=make_model_pipeline('majority','none').fit(X,np.full(len(X),label))
            np.testing.assert_array_equal(cancellation_probability(pipe,X),np.full(len(X),label))
        self.assertIsNone(classification_metrics([0,0],[0,0])['roc_auc'])

    def test_raw_model_pipelines_clone_and_keep_training_state_after_prediction(self):
        X=fixture(); X['adr']=[10,20,30,40]; X['children']=[0,1,2,3]
        y=np.array([0,0,1,1])
        for family in ['logistic_regression','random_forest']:
            for mode in ['full','selected']:
                pipe=clone(make_model_pipeline(family,mode))
                if family=='random_forest': pipe.set_params(model__n_estimators=5,model__n_jobs=1)
                pipe.fit(X,y)
                state=joblib.hash(pipe.named_steps['representation'])
                V=X.copy(); V['country']='UNSEEN'; V['adr']=1e7
                probabilities=cancellation_probability(pipe,V)
                self.assertEqual(state,joblib.hash(pipe.named_steps['representation']))
                self.assertTrue(np.isfinite(probabilities).all())
                self.assertTrue(((probabilities>=0)&(probabilities<=1)).all())
                with self.assertRaises(ValueError): cancellation_probability(pipe,V.assign(is_canceled=y))

    def test_metrics_reject_misalignment_unknown_labels_and_invalid_probabilities(self):
        for target,probability in [([0,1],[.5]),([0,2],[.2,.5]),([0,1],[np.nan,.5]),
                                   ([0,1],[-.1,.5]),([],[])]:
            with self.assertRaises(ValueError): classification_metrics(target,probability)
        with self.assertRaises(ValueError): make_model_pipeline('unknown')
        with self.assertRaises(ValueError): make_model_pipeline('majority','selected')
        with self.assertRaises(ValueError): make_model_pipeline('random_forest','pca')


if __name__=='__main__': unittest.main()
