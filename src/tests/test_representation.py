"""representation comparison known-signal selection, PCA geometry, and training-state isolation."""
import unittest
import numpy as np
from scipy import sparse
from sklearn.base import clone
from src.representation import EncodedRepresentation, BookingRepresentation
from src.preprocessing_audit import fitted_state
from test_feature_engineering import fixture


class RepresentationTests(unittest.TestCase):
    def matrix(self):
        rng=np.random.default_rng(42)
        y=np.tile([0,1], 30)
        signal=y+.05*rng.normal(size=len(y))
        X=sparse.csr_matrix(np.column_stack([signal,signal, rng.normal(size=len(y)),
                                           np.ones(len(y)),rng.integers(0,2,len(y))]))
        names=('numeric__signal','numeric__copy','numeric__noise','missing__constant','categorical__A')
        return X,y,names

    def test_selection_keeps_signal_drops_constants_and_breaks_ties_stably(self):
        X,y,names=self.matrix()
        rep=EncodedRepresentation(names,mode='selected',percentile=25).fit(X,y)
        self.assertEqual(rep.get_feature_names_out().tolist(),['numeric__signal'])
        self.assertNotIn(3,rep.selected_indices_)
        self.assertEqual(rep.transform(X).shape,(60,1))

    def test_pca_centers_numeric_block_preserves_categories_and_retains_variance(self):
        X,y,names=self.matrix()
        rep=EncodedRepresentation(names,mode='pca').fit(X,y)
        Z=rep.transform(X).toarray()
        k=rep.pca_.n_components_
        self.assertGreaterEqual(rep.pca_.explained_variance_ratio_.sum(),.95)
        np.testing.assert_allclose(Z[:,:k].mean(axis=0),0,atol=1e-12)
        np.testing.assert_allclose(Z[:,k:],X.toarray()[:,3:])
        self.assertLessEqual(rep.dense_numeric_training_bytes_,60*3*8)

    def test_validation_values_do_not_refit_selection_or_pca(self):
        X,y,names=self.matrix()
        rep=EncodedRepresentation(names,mode='selected_pca').fit(X,y)
        selected=rep.selected_indices_.copy(); components=rep.pca_.components_.copy()
        mean=rep.pca_.mean_.copy(); score=rep.scores_.copy()
        rep.transform(X*10000)
        np.testing.assert_array_equal(selected,rep.selected_indices_)
        np.testing.assert_array_equal(components,rep.pca_.components_)
        np.testing.assert_array_equal(mean,rep.pca_.mean_)
        np.testing.assert_array_equal(score,rep.scores_)

    def test_label_changes_affect_supervised_selection_but_not_pca_only(self):
        rng=np.random.default_rng(9); a=np.tile([0,0,1,1],20); b=np.tile([0,1,0,1],20)
        X=sparse.csr_matrix(np.column_stack([a+.01*rng.normal(size=80),b+.01*rng.normal(size=80)]))
        names=('numeric__a','numeric__b')
        self.assertEqual(EncodedRepresentation(names,'selected',50).fit(X,a).get_feature_names_out().tolist(),['numeric__a'])
        self.assertEqual(EncodedRepresentation(names,'selected',50).fit(X,b).get_feature_names_out().tolist(),['numeric__b'])
        pa=EncodedRepresentation(names,'pca').fit(X,a); pb=EncodedRepresentation(names,'pca').fit(X,b)
        np.testing.assert_allclose(pa.pca_.components_,pb.pca_.components_)

    def test_invalid_labels_schema_mode_and_all_constants_fail(self):
        X,y,names=self.matrix()
        for labels in [None,np.zeros(len(y)),y[:-1]]:
            with self.assertRaises(ValueError): EncodedRepresentation(names,'selected').fit(X,labels)
        for mode in ['bad']:
            with self.assertRaises(ValueError): EncodedRepresentation(names,mode).fit(X,y)
        with self.assertRaises(ValueError): EncodedRepresentation(names,'selected').fit(sparse.csr_matrix(np.ones(X.shape)),y)
        with self.assertRaises(ValueError): EncodedRepresentation(names).fit(X[:,:2],y)

    def test_raw_wrapper_is_cloneable_and_preserves_training_only_preprocessing(self):
        X=fixture(); X['children']=[0,1,2,3]; X['adr']=[10,20,30,40]
        y=np.array([0,1,0,1])
        for mode in ['full','selected','pca','selected_pca']:
            rep=clone(BookingRepresentation(mode=mode)).fit(X,y)
            state=fitted_state(rep.preprocessor_)
            V=X.copy(); V['adr']=1e6; V['arrival_date_month']='June'
            out=rep.transform(V)
            self.assertTrue(np.isfinite(out.data).all())
            self.assertEqual(state,fitted_state(rep.preprocessor_))
            self.assertEqual(out.shape[1],len(rep.get_feature_names_out()))
        with self.assertRaises(ValueError): BookingRepresentation('pca',scale_numeric=False).fit(X,y)
        with self.assertRaises(ValueError): BookingRepresentation('selected').fit(X.assign(is_canceled=y),y)


if __name__=='__main__': unittest.main()
