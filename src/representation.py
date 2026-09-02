"""Step 10 fold-fitted supervised selection and centered numeric-block PCA."""
from __future__ import annotations

import math
import warnings
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.utils.validation import check_is_fitted

from .feature_engineering import make_feature_preprocessor

MODES = ('full', 'selected', 'pca', 'selected_pca')


class EncodedRepresentation(TransformerMixin, BaseEstimator):
    """Select 75% of nonconstant encoded fields, optionally reduce numerics.

    ANOVA F statistics rank associations only: p-values are discarded. Ties
    use original encoded order. Constant means training variance <= 1e-12.
    PCA is centered, full-SVD PCA retaining >=95% of selected numeric variance;
    it never densifies categorical/indicator blocks. Modes expose independent
    selection/reduction comparisons without altering the upstream features.
    """
    def __init__(self, feature_names, mode='full', percentile=75, variance_target=.95):
        self.feature_names = feature_names
        self.mode = mode
        self.percentile = percentile
        self.variance_target = variance_target

    def _matrix(self, X):
        matrix = sparse.csr_matrix(X, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError('Encoded columns do not match the supplied feature names.')
        if not np.isfinite(matrix.data).all():
            raise ValueError('Encoded inputs must be finite.')
        return matrix

    def fit(self, X, y=None):
        if self.mode not in MODES or not 0 < self.percentile <= 100 or not 0 < self.variance_target < 1:
            raise ValueError('Invalid representation mode, percentile, or variance target.')
        matrix = self._matrix(X)
        names = np.asarray(self.feature_names, dtype=object)
        if len(set(names)) != len(names):
            raise ValueError('Feature names must be unique.')
        self.n_features_in_ = matrix.shape[1]
        self.training_rows_ = matrix.shape[0]
        self.variances_ = mean_variance_axis(matrix, axis=0)[1]
        self.scores_ = np.full(matrix.shape[1], np.nan)
        if self.mode in ('selected', 'selected_pca'):
            target = np.asarray(y)
            if target.ndim != 1 or len(target) != len(matrix.indptr)-1 or set(np.unique(target)) != {0, 1}:
                raise ValueError('Supervised selection requires aligned binary training labels with both classes.')
            eligible = np.flatnonzero(self.variances_ > 1e-12)
            if not len(eligible):
                raise ValueError('No nonconstant training features remain.')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                scores, _ = f_classif(matrix[:, eligible], target)
            scores = np.nan_to_num(scores, nan=0.0, posinf=np.finfo(float).max, neginf=0.0)
            scores = np.maximum(scores, 0)
            self.scores_[eligible] = scores
            count = max(1, math.ceil(len(eligible) * self.percentile / 100))
            ranked = eligible[np.argsort(-scores, kind='stable')]
            self.selected_indices_ = np.sort(ranked[:count])
        else:
            self.selected_indices_ = np.arange(matrix.shape[1])
        kept_names = names[self.selected_indices_]
        numeric = np.array([n.startswith(('log_numeric__', 'numeric__')) for n in kept_names])
        self.numeric_positions_ = np.flatnonzero(numeric)
        self.other_positions_ = np.flatnonzero(~numeric)
        self.selected_names_ = kept_names
        self.pca_ = None
        self.dense_numeric_training_bytes_ = 0
        if self.mode in ('pca', 'selected_pca'):
            if not len(self.numeric_positions_):
                raise ValueError('No numeric features remain for PCA.')
            block = matrix[:, self.selected_indices_][:, self.numeric_positions_].toarray()
            self.dense_numeric_training_bytes_ = block.nbytes
            if np.var(block, axis=0).sum() <= 1e-12:
                raise ValueError('Numeric training block has no variance for PCA.')
            self.pca_ = PCA(n_components=self.variance_target, svd_solver='full').fit(block)
            self.output_names_ = np.concatenate([
                np.array([f'pca_numeric__component_{i+1:02d}' for i in range(self.pca_.n_components_)]),
                kept_names[self.other_positions_]])
        else:
            self.output_names_ = kept_names.copy()
        return self

    def transform(self, X):
        check_is_fitted(self, 'selected_indices_')
        matrix = self._matrix(X)[:, self.selected_indices_]
        if self.pca_ is None:
            return matrix
        components = self.pca_.transform(matrix[:, self.numeric_positions_].toarray())
        return sparse.hstack([sparse.csr_matrix(components), matrix[:, self.other_positions_]], format='csr')

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'output_names_')
        return self.output_names_.copy()


class BookingRepresentation(TransformerMixin, BaseEstimator):
    """Cloneable raw-candidate transformer for future sklearn model pipelines.

    Fitting this object fits preprocessing, selection and PCA on only the rows
    and labels provided to fit. Place it inside the pipeline passed to CV.
    PCA modes require scaling even when used with a tree classifier.
    """
    def __init__(self, mode='full', percentile=75, variance_target=.95, scale_numeric=True):
        self.mode = mode
        self.percentile = percentile
        self.variance_target = variance_target
        self.scale_numeric = scale_numeric

    def fit(self, X, y=None):
        if self.mode in ('pca', 'selected_pca') and not self.scale_numeric:
            raise ValueError('PCA requires training-fold scaled numeric inputs.')
        self.preprocessor_ = make_feature_preprocessor(scale_numeric=self.scale_numeric)
        matrix = self.preprocessor_.fit_transform(X, y)
        self.representation_ = EncodedRepresentation(
            tuple(self.preprocessor_.get_feature_names_out()), self.mode,
            self.percentile, self.variance_target).fit(matrix, y)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, 'representation_')
        return self.representation_.transform(self.preprocessor_.transform(X))

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'representation_')
        return self.representation_.get_feature_names_out()
