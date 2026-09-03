"""model comparison unfitted baseline/classifier pipelines and consistent metrics."""
from __future__ import annotations
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from .representation import BookingRepresentation

MODEL_SETTINGS = {
    'logistic_regression': {'C':1.0,'solver':'lbfgs','max_iter':2000,'tol':1e-4,
                            'class_weight':None,'random_state':42},
    'random_forest': {'n_estimators':100,'criterion':'gini','max_depth':None,
        'min_samples_split':2,'min_samples_leaf':1,'max_features':'sqrt',
        'bootstrap':True,'class_weight':None,'random_state':42,'n_jobs':2},
}
CANDIDATES = (
    ('majority', 'majority', 'none'),
    ('lr_full', 'logistic_regression', 'full'),
    ('lr_selected', 'logistic_regression', 'selected'),
    ('rf_full', 'random_forest', 'full'),
    ('rf_selected', 'random_forest', 'selected'),
)


def make_model_pipeline(family, representation='selected'):
    """Return a new model, with all learned feature work inside the pipeline.

    The majority baseline learns a class from y and ignores X. LR uses scaled
    numerics; full/selected RF uses the verified unscaled variant. No resampling,
    class reweighting, model-parameter search or threshold tuning is performed.
    """
    if family=='majority':
        if representation!='none': raise ValueError('The majority baseline requires representation="none".')
        return Pipeline([('model',DummyClassifier(strategy='most_frequent',random_state=42))])
    if family not in MODEL_SETTINGS or representation not in ('full','selected'):
        raise ValueError('Unknown model comparison model family or representation.')
    model = (LogisticRegression(**MODEL_SETTINGS[family]) if family=='logistic_regression'
             else RandomForestClassifier(**MODEL_SETTINGS[family]))
    features=BookingRepresentation(mode=representation,percentile=75,variance_target=.95,
                                   scale_numeric=family=='logistic_regression')
    return Pipeline([('representation',features),('model',model)])


def cancellation_probability(estimator,X):
    """Resolve probability for class 1 explicitly, including single-class dummy."""
    classes=np.asarray(estimator.classes_)
    if not np.isin(classes,[0,1]).all(): raise ValueError('Expected cancellation labels 0 and/or 1.')
    probabilities=estimator.predict_proba(X)
    index=np.flatnonzero(classes==1)
    if not len(index): return np.zeros(len(X),dtype=float)
    return probabilities[:,index[0]]


def classification_metrics(y,probability,threshold=.5):
    """Evaluate cancellation=1; F1 uses the frozen >=0.5 threshold policy."""
    target=np.asarray(y); probability=np.asarray(probability,dtype=float)
    if target.ndim!=1 or probability.ndim!=1 or len(target)!=len(probability) or not len(target):
        raise ValueError('Expected aligned nonempty vectors.')
    if not np.isin(target,[0,1]).all() or not np.isfinite(probability).all():
        raise ValueError('Binary labels and finite probabilities are required.')
    if (probability<0).any() or (probability>1).any() or not 0<=threshold<=1:
        raise ValueError('Probability and threshold must be within [0,1].')
    predicted=(probability>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(target,predicted,labels=[0,1]).ravel()
    return {'f1':float(f1_score(target,predicted,zero_division=0)),
        'accuracy':float(accuracy_score(target,predicted)),
        'precision':float(precision_score(target,predicted,zero_division=0)),
        'recall':float(recall_score(target,predicted,zero_division=0)),
        'roc_auc':float(roc_auc_score(target,probability)) if len(np.unique(target))==2 else None,
        'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}
