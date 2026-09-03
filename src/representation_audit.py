"""representation comparison fixed representation comparisons; final test remains untouched."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from threadpoolctl import threadpool_limits

from .development_eda import read_selected_rows
from .feature_audit import load_development
from .feature_engineering import make_feature_preprocessor
from .representation import EncodedRepresentation, MODES
from .splitting import development_cv

PROTOCOL = {
    'modes': list(MODES), 'selection_percentile':75, 'constant_variance_threshold':1e-12,
    'pca_numeric_variance_target':.95, 'pca_svd_solver':'full',
    'reference_model':{'family':'LogisticRegression','C':1.0,'solver':'lbfgs','max_iter':2000,
        'tol':1e-4,'class_weight':None,'random_state':42},
    'threshold':.5, 'primary_metric':'unweighted mean cancellation-class F1 across the three frozen folds',
    'decision_rule':'highest mean F1; exact ties prefer lower mean output width, then declared mode order',
    'scope':'representation selection with one fixed reference classifier; not final model-family evaluation or tuning',
}


def plot_evidence(root, scores, schemas):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(12,5.4))
    palette=['#2672a5','#de8732','#378c6b']
    for ax, mode in zip(axes,['pca','selected_pca']):
        for fold in range(1,4):
            record=schemas[f'fold_{fold}'][mode]['pca']
            values=np.cumsum(record['explained_variance_ratio'])
            ax.plot(np.arange(1,len(values)+1),values,marker='o',markersize=3,
                    color=palette[fold-1],label=f'Fold {fold}: {len(values)} components')
        ax.axhline(.95,color='#555555',linestyle='--',linewidth=1,label='95% training variance')
        ax.set_ylim(0,1.025); ax.set_xlabel('Retained numeric components')
        ax.set_ylabel('Cumulative explained variance')
        ax.set_title('Numeric PCA' if mode=='pca' else 'Selection then numeric PCA',loc='left')
        ax.spines[['top','right']].set_visible(False)
        ax.legend(loc='lower right',fontsize=9,frameon=False)
    fig.text(.07,.025,'Training folds only • Centered PCA on standardized numeric fields; categories stay sparse.\n'
             '95% refers to the input numeric block, not all encoded features or predictive information.',fontsize=9)
    fig.subplots_adjust(left=.07,right=.98,top=.88,bottom=.2,wspace=.24)
    path=root/'figures/06_numeric_pca_variance.png'; path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=150,facecolor='white'); plt.close(fig)
    return path


def run_representation_audit(root):
    root=Path(root); output=root/'data/processed/representations'; output.mkdir(parents=True,exist_ok=True)
    # Persist choices before evaluating any candidate; do not revise from scores.
    protocol_path=output/'comparison_protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != PROTOCOL:
        raise ValueError('Comparison protocol changed; version and justify explicitly.')
    protocol_path.write_text(json.dumps(PROTOCOL,indent=2)+'\n')
    X, assignments, hashes=load_development(root)
    plan=json.loads((root/'data/processed/splits/validation_split_plan.json').read_text())
    target_path=root/'data/processed/eligibility_target.csv.gz'
    digest=hashlib.sha256(target_path.read_bytes()).hexdigest()
    if digest!=plan['upstream_output_sha256'][target_path.name]: raise ValueError('Target checksum changed.')
    hashes[target_path.name]=digest
    y=read_selected_rows(target_path,assignments.partition.eq('development')).is_canceled
    if len(y)!=len(X) or set(y.unique())!={0,1}: raise ValueError('Development labels do not align.')
    rows=[]; schemas={}; ranking_rows=[]
    with threadpool_limits(limits=1):
        for fold,(train_idx,val_idx) in enumerate(development_cv(assignments),1):
            train,validation=X.iloc[train_idx],X.iloc[val_idx]
            y_train,y_val=y.iloc[train_idx],y.iloc[val_idx]
            pre=make_feature_preprocessor()
            train_encoded=pre.fit_transform(train,y_train)
            pre_state=joblib.hash(pre)
            val_encoded=pre.transform(validation)
            if pre_state!=joblib.hash(pre): raise AssertionError('Validation refitted preprocessing.')
            names=tuple(pre.get_feature_names_out()); schemas[f'fold_{fold}']={}
            for mode in MODES:
                start=time.perf_counter()
                rep=EncodedRepresentation(names,mode=mode,percentile=75,variance_target=.95).fit(train_encoded,y_train)
                transformed_train=rep.transform(train_encoded)
                state=joblib.hash(rep)
                transformed_val=rep.transform(val_encoded)
                if state!=joblib.hash(rep): raise AssertionError('Validation changed selector/PCA state.')
                if transformed_train.shape[1]!=transformed_val.shape[1]: raise AssertionError('Fold widths mismatch.')
                if not np.isfinite(transformed_train.data).all() or not np.isfinite(transformed_val.data).all():
                    raise AssertionError('Nonfinite model inputs.')
                kwargs={k:v for k,v in PROTOCOL['reference_model'].items() if k!='family'}
                model=LogisticRegression(**kwargs)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always',ConvergenceWarning)
                    model.fit(transformed_train,y_train)
                convergence=[str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]
                if convergence: raise RuntimeError(f'{mode}, fold {fold}: '+convergence[0])
                probabilities=model.predict_proba(transformed_val)[:,1]
                predicted=(probabilities>=.5).astype(int)
                numeric_count=len(rep.numeric_positions_)
                component_count=int(rep.pca_.n_components_) if rep.pca_ is not None else 0
                explained=float(rep.pca_.explained_variance_ratio_.sum()) if rep.pca_ is not None else None
                if explained is not None and explained<.95: raise AssertionError('PCA retained too little training variance.')
                rows.append({'fold':fold,'mode':mode,'train_rows':len(train),'validation_rows':len(validation),
                    'encoded_inputs':len(names),'selected_inputs':len(rep.selected_indices_),
                    'numeric_inputs_to_pca':numeric_count if component_count else 0,
                    'pca_components':component_count,'pca_retained_numeric_variance':explained,
                    'output_columns':transformed_train.shape[1],
                    'f1':f1_score(y_val,predicted,zero_division=0),'accuracy':accuracy_score(y_val,predicted),
                    'precision':precision_score(y_val,predicted,zero_division=0),'recall':recall_score(y_val,predicted,zero_division=0),
                    'roc_auc':roc_auc_score(y_val,probabilities),'iterations':int(model.n_iter_[0]),
                    'fit_seconds':time.perf_counter()-start,'training_only_state_verified':True,
                    'dense_numeric_training_bytes':rep.dense_numeric_training_bytes_})
                pca_record=None
                if rep.pca_ is not None:
                    pca_record={'input_names':rep.selected_names_[rep.numeric_positions_].tolist(),
                        'components':rep.pca_.components_.tolist(), 'means':rep.pca_.mean_.tolist(),
                        'explained_variance_ratio':rep.pca_.explained_variance_ratio_.tolist(),
                        'explained_variance':rep.pca_.explained_variance_.tolist()}
                schemas[f'fold_{fold}'][mode]={'selected_names':rep.selected_names_.tolist(),
                    'output_names':rep.get_feature_names_out().tolist(),'pca':pca_record}
                if mode=='selected':
                    support=set(rep.selected_indices_)
                    order=np.argsort(-np.nan_to_num(rep.scores_,nan=-1),kind='stable')
                    for rank,index in enumerate(order,1):
                        ranking_rows.append({'fold':fold,'rank':rank,'feature':names[index],
                            'training_variance':float(rep.variances_[index]),
                            'f_score':None if np.isnan(rep.scores_[index]) else float(rep.scores_[index]),
                            'selected':index in support})
                print(f'representation comparison fold {fold} {mode}: F1={rows[-1]["f1"]:.6f}, width={rows[-1]["output_columns"]}',flush=True)
    scores=pd.DataFrame(rows)
    comparison=scores.groupby('mode',sort=False).agg(mean_f1=('f1','mean'),fold_sd_f1=('f1','std'),
        mean_accuracy=('accuracy','mean'),mean_precision=('precision','mean'),mean_recall=('recall','mean'),
        mean_roc_auc=('roc_auc','mean'),mean_output_columns=('output_columns','mean')).reset_index()
    ranked=comparison.assign(mode_order=comparison['mode'].map({m:i for i,m in enumerate(MODES)})).sort_values(
        ['mean_f1','mean_output_columns','mode_order'],ascending=[False,True,True],kind='stable')
    preferred=str(ranked.iloc[0]['mode'])
    scores.to_csv(output/'fold_results.csv',index=False)
    comparison.to_csv(output/'representation_comparison.csv',index=False)
    pd.DataFrame(ranking_rows).to_csv(output/'feature_rankings.csv',index=False)
    (output/'representation_schemas.json').write_text(json.dumps(schemas,indent=2,allow_nan=False)+'\n')
    figure=plot_evidence(root,scores,schemas)
    summary={'analysis': 'representations','status':'completed','development_rows':len(X),'test_rows_fitted_transformed_or_scored':0,'test_target_distribution_computed':False,
        'training_label_use':'F-score selection and fixed reference classifier fitting',
        'validation_label_use':'representation comparison only; no selector/reducer fitting',
        'reference_model_fits':12,'model_family_comparison_completed':False,'hyperparameter_tuning_completed':False,
        'rows_removed':0,'input_sha256':hashes,'protocol':PROTOCOL,'preferred_mode':preferred,
        'decision_scope':'reference-logistic-regression development choice; not final best-model claim',
        'mean_f1':float(ranked.iloc[0]['mean_f1']),
        'all_feature_reference_mean_f1':float(comparison.set_index('mode').loc['full','mean_f1']),
        'final_feature_rule':'Refit chosen representation on each model-training fold; retain its learned names/components. Refit on all development only after final choices are frozen.',
        'max_dense_numeric_training_bytes':int(scores.dense_numeric_training_bytes.max()),
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__},
        'limitations':['F statistics are ranking heuristics, not p-values, causal effects, or independent evidence for repeated bookings.',
            'Supervised univariate selection can miss nonlinear interactions and rare-category signal.',
            'PCA preserves input numeric variance, not predictive information; category/indicator features are not reduced by PCA.',
            'Selection and reduction are fitted within each training fold; retained feature identities can vary by fold.',
            'The same development folds choose the representation, so reported CV scores are selection estimates, not unbiased final performance.',
            'Representation rankings may differ for random forest; model comparison must compare model families and retain a full-feature control.',
            'No final full-development selector/PCA/model is fitted at this stage; final refit and held-out evaluation remain evaluation.']}
    artifacts=[output/name for name in ['comparison_protocol.json','fold_results.csv','representation_comparison.csv',
               'feature_rankings.csv','representation_schemas.json']]+[figure]
    summary['output_sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}
    (output/'representation_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return summary,comparison,scores


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args(); summary,comparison,_=run_representation_audit(args.root)
    print(comparison.to_string(index=False)); print('Preferred mode:',summary['preferred_mode'])


if __name__=='__main__': main()
