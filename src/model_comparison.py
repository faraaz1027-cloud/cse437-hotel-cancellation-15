"""Step 11 baseline and two-family comparison on frozen development folds."""
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
from threadpoolctl import threadpool_limits
from .development_eda import read_selected_rows
from .feature_audit import load_development
from .modeling import CANDIDATES,MODEL_SETTINGS,make_model_pipeline,cancellation_probability,classification_metrics
from .splitting import development_cv

PROTOCOL={
    'candidates':[{'candidate':name,'family':family,'representation':mode} for name,family,mode in CANDIDATES],
    'model_settings':MODEL_SETTINGS,'majority_settings':{'strategy':'most_frequent','random_state':42},
    'selection':{'percentile':75,'training_variance_threshold':1e-12,'score':'ANOVA F'},
    'logistic_numeric_scaling':True,'forest_numeric_scaling':False,
    'threshold':.5,'threshold_rule':'class 1 when probability >= 0.5',
    'primary_metric':'unweighted mean cancellation-class F1 over three frozen forward folds',
    'secondary_metrics':['accuracy','precision','recall','roc_auc'],
    'decision_rule':'highest mean validation F1; exact ties prefer lower mean width then declared candidate order',
    'class_weighting':None,'resampling':None,'parameter_search':False,
    'scope':'untuned development model-family comparison; no final test evaluation',
}


def plot_comparison(root,fold_results,comparison):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'majority':'Majority baseline','lr_full':'Logistic · full','lr_selected':'Logistic · selected',
            'rf_full':'Forest · full','rf_selected':'Forest · selected'}
    order=[c[0] for c in CANDIDATES]
    table=comparison.set_index('candidate').loc[order]
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.6))
    ypos=np.arange(len(order))
    colors=['#969ba2','#3581ad','#3581ad','#358a64','#358a64']
    axes[0].barh(ypos,table.mean_f1,color=colors,height=.60,alpha=.75)
    for i,row in enumerate(table.itertuples()):
        axes[0].text(row.mean_f1+.012,i,f'{row.mean_f1:.3f}',va='center',fontsize=10)
    axes[0].set_yticks(ypos,[labels[k] for k in order]); axes[0].invert_yaxis()
    axes[0].set_xlim(0,1); axes[0].set_xlabel('Mean validation F1 (cancellation = 1)')
    axes[0].set_title('Untuned model comparison',loc='left',fontsize=13)
    markers=['o','s','^']
    for i,key in enumerate(order[1:]):
        group=fold_results.query('candidate == @key').sort_values('fold')
        axes[1].plot(group.fold,group.f1,marker=markers[i%3],
            linestyle='--' if key.endswith('full') else '-',label=labels[key],
            color='#3581ad' if key.startswith('lr') else '#358a64')
    axes[1].set_xticks([1,2,3]); axes[1].set_xlabel('Forward validation fold')
    axes[1].set_ylabel('Cancellation-class F1'); axes[1].set_ylim(0,1)
    axes[1].set_title('Performance varies across periods',loc='left',fontsize=13)
    axes[1].legend(loc='lower right',fontsize=9,frameon=False)
    for ax in axes: ax.spines[['top','right']].set_visible(False)
    fig.text(.18,.025,'Development only • Same three frozen folds; probability threshold 0.5.\n'
             'These results guide model tuning; final held-out performance remains unknown.',fontsize=9)
    fig.subplots_adjust(left=.18,right=.98,top=.87,bottom=.20,wspace=.35)
    path=root/'figures/07_model_comparison.png';path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=150,facecolor='white');plt.close(fig)
    return path


def run_model_comparison(root):
    root=Path(root);output=root/'data/results/step11';output.mkdir(parents=True,exist_ok=True)
    protocol_path=output/'comparison_protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=PROTOCOL:
        raise ValueError('Step 11 protocol changed; version and justify explicitly.')
    protocol_path.write_text(json.dumps(PROTOCOL,indent=2)+'\n')
    X,assignments,hashes=load_development(root)
    plan=json.loads((root/'data/splits/step6_split_plan.json').read_text())
    target_path=root/'data/processed/step5_target.csv.gz'
    digest=hashlib.sha256(target_path.read_bytes()).hexdigest()
    if digest!=plan['upstream_output_sha256'][target_path.name]:raise ValueError('Frozen target changed.')
    hashes[target_path.name]=digest
    y=read_selected_rows(target_path,assignments.partition.eq('development')).is_canceled
    if len(y)!=len(X) or set(y.unique())!={0,1}:raise ValueError('Development target mismatch.')
    dev_assignments=assignments.loc[assignments.partition.eq('development')].reset_index(drop=True)
    records=[];schemas={};configured={}
    with threadpool_limits(limits=1):
        for fold,(train_idx,val_idx) in enumerate(development_cv(assignments),1):
            train,val=X.iloc[train_idx],X.iloc[val_idx]
            y_train,y_val=y.iloc[train_idx],y.iloc[val_idx]
            schemas[f'fold_{fold}']={}
            membership_hash=hashlib.sha256(dev_assignments.iloc[val_idx].source_row_id.to_csv(index=False).encode()).hexdigest()
            for name,family,mode in CANDIDATES:
                print(f'Step 11: fitting {name}, fold {fold} ({len(train):,} training rows)',flush=True)
                pipe=make_model_pipeline(family,mode)
                configured[name]=pipe.named_steps['model'].get_params(deep=False)
                start=time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always',ConvergenceWarning)
                    pipe.fit(train,y_train)
                fit_seconds=time.perf_counter()-start
                convergence=[str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]
                if convergence:raise RuntimeError(f'{name}, fold {fold}: '+convergence[0])
                learned=pipe.named_steps.get('representation')
                state=joblib.hash(learned) if learned is not None else None
                validation_probability=cancellation_probability(pipe,val)
                validation_metrics=classification_metrics(y_val,validation_probability)
                training_metrics=classification_metrics(y_train,cancellation_probability(pipe,train))
                if learned is not None and state!=joblib.hash(learned):
                    raise AssertionError('Prediction changed training-fitted representation.')
                names=learned.get_feature_names_out().tolist() if learned is not None else []
                if len(names)!=len(set(names)):raise AssertionError('Duplicate output feature names.')
                schemas[f'fold_{fold}'][name]=names
                majority=int(y_train.mode().iloc[0])
                if family=='majority':
                    if not np.all(validation_probability==majority):raise AssertionError('Baseline must predict training majority.')
                if family=='random_forest':
                    model=pipe.named_steps['model']
                    capacity={'trees':len(model.estimators_),'mean_tree_depth':float(np.mean([t.tree_.max_depth for t in model.estimators_])),
                              'mean_tree_leaves':float(np.mean([t.tree_.n_leaves for t in model.estimators_]))}
                else:capacity={'trees':0,'mean_tree_depth':None,'mean_tree_leaves':None}
                row={'candidate':name,'family':family,'representation':mode,'fold':fold,
                    'train_rows':len(train),'validation_rows':len(val),
                    'training_canceled':int(y_train.sum()),'validation_canceled':int(y_val.sum()),
                    'training_majority_class':majority,'encoded_columns':len(names),
                    **validation_metrics,'training_f1':training_metrics['f1'],
                    'training_accuracy':training_metrics['accuracy'],'training_roc_auc':training_metrics['roc_auc'],
                    'f1_train_minus_validation':training_metrics['f1']-validation_metrics['f1'],
                    'fit_seconds':fit_seconds,'convergence_warning':False,
                    'iterations':int(pipe.named_steps['model'].n_iter_[0]) if family=='logistic_regression' else None,
                    'representation_unchanged_after_prediction':True,'validation_membership_sha256':membership_hash,**capacity}
                records.append(row)
                print(f'Completed {name}, fold {fold}: validation F1={row["f1"]:.6f}; training F1={row["training_f1"]:.6f}',flush=True)
    results=pd.DataFrame(records)
    comparison=results.groupby(['candidate','family','representation'],sort=False).agg(
        mean_f1=('f1','mean'),fold_sd_f1=('f1','std'),mean_accuracy=('accuracy','mean'),
        mean_precision=('precision','mean'),mean_recall=('recall','mean'),mean_roc_auc=('roc_auc','mean'),
        mean_training_f1=('training_f1','mean'),mean_f1_gap=('f1_train_minus_validation','mean'),
        mean_fit_seconds=('fit_seconds','mean'),mean_encoded_columns=('encoded_columns','mean')).reset_index()
    # Check integration parity against the already-published Step 10 reference.
    old_path=root/'data/processed/step10/fold_results.csv'
    old_summary=json.loads((root/'data/processed/step10/representation_summary.json').read_text())
    old_digest=hashlib.sha256(old_path.read_bytes()).hexdigest()
    # Earlier run-local summaries may use Windows separators. Normalize keys only;
    # keep the stored digest and strict numerical comparisons unchanged.
    old_hashes={key.replace('\\', '/'): value
                for key,value in old_summary['output_sha256'].items()}
    if old_digest!=old_hashes['data/processed/step10/fold_results.csv']:
        raise ValueError('Step 10 reference evidence changed.')
    old=pd.read_csv(old_path)
    for mode in ['full','selected']:
        previous=old.loc[old['mode'].eq(mode)].sort_values('fold')
        current=results.loc[results.candidate.eq('lr_'+mode)].sort_values('fold')
        np.testing.assert_allclose(previous[['f1','accuracy','precision','recall','roc_auc']],
            current[['f1','accuracy','precision','recall','roc_auc']],rtol=0,atol=1e-12)
    order={name:i for i,(name,_,_) in enumerate(CANDIDATES)}
    ranked=comparison.assign(order=comparison.candidate.map(order)).sort_values(
        ['mean_f1','mean_encoded_columns','order'],ascending=[False,True,True],kind='stable')
    preferred=ranked.iloc[0]
    results.to_csv(output/'fold_results.csv',index=False)
    comparison.to_csv(output/'model_comparison.csv',index=False)
    (output/'feature_schemas.json').write_text(json.dumps(schemas,indent=2)+'\n')
    (output/'estimator_parameters.json').write_text(json.dumps(configured,indent=2)+'\n')
    figure=plot_comparison(root,results,comparison)
    summary={'step':11,'status':'completed','responsible_member':'Sadat','next_step':12,'next_owner':'Sadat',
        'development_rows':len(X),'candidate_count':5,'fold_count':3,'model_fits':15,
        'learned_model_families':['logistic_regression','random_forest'],'baseline':'training-majority class',
        'preferred_candidate':str(preferred.candidate),'preferred_family':str(preferred.family),
        'preferred_representation':str(preferred.representation),'mean_f1':float(preferred.mean_f1),
        'decision_scope':'untuned development candidate for further tuning; not final best-model or test claim',
        'protocol':PROTOCOL,'input_sha256':hashes,'step10_reference_sha256':old_digest,
        'logistic_results_match_step10':True,'convergence_warnings':0,
        'test_rows_fitted_transformed_or_scored':0,'test_target_distribution_computed':False,
        'full_development_model_fitted':False,'model_hyperparameter_tuning_completed':False,
        'rows_removed':0,'row_level_predictions_published':False,
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__},
        'limitations':['Development scores are reused for representation/model selection; they are not unbiased final performance estimates.',
            'Training scores are resubstitution diagnostics; train-validation gaps mix capacity, temporal shift and recorded repetition.',
            'Class weighting and threshold optimization are not performed; comparisons use the fixed cancellation-class F1 policy.',
            'No nested validation or final held-out result is reported; model hyperparameter tuning remains Step 12.',
            'Source timing and repeated-record limitations from earlier steps still apply.']}
    artifacts=[output/name for name in ['comparison_protocol.json','fold_results.csv','model_comparison.csv',
        'feature_schemas.json','estimator_parameters.json']]+[figure]
    summary['output_sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}
    (output/'model_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return summary,comparison,results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    summary,comparison,_=run_model_comparison(parser.parse_args().root)
    print(comparison.to_string(index=False));print('Current preferred candidate:',summary['preferred_candidate'])


if __name__=='__main__':main()
