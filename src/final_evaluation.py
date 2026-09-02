"""Step 13: frozen-pipeline refit, one official held-out evaluation and errors."""
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
from sklearn.metrics import brier_score_loss
from threadpoolctl import threadpool_limits

from .development_eda import read_selected_rows
from .feature_audit import load_development
from .modeling import cancellation_probability, classification_metrics
from .splitting import check_assignments
from .tuning import build_frozen_pipeline

ERROR_FEATURES = ['hotel', 'lead_time', 'deposit_type', 'market_segment',
                  'customer_type', 'previous_cancellations',
                  'total_of_special_requests']
METRICS = ['f1', 'accuracy', 'precision', 'recall', 'roc_auc']
PROTOCOL = {
    'step': 13,
    'official_evaluation': 'fit frozen Step 12 pipeline on development; score final test once',
    'selection_source': 'data/results/step12/final_selection.json',
    'family': 'logistic_regression', 'representation': 'selected',
    'model_parameters': {'C': 1.0, 'class_weight': 'balanced'},
    'threshold': .5, 'threshold_rule': 'class 1 when probability >= 0.5',
    'metrics': METRICS, 'additional_probability_diagnostic': 'Brier score',
    'subgroups': ['hotel', 'lead_time_band', 'deposit_type', 'market_segment', 'customer_type'],
    'lead_time_bands': {'edges': [-1, 7, 30, 90, 180, 365, None],
                        'labels': ['0–7', '8–30', '31–90', '91–180', '181–365', '366+']},
    'probability_bins': [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0],
    'error_examples': 'per FP/FN: five most confident plus five closest to threshold among remaining errors',
    'subgroup_policy': 'descriptive post-evaluation diagnostics only; no model, feature or threshold changes',
    'test_policy': 'report results plainly; never reselect or retune from held-out scores',
    'saved_model': 'models/final_logistic_regression.joblib',
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_protocol(path):
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != PROTOCOL:
        raise ValueError('Frozen Step 13 protocol changed; do not overwrite after test access.')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(PROTOCOL, indent=2) + '\n')


def checked_selection(root):
    root = Path(root)
    summary = json.loads((root/'data/results/step12/tuning_summary.json').read_text())
    for relative, expected in summary['output_sha256'].items():
        if sha256(root/relative) != expected:
            raise ValueError('Step 12 output changed: ' + relative)
    selection_path = root/'data/results/step12/final_selection.json'
    selection = json.loads(selection_path.read_text())
    if selection['family'] != PROTOCOL['family'] or selection['representation'] != PROTOCOL['representation']:
        raise ValueError('Step 13 protocol does not match frozen selection.')
    if selection['search_parameters'] != {'model__C': 1.0, 'model__class_weight': 'balanced'}:
        raise ValueError('Frozen Step 12 model settings changed.')
    if selection['threshold'] != PROTOCOL['threshold']:
        raise ValueError('Frozen threshold changed.')
    return selection, summary, sha256(selection_path)


def add_diagnostic_columns(X, metadata, actual, probability, threshold=.5):
    if len(X) != len(metadata) or len(X) != len(actual) or len(X) != len(probability):
        raise ValueError('Final-test diagnostic inputs are misaligned.')
    result = pd.DataFrame({
        'cohort_row': metadata['cohort_row'].to_numpy(),
        'source_row_id': metadata['source_row_id'].to_numpy(),
        'arrival_date': metadata['arrival_date'].astype(str).to_numpy(),
        'actual': np.asarray(actual, dtype=int),
        'cancellation_probability': np.asarray(probability, dtype=float),
    })
    result['predicted'] = (result.cancellation_probability >= threshold).astype(int)
    result['error_type'] = np.select(
        [(result.actual.eq(0)&result.predicted.eq(1)),
         (result.actual.eq(1)&result.predicted.eq(0))], ['FP','FN'], default='correct')
    for column in ERROR_FEATURES:
        result[column] = X[column].reset_index(drop=True)
    result['lead_time_band'] = pd.cut(result.lead_time, bins=[-1,7,30,90,180,365,np.inf],
                                      labels=PROTOCOL['lead_time_bands']['labels']).astype('object')
    if result.lead_time_band.isna().any():
        raise ValueError('Lead-time bands do not cover final-test rows.')
    return result


def grouped_metrics(diagnostics):
    rows=[]
    for dimension in PROTOCOL['subgroups']:
        values=diagnostics[dimension].astype('object').where(diagnostics[dimension].notna(), '__MISSING__')
        for value in pd.unique(values):
            group=diagnostics.loc[values.eq(value)]
            metrics=classification_metrics(group.actual, group.cancellation_probability, PROTOCOL['threshold'])
            rows.append({'dimension':dimension,'group':str(value),'rows':len(group),
                         'cancellations':int(group.actual.sum()),
                         'actual_cancellation_rate':float(group.actual.mean()),
                         'predicted_cancellation_rate':float(group.predicted.mean()),
                         'error_rate':float(group.actual.ne(group.predicted).mean()), **metrics})
        if sum(r['rows'] for r in rows if r['dimension']==dimension) != len(diagnostics):
            raise AssertionError('Subgroup rows do not reconcile.')
    return pd.DataFrame(rows)


def probability_diagnostics(diagnostics):
    bins=PROTOCOL['probability_bins']
    labels=[f'{bins[i]:.1f}–{bins[i+1]:.1f}' for i in range(len(bins)-1)]
    bucket=pd.cut(diagnostics.cancellation_probability, bins=bins, labels=labels,
                  include_lowest=True, right=True)
    rows=[]
    for label in labels:
        group=diagnostics.loc[bucket.eq(label)]
        rows.append({'probability_bin':label,'rows':len(group),
                     'mean_predicted_probability':float(group.cancellation_probability.mean()) if len(group) else None,
                     'observed_cancellation_rate':float(group.actual.mean()) if len(group) else None,
                     'errors':int(group.actual.ne(group.predicted).sum())})
    if sum(r['rows'] for r in rows) != len(diagnostics):
        raise AssertionError('Probability bins do not cover all rows.')
    return pd.DataFrame(rows)


def select_error_examples(diagnostics, per_slice=5):
    chosen=[]
    for error in ['FP','FN']:
        group=diagnostics.loc[diagnostics.error_type.eq(error)].copy()
        if len(group) < 2*per_slice:
            raise ValueError(f'Not enough {error} rows for the frozen example policy.')
        confident=group.sort_values(['cancellation_probability','source_row_id'],
                                    ascending=[error=='FN',True],kind='stable').head(per_slice)
        remaining=group.drop(index=confident.index).assign(
            threshold_distance=lambda d:(d.cancellation_probability-PROTOCOL['threshold']).abs())
        boundary=remaining.sort_values(['threshold_distance','source_row_id'],kind='stable').head(per_slice)
        confident=confident.assign(example_reason='most_confident_'+error)
        boundary=boundary.assign(example_reason='closest_to_threshold_'+error)
        chosen.extend([confident,boundary])
    result=pd.concat(chosen,ignore_index=True)
    if result.source_row_id.duplicated().any() or len(result)!=4*per_slice:
        raise AssertionError('Error examples must be distinct and complete.')
    return result[['example_reason','source_row_id','arrival_date','hotel','lead_time','lead_time_band',
                   'deposit_type','market_segment','customer_type','previous_cancellations',
                   'total_of_special_requests','actual','predicted','cancellation_probability','error_type']]


def coefficient_table(pipeline):
    names=pipeline.named_steps['representation'].get_feature_names_out()
    model=pipeline.named_steps['model']
    if model.coef_.shape != (1,len(names)):
        raise ValueError('Binary logistic coefficient shape does not match features.')
    table=pd.DataFrame({'feature':names,'coefficient':model.coef_[0]})
    table['absolute_coefficient']=table.coefficient.abs()
    table['direction']=np.where(table.coefficient>=0,'higher cancellation score','lower cancellation score')
    return table.sort_values(['absolute_coefficient','feature'],ascending=[False,True],kind='stable').reset_index(drop=True)


def plot_results(root, metrics, diagnostics, groups, probability):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root)
    # Final metrics and confusion matrix.
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.8))
    labels=['F1','Accuracy','Precision','Recall','ROC-AUC']
    values=[metrics[k] for k in ['f1','accuracy','precision','recall','roc_auc']]
    bars=axes[0].bar(labels,values,color=['#267f9c','#8096a3','#4d8c75','#b06d4f','#756aa8'])
    axes[0].bar_label(bars,fmt='%.3f',padding=3);axes[0].set_ylim(0,1)
    axes[0].set_ylabel('Held-out test metric');axes[0].set_title('Final model performance',loc='left')
    matrix=np.array([[metrics['tn'],metrics['fp']],[metrics['fn'],metrics['tp']]])
    axes[1].imshow(matrix,cmap='Blues')
    for i in range(2):
        for j in range(2): axes[1].text(j,i,f'{matrix[i,j]:,}',ha='center',va='center',fontsize=13)
    axes[1].set_xticks([0,1],['Predicted 0','Predicted 1']);axes[1].set_yticks([0,1],['Actual 0','Actual 1'])
    axes[1].set_title('Confusion matrix',loc='left')
    for ax in axes: ax.spines[['top','right']].set_visible(False)
    fig.text(.08,.02,'Frozen Step 12 pipeline • Threshold 0.5 • 23,795 later-arrival bookings',fontsize=9)
    fig.subplots_adjust(left=.08,right=.98,top=.88,bottom=.18,wspace=.32)
    final_path=root/'figures/09_final_test_performance.png';fig.savefig(final_path,dpi=150,facecolor='white');plt.close(fig)
    # Probability reliability view and error counts by hotel.
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.8))
    shown=probability.loc[probability.rows.gt(0)]
    axes[0].plot(shown.mean_predicted_probability,shown.observed_cancellation_rate,marker='o',color='#267f9c')
    axes[0].plot([0,1],[0,1],linestyle='--',color='#929292',label='ideal reference')
    axes[0].set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Observed cancellation rate')
    axes[0].set_title('Fixed-bin probability diagnostic',loc='left');axes[0].legend(frameon=False)
    hotels=groups.loc[groups.dimension.eq('hotel')]
    x=np.arange(len(hotels));axes[1].bar(x-.18,hotels.fp,.36,label='False positives',color='#b06d4f')
    axes[1].bar(x+.18,hotels.fn,.36,label='False negatives',color='#756aa8')
    axes[1].set_xticks(x,hotels.group);axes[1].set_ylabel('Error count')
    axes[1].set_title('Errors by hotel',loc='left');axes[1].legend(frameon=False)
    for ax in axes: ax.spines[['top','right']].set_visible(False)
    fig.text(.08,.02,'Post-evaluation diagnostics only; no setting or threshold changes',fontsize=9)
    fig.subplots_adjust(left=.08,right=.98,top=.88,bottom=.18,wspace=.32)
    error_path=root/'figures/10_final_error_analysis.png';fig.savefig(error_path,dpi=150,facecolor='white');plt.close(fig)
    return [final_path,error_path]


def run_final_evaluation(root):
    root=Path(root);out=root/'data/results/step13';write_protocol(out/'evaluation_protocol.json')
    selection,step12,selection_hash=checked_selection(root)
    X_dev,assignments,input_hashes=load_development(root)
    check_assignments(assignments)
    dev_mask=assignments.partition.eq('development').to_numpy();test_mask=assignments.partition.eq('test').to_numpy()
    if (dev_mask.sum(),test_mask.sum())!=(95415,23795): raise ValueError('Frozen partition sizes changed.')
    target_path=root/'data/processed/step5_target.csv.gz'
    if sha256(target_path)!=selection['input_sha256'][target_path.name]: raise ValueError('Frozen target changed.')
    y_dev=read_selected_rows(target_path,dev_mask).is_canceled.reset_index(drop=True)
    pipeline=build_frozen_pipeline(selection)
    if hasattr(pipeline.named_steps['representation'],'preprocessor_'): raise AssertionError('Pipeline was not fresh.')
    start=time.perf_counter()
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning);pipeline.fit(X_dev,y_dev)
    fit_seconds=time.perf_counter()-start
    if pipeline.named_steps['representation'].preprocessor_.named_steps['columns'] is None:
        raise AssertionError('Representation did not fit.')
    # Test features can now be transformed; choices are fixed and fit is complete.
    X_test=read_selected_rows(root/'data/processed/step5_candidates.csv.gz',test_mask).reset_index(drop=True)
    test_assignments=assignments.loc[test_mask,['cohort_row','source_row_id','arrival_date']].reset_index(drop=True)
    state=joblib.hash(pipeline.named_steps['representation'])
    probability=cancellation_probability(pipeline,X_test)
    if state!=joblib.hash(pipeline.named_steps['representation']): raise AssertionError('Test prediction changed fitted features.')
    # First and only official access to held-out target values occurs here.
    y_test=read_selected_rows(target_path,test_mask).is_canceled.reset_index(drop=True)
    metrics=classification_metrics(y_test,probability,selection['threshold'])
    metrics['brier_score']=float(brier_score_loss(y_test,probability))
    metrics.update({'test_rows':len(y_test),'test_cancellations':int(y_test.sum()),
                    'test_cancellation_rate':float(y_test.mean()),'threshold':selection['threshold']})
    diagnostics=add_diagnostic_columns(X_test,test_assignments,y_test,probability,selection['threshold'])
    groups=grouped_metrics(diagnostics);prob_diag=probability_diagnostics(diagnostics)
    examples=select_error_examples(diagnostics);coefficients=coefficient_table(pipeline)
    prediction_export=diagnostics[['cohort_row','source_row_id','arrival_date','actual','predicted',
                                   'cancellation_probability','error_type']]
    prediction_paths=[]
    for part,indices in enumerate(np.array_split(np.arange(len(prediction_export)),4),1):
        path=out/f'test_predictions_{part:02d}.csv.gz'
        prediction_export.iloc[indices].to_csv(path,index=False,compression='gzip')
        prediction_paths.append(path)
    pd.DataFrame([metrics]).to_csv(out/'final_metrics.csv',index=False)
    groups.to_csv(out/'subgroup_metrics.csv',index=False)
    prob_diag.to_csv(out/'probability_diagnostics.csv',index=False)
    examples.to_csv(out/'error_examples.csv',index=False)
    coefficients.to_csv(out/'feature_coefficients.csv',index=False)
    model_path=root/PROTOCOL['saved_model'];model_path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(pipeline,model_path,compress=3)
    figures=plot_results(root,metrics,diagnostics,groups,prob_diag)
    summary={'step':13,'status':'completed','responsible_member':'Sadat','next_step':14,'next_owner':'Sadat',
             'official_test_evaluations':1,'selection_sha256':selection_hash,
             'selection':{'family':selection['family'],'representation':selection['representation'],
                          'search_parameters':selection['search_parameters'],'threshold':selection['threshold'],
                          'mean_development_f1':selection['mean_development_f1']},
             'development_rows_fitted':len(X_dev),'test_rows_evaluated':len(X_test),
             'metrics':metrics,'fit_seconds':fit_seconds,'encoded_features':len(coefficients),
             'model_iterations':int(pipeline.named_steps['model'].n_iter_[0]),
             'representation_unchanged_after_test_prediction':True,
             'subgroup_dimensions':PROTOCOL['subgroups'],'published_error_examples':len(examples),
             'model_reselected_from_test':False,'threshold_changed_after_test':False,
             'input_sha256':{**input_hashes,target_path.name:sha256(target_path)},
             'step12_output_hashes_verified':True,
             'runtime':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                        'scikit_learn':sklearn.__version__,'joblib':joblib.__version__},
             'limitations':['A single chronological holdout gives one period-specific estimate, not a confidence interval.',
                            'Development folds informed representation, model and parameters; only the final test was untouched until this step.',
                            'Subgroup metrics and selected errors are descriptive after test access and do not authorize model changes.',
                            'Coefficient magnitude is not causal importance; encoded/scaled features differ in interpretation.',
                            'Retrospective feature timing, repeated profiles, partial seasonal coverage and source-provenance limits remain.']}
    artifacts=[out/name for name in ['evaluation_protocol.json','final_metrics.csv','subgroup_metrics.csv',
               'probability_diagnostics.csv','error_examples.csv','feature_coefficients.csv']]+prediction_paths+[model_path]+figures
    summary['output_sha256']={str(p.relative_to(root)):sha256(p) for p in artifacts}
    (out/'evaluation_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return summary,groups,prob_diag,examples,coefficients


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    summary,groups,_,examples,coefficients=run_final_evaluation(parser.parse_args().root)
    print(json.dumps(summary['metrics'],indent=2));print('\nSubgroup metrics:\n',groups.to_string(index=False))
    print('\nError examples:\n',examples.to_string(index=False));print('\nTop coefficients:\n',coefficients.head(20).to_string(index=False))


if __name__=='__main__':main()
