# Handoff from Faraaz's stages to Sadat

**CSE437 Group 15, Section 05, Summer 2026. Step 15 in progress. Owner: Sadat, with both members reviewing.**

The 10-page report/PDF and 168-word summary are assembled and visually checked.
58 tests, all five canonical notebook-format checks, and dependency consistency
pass. Fresh-kernel execution remains pending because the host blocks startup;
the user explicitly approved publishing with that limitation. Follow
[FINAL_CHECKS.md](FINAL_CHECKS.md) locally, publish the unchanged raw CSV with
attribution, and confirm actual contributions before declaring submission-ready.

The approved Step 15 test-comparison supplement is complete. Test F1:
baseline 0, frozen Logistic Regression 0.750592, development-selected Random
Forest 0.723115. No selected model, threshold or Step 13 artifact changed.
The comparison was authorized after the original test result and is explicitly
reporting-only. See `data/results/step15/README.md` and the current gates in
`PROJECT_STATUS.md`. The original download date/version are unknown, as the
user confirmed. Next: finish Step 15 submission checks; no Step 16.

This is an implementation handoff prepared with ChatGPT/Codex assistance.
Faraaz must review/explain the earlier stages and report actual contributions;
the document does not claim he personally wrote AI-prepared code.

## Start here

1. Read PROJECT_STATUS.md and the approved problem/questions in README.md.
   Do not change the original CSE437 document, dataset, target, or questions.
2. Use the committed Step 5 data and Step 6 assignment loader. Development has
   95,415 rows; test has 23,795. Do not reorder aligned files independently.
3. Read `src/preprocessing.py` and the Step 7 report. A new factory belongs
   inside the model pipeline so training-fold medians/scalers/vocabularies
   cannot see validation data. CV indices are relative to the development subset.
4. Read Notebook 01's development EDA, Notebook 03's executed Steps 9–10, and
   report Sections 1–8 and the Step 14 answers. Notebook 04 contains comparison/tuning; Notebook 05 contains final evaluation.

## Step 9: implemented derived-feature candidates

- Total nights = weekend plus weekday nights; retain legitimate zero nights.
- Total guests = adults + children + babies, preserving unknown child totals
  until the documented fold-fitted imputation/indicator rule is applied.
- Prior-history presence and cancellation ratio. Define the denominator as
  previous cancellations plus previous noncanceled bookings; distinguish no
  recorded history from a genuinely observed zero cancellation ratio.
- Company-code recording flag: 1 for any present code and 0 for null. This is
  a recording proxy, not evidence of corporate payment. Sparse IDs stay excluded.
- Cyclic month/calendar features using fixed calendar rules. The first CV
  training period does not include all validation months.

The eight implemented derived fields and missing/zero-denominator rules are
documented in [the Step 9 report](report/step9_feature_engineering.md). The new
`make_feature_preprocessor()` explicitly extends the schema to 32 fields and
replaces month names with cyclic coordinates. Its scaled/unscaled variants
pass all three development folds (332/421/490 encoded columns). Use this new
factory inside later model pipelines; the original Step 7 factory is preserved.
Step 10 now compares these representations with a fixed reference classifier;
the original feature definitions are preserved.

## Step 10 selection/reduction reference

Supervised selection retains 75% of nonconstant training encoded features by
F-score. Centered numeric PCA retains at least 95% of its input training variance.
Four modes were evaluated using a fixed logistic-regression reference. Selection
alone has the highest mean F1 (0.713609), versus full features (0.693094), PCA
(0.694633), and selection then PCA (0.701023). It retains 247/314/366 columns by
fold. PCA is demonstrated but not retained in the preferred representation.

Read [the Step 10 report](report/step10_selection_and_reduction.md) for the selection/PCA evidence. Step 11 compared full and selected representations for both learned model families, using the same frozen folds.

## Step 11 untuned comparison

The majority baseline, logistic regression and random forest are implemented
in Notebook 04. Across three frozen folds, mean cancellation F1 is 0 for the
baseline, 0.693094 for full LR, **0.713609 for selected LR**, 0.626504 for full
forest and 0.657994 for selected forest. All 15 fits completed; full/selected
logistic scores reproduce Step 10's reference.

Selected LR is the current untuned leader; selected features also lead within
the forest family. The forest's high training F1 and much lower validation F1
warrant regularization experiments. Higher forest ROC-AUC does not override
the agreed F1 selection metric. See [the Step 11 report](report/step11_model_comparison.md)
and [measured evidence](data/results/step11/README.md).

## Step 12 tuning reference

Both families have been tuned through an exhaustive 20-setting, 60-fit grid.
Selected-feature Logistic Regression is the development-selected model: **C=1.0, class_weight=balanced**,
mean cancellation F1 **0.732102**. The threshold remains 0.5.
Both untuned controls match Step 11 under documented numerical tolerances.
At Step 12, all 47 tests passed and all 60 fits completed
with no failures or convergence warnings. No final test rows were transformed
or scored and no full-development refit occurred at that stage; Step 13
subsequently performed the frozen evaluation.

## Step 13 completed evaluation

The frozen Logistic Regression pipeline was fitted on development and evaluated
once on the final test: F1 **0.750592**, accuracy 0.760874, precision 0.654187,
recall 0.880321 and ROC-AUC 0.875977. Confusion counts are TN 9,543, FP 4,526,
FN 1,164 and TP 8,562. No test result changed the model or threshold.

Read [the Step 13 report](report/step13_final_evaluation.md), [Notebook 05](notebooks/05_evaluation_and_error_analysis.ipynb)
and [the complete evidence](data/results/step13/README.md). Do not make new
test-driven modeling choices.

## Step 14 complete; next is Step 15

Read [the completed research-question answers](report/step14_research_answers.md)
and report Sections 7.3–8. Deposit type and lead time show strong descriptive
associations; prior cancellations are non-monotonic and repetition-sensitive.
The final model detects most cancellations but has substantial false alerts
and uncalibrated probabilities. Selected-feature balanced Logistic Regression
is best evaluated by development F1; PCA was tested but not retained.

All conclusions link to existing tables, distinguish causality from association,
and separate development model choices from final held-out performance. Step 14
does not change models, data, splits, figures, notebooks or predictions.

For Step 15, finalize the 150–200-word summary and template-compliant 10-page
Markdown/PDF report; resolve provenance and publish the unchanged raw CSV;
verify a clean environment, canonical notebook format and five fresh-kernel
runs; verify references, figure links and genuine member contributions.
See PROJECT_STATUS.md for the full checklist. Do not fabricate provenance,
execution checks or personal contributions to mark a gate complete.

## Findings to carry into model analysis

- Lead-time cancellation rates increase across the descriptive bins, including
  within each hotel. This is an association, not a causal estimate.
- Non-refundable rates are extremely high in both hotels. Discuss dependence
  on this signal and its uncertain source timing using the existing evidence;
  do not change the frozen pipeline in response to final test results.
- Prior cancellation counts are non-monotonic; a simple 'more always means
  worse' interpretation is unsupported. Groups above one cancellation are small.
- Repeated-record frequency materially affects rates. The equal-group EDA is
  a sensitivity view, not authorization to change the frozen dataset or weights.
- Hotel mix and partial-year/month coverage affect pooled summaries.

## Later gates

Step 10 has demonstrated selection and reduction with complete fold lists and
components. Step 11 has compared the majority baseline plus logistic
regression and random forest. Step 12 has tuned with the same three forward folds.
Primary metric is mean cancellation-class F1; secondary metrics are accuracy,
precision, recall, and ROC-AUC. Preserve seed 42 for stochastic components.
The chosen full pipeline was refitted on development and evaluated on the
final test in Step 13. Preserve it; later reproduction is verification only.

Steps 11–14 modeling, tuning, held-out analysis and research-question synthesis are complete. All 53 tests pass. Notebook 01
has ten preserved executed audit cells plus five newly executed EDA cells;
Notebook 02 has thirteen executed cells; Notebook 03 has ten executed Python
cells with saved outputs (five Step 9 and five Step 10); Notebook 04 preserves six Step 11 cells and adds five Step 12
executed Python cells; Notebook 05 adds six executed evidence-analysis cells. Full fresh-kernel runs, canonical
notebook validation, clean-install dependencies, raw CSV/provenance completion,
and the final 10-page Markdown/PDF report remain submission tasks.

Faraaz owns reviewing report Sections 1–3. Sadat owns Sections 4–8 and assembly.
Both complete genuine contributions, references, AI disclosure, and final checks.
