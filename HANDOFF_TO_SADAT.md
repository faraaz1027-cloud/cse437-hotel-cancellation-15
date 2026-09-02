# Handoff from Faraaz's stages to Sadat

**CSE437 Group 15. Step 11 complete; Step 12 is next. Owner: Sadat.**

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
   report Sections 1–5. Notebook 04 adds the Step 11 model-family comparison.

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

## Step 11 completed; next is Step 12

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

Step 12 should tune both families using fresh `make_model_pipeline` instances
from `src/modeling.py`. Keep all feature learning inside CV, use development-
relative fold indices and seed 42, document the spaces/results, and compare
with the untuned references. Add actual search outputs to Notebook 04 and
report Section 6. Any threshold changes use development data only. Freeze all
choices before final refitting and test evaluation in Step 13.

## Findings to carry into model analysis

- Lead-time cancellation rates increase across the descriptive bins, including
  within each hotel. This is an association, not a causal estimate.
- Non-refundable rates are extremely high in both hotels. Consider a
  development-only with/without-deposit comparison to understand dependence
  on this signal; do not use final test results to choose it.
- Prior cancellation counts are non-monotonic; a simple 'more always means
  worse' interpretation is unsupported. Groups above one cancellation are small.
- Repeated-record frequency materially affects rates. The equal-group EDA is
  a sensitivity view, not authorization to change the frozen dataset or weights.
- Hotel mix and partial-year/month coverage affect pooled summaries.

## Later gates

Step 10 has demonstrated selection and reduction with complete fold lists and
components. Step 11 has compared the majority baseline plus logistic
regression and random forest. Step 12 tunes with the same three forward folds.
Primary metric is mean cancellation-class F1; secondary metrics are accuracy,
precision, recall, and ROC-AUC. Preserve seed 42 for stochastic components.
Only after choosing the full pipeline may it be refitted on all development
rows and evaluated on the final test in Step 13.

Step 11 model-family comparisons are complete; tuning and final held-out
results remain pending. All 40 tests pass. Notebook 01
has ten preserved executed audit cells plus five newly executed EDA cells;
Notebook 02 has thirteen executed cells; Notebook 03 has ten executed Python
cells with saved outputs (five Step 9 and five Step 10); Notebook 04 adds six
executed Python cells. Full fresh-kernel runs, canonical
notebook validation, clean-install dependencies, raw CSV/provenance completion,
and the final 10-page Markdown/PDF report remain submission tasks.

Faraaz owns reviewing report Sections 1–3. Sadat owns Sections 4–8 and assembly.
Both complete genuine contributions, references, AI disclosure, and final checks.
