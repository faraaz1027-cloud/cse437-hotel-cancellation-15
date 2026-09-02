# Handoff from Faraaz's stages to Sadat

**CSE437 Group 15. Step 8 complete; Step 9 is next. Owner: Sadat.**

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
4. Read Notebook 01's development EDA and draft report Sections 1–3.

## Step 9: derived-feature candidates to implement and justify

- Total nights = weekend plus weekday nights; retain legitimate zero nights.
- Total guests = adults + children + babies, preserving unknown child totals
  until the documented fold-fitted imputation/indicator rule is applied.
- Prior-history presence and cancellation ratio. Define the denominator as
  previous cancellations plus previous noncanceled bookings; distinguish no
  recorded history from a genuinely observed zero cancellation ratio.
- Company presence, using source NULL semantics; avoid treating sparse company
  IDs as ordered measurements.
- Cyclic month/calendar features using fixed calendar rules. The first CV
  training period does not include all validation months.

Do not automatically add every candidate. Document formulas, missing/zero
denominator choices, and later validation results. Extend the preprocessing
schema explicitly; it currently rejects undeclared columns to prevent silent
feature loss. Learned feature decisions must stay inside training folds.

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

Step 10 must demonstrate both feature selection and dimensionality reduction,
with a justified final set. Step 11 compares the majority baseline plus logistic
regression and random forest. Step 12 tunes with the same three forward folds.
Primary metric is mean cancellation-class F1; secondary metrics are accuracy,
precision, recall, and ROC-AUC. Preserve seed 42 for stochastic components.
Only after choosing the full pipeline may it be refitted on all development
rows and evaluated on the final test in Step 13.

No predictive model has been trained. The 22 current tests pass. Notebook 01
has ten preserved executed audit cells plus five newly executed EDA cells;
Notebook 02 has thirteen executed cells. Full fresh-kernel runs, canonical
notebook validation, clean-install dependencies, raw CSV/provenance completion,
and the final 10-page Markdown/PDF report remain submission tasks.

Faraaz owns reviewing report Sections 1–3. Sadat owns Sections 4–8 and assembly.
Both complete genuine contributions, references, AI disclosure, and final checks.
