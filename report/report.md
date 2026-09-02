# Hotel Booking Cancellation Project

**CSE437: Data Science | Group 15**

| Member | Student ID |
| --- | --- |
| Faraaz Jamil Chowdhury | 24241205 |
| Ihfaz Rashid Sadat | 23301499 |

Repository: https://github.com/faraaz1027-cloud/cse437-hotel-cancellation-15

**Working report:** Sections 1–8 are drafted from verified Steps 1–14, including the frozen held-out evaluation, answers to all three unchanged questions, and limitations. Step 15 must finalize the summary, contributions, provenance, reproducibility checks and PDF to the supplied template within 10 pages. Statements about an untouched test in earlier sections describe those stages, before the Step 13 evaluation.

## Summary

Final results and research-question answers are available. Write and verify the required 150–200-word summary during Step 15 assembly.

## 1. Problem and Dataset

### 1.1 Problem statement

Hotel booking cancellations can cause loss of money and make room planning difficult for hotels. The goal of this project is to study the factors that affect booking cancellations and build a machine-learning model that can predict whether a hotel booking will be canceled.

### 1.2 Dataset

The project uses Jesse Mostipak's [Hotel Booking Demand dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand), documented by Antonio et al. [1]. The supplied CSV combines city/resort hotel reservations: 119,390 rows, 32 columns, and 16,855,599 bytes (16.86 MB). Recorded arrivals span 2015-07-01 to 2017-08-31. The original publication describes extraction from hotel property-management databases.

The original bytes are preserved; the file checksum is in [data/README.md](../data/README.md). Exact download date, source version, and Kaggle licence still require confirmation. The raw CSV must still be committed for final submission.

### 1.3 Target and analytic population

The target is `is_canceled`: 1 for canceled and 0 otherwise. The original data contain 44,224 canceled and 75,166 noncanceled bookings. Excluding 180 records with a confirmed total of zero guests leaves 119,210 bookings. Missing guest counts do not establish zero guests.

The frozen development cohort has 95,415 bookings (2015-07-01 through 2017-04-22): 34,473 canceled and 60,942 noncanceled, a cancellation rate of **36.13%**. The 23,795 later bookings are reserved for final testing. Section 3 uses development data only; no held-out class distribution or model score is reported.

### 1.4 Approved research questions

1. Which booking and customer-related factors have the biggest effect on hotel cancellations?
2. How accurately can machine-learning models predict whether a hotel booking will be canceled?
3. Which machine-learning model gives the best result after data preprocessing, feature selection, dimensionality reduction, and hyperparameter tuning?

The wording above is unchanged. The predeclared factors are lead time, deposit type, and prior cancellations; observational associations are not treated as causal effects.

## 2. Data Handling and Preprocessing

### 2.1 Audit and eligibility

The original audit found 31,994 additional exact full-row copies, four missing child counts, 488 missing countries, 16,340 missing agents, and 112,593 missing companies. ADR ranged from −6.38 to 5,400. Both reservation-status fields encode outcome information and are excluded from predictors; the target is stored separately.

| Stage | Rows | Source predictor fields |
| --- | ---: | ---: |
| Original CSV | 119,390 | 29 after separating target and two status fields |
| Known-zero-guest exclusion | −180 | Unchanged |
| Eligible cohort | 119,210 | 29 |
| Initial preprocessing schema | 119,210 | 25 |

No booking identifier distinguishes accidental copies from legitimate repeated reservations. Records are retained and grouped by the 29 candidate fields for partition checks. No group crosses development/test or train/validation within a fold. The Step 8 equal-group analysis changes descriptive weighting only.

### 2.2 Missing values and outliers

For model preprocessing, negative ADR becomes missing; zero and high positive prices remain. Numeric medians are fitted inside each training fold. Missing country becomes `Unknown`; missing agent becomes `NoAgent` under the source NULL convention [1]. Agent IDs are nominal strings. Explicit `Undefined` categories remain distinct from unknown values.

The sparse company identifier and three potentially updated fields—assigned room, booking changes, and waiting-list days—are excluded from the initial model schema. This conservative policy does not prove booking-time availability of every retained field. Step 9 adds a company-code-recording indicator while continuing to exclude the identifier itself.

### 2.3 Transformations, scaling, and encoding

Lead time, prior cancellations, prior noncanceled bookings, and ADR receive log1p after imputation. Numeric scaling and one-hot vocabularies are learned only from each training fold. Unseen categories map to an all-zero block for that field. Children and ADR retain missing indicators. A tree-compatible variant omits numeric standardization. Entirely missing training columns use a documented zero fallback.

The three folds produce 328, 422, and 491 encoded columns, respectively; each train/validation pair has matching width and zero nonfinite values. ADR training medians are 76.50, 80.75, and 90.00. The negative ADR occurs in fold 3 validation and uses 90.00. No extra rows are removed and no final test row is fitted/transformed during these checks.

### 2.4 Reproducibility

Notebook 02 and reusable modules implement Steps 5–7; [the split plan](../data/splits/step6_split_plan.json) fixes dates and metrics. Section 3 EDA applies only fixed domain rules and reports observed-value statistics without statistical imputation. The original full-source quality audit was inspected before the holdout was constructed; that limitation is disclosed.

## 3. Statistical Analysis

### 3.1 Development descriptive statistics

All rows in this section belong to development. Negative ADR is counted as missing; valid ADR includes zero and high positive values. Missingness uses field-specific denominators.

| Field | Valid n | Mean | Median | Q1–Q3 | Maximum |
| --- | ---: | ---: | ---: | --- | ---: |
| Lead time (days) | 95,415 | 96.59 | 60 | 15–145 | 737 |
| ADR (source units) | 95,414 | 93.71 | 88 | 65–115 | 5,400 |
| Previous cancellations | 95,415 | 0.106 | 0 | 0–0 | 26 |
| Weekday nights | 95,415 | 2.45 | 2 | 1–3 | 50 |
| Weekend nights | 95,415 | 0.90 | 1 | 0–2 | 19 |

These summaries show skew in lead time and ADR and a concentration of prior cancellations at zero. Full descriptive, missingness, category, and monthly tables are in [data/eda](../data/eda/README.md). April 2017 ends on the 22nd and is a partial month.

### 3.2 Relationships and observations

1. **Lead time:** the cancellation rate rises across fixed bins, from **9.43%** at 0–7 days (n=17,247) to **80.81%** at 366+ days (n=2,079). The ascending pattern also appears within each hotel. This supports the planned association, not a causal effect.

![Lead time and cancellation](../figures/03_lead_time_cancellation.png)

2. **Deposit type:** non-refundable bookings have **99.25%** cancellation (n=12,296), versus **26.82%** without a deposit (n=82,979). Refundable bookings have **10.71%**, but only 140 records. Non-refundable rates remain high within both hotels. Policies, booking mix, or recording timing could contribute; no causal explanation is established.

![Deposit type and hotel stratification](../figures/04_deposit_cancellation.png)

3. **Prior cancellations:** rates are **32.07%, 95.65%, 31.29%, and 74.77%** for 0, 1, 2–3, and 4+ respectively. Counts are 89,079; 5,951; 163; and 222. The relationship is not monotonic; sparse higher-count groups require caution.
4. **Hotel mix:** City Hotel's rate is **41.41%** (n=62,827), compared with **25.94%** for Resort Hotel (n=32,588). Pooled interpretations should account for this difference.
5. **Repeated-record sensitivity:** assigning total weight one to each of 67,961 duplicate-profile groups changes the overall rate from **36.13% to 25.26%**. For 4+ prior cancellations, **74.77% becomes 26.32%**. Conflicting labels within a profile are averaged, not discarded. This alternative describes an average profile rather than an average booking; neither is asserted to be a correction to an unknown true population.

![Prior cancellations and weighting sensitivity](../figures/05_prior_cancellations_sensitivity.png)

These are descriptive associations, not model importance rankings or predictive performance. No p-values or row-independent confidence intervals are used because retained records need not be independent. Later modeling must preserve the frozen split, examine development-only comparisons, and avoid choosing settings from final test results.

## 4. Feature Engineering

### 4.1 Derived features (Step 9)

Eight fixed features yield **32 candidate fields before encoding**: 24 retained source fields plus the eight below. Month names are replaced by fixed cyclic coordinates.

| Feature | Definition |
| --- | --- |
| Total nights | Weekend + weekday nights |
| Total guests | Adults + children + babies |
| Previous bookings total | Prior canceled + prior noncanceled bookings |
| History presence | 1 for positive history total, 0 for zero, missing if unknown |
| Previous cancellation share | Prior cancellations / history total; zero for no history, interpreted with presence |
| Company code recorded | 1 for a present code, 0 for null; does not prove corporate payment |
| Month sine and cosine | sin/cos(2π(m−1)/12), m=1…12 |

Unknown components propagate to totals before training-fold median imputation;
known zero denominators get a zero share and a no-history flag. Separate imputed
totals need not equal sums of imputed components. Three totals receive log1p;
ratios, flags, and calendar coordinates do not. Children, ADR, total nights,
total guests, and cancellation share have fixed missing indicators.

The audit retains 604 zero-night development bookings and four unknown guest
totals. Both scaled and unscaled variants pass all three frozen folds, producing
332/421/490 encoded columns with matching train/validation widths and no nonfinite
values. Learned statistics use training data only. All 29 tests pass. Notebook 03
contains five executed Python cells. No target values are read by the feature
audit and no test rows are fitted/transformed.

These features summarize duration, party size, history and seasonality, but
their predictive benefit is untested. Fixed calendar coordinates cover the five
months missing from the first training window without proving seasonal
generalization. Full formulas and diagnostics are in [the Step 9 report](step9_feature_engineering.md).

### 4.2 Selection and dimensionality reduction

Step 10 removes training-constant encoded columns (variance ≤1e−12), ranks
remaining fields by training-label ANOVA F, and retains the top 75% rounded
upward. Ties follow encoded order. Scores are selection heuristics; no p-values
or causal effects are inferred. Centered full-SVD PCA separately reduces the
scaled numeric block to at least 95% cumulative training variance; categorical
and missing-indicator columns bypass PCA.

Four fixed representations use the same logistic-regression reference (C=1,
lbfgs, max_iter=2000, tol=1e−4, no class weights, seed=42, threshold=0.5) and the
three frozen forward folds. All preprocessing, selection and PCA fit within
each training fold. The protocol is stored with the results.

| Representation | Mean development F1 | Output columns across folds |
| --- | ---: | --- |
| All Step 9 features | 0.693094 | 332 / 421 / 490 |
| Selection only | **0.713609** | 247 / 314 / 366 |
| Numeric PCA | 0.694633 | 325 / 413 / 483 |
| Selection then PCA | 0.701023 | 242 / 308 / 360 |

Selection alone is the current choice: its mean F1 is highest, although its
third-fold F1 is slightly lower than all features. PCA was demonstrated but
not retained in the preferred pipeline because its validation result was worse.
PCA reduces 23 numeric fields to 16/15/16 components, retaining 95.88%/95.00%/
95.69% numeric variance. After selection, PCA reduces 20/21/21 numeric fields
to 15 components per fold. Variance retention does not guarantee predictive
information retention.

The preferred rule is refitted on each later training fold; complete retained
names and component coefficients are saved in [the Step 10 evidence](../data/processed/step10/README.md).
No global full-development mask is fitted now. These are development selection
scores, not unbiased final performance. Nonlinear interactions may be missed;
Step 11 retains an all-feature control for model-family comparison. All
35 tests pass and all 12 reference fits converged. The held-out test remains
untouched. See [the full Step 10 report](step10_selection_and_reduction.md).

## 5. Modeling and Validation

### 5.1 Models and protocol

Step 11 compares a training-majority baseline and two families: logistic
regression and random forest. Each learned family uses full and selected
features. Logistic regression uses C=1, lbfgs, max_iter=2000, tol=1e−4 and scaled
numeric inputs. Random forest uses 100 trees, unlimited depth, leaf size 1,
sqrt feature sampling and bootstrap with unscaled numeric inputs. Seeds are
42; neither class weighting nor resampling is used. Threshold is 0.5.

All five candidates use the same three frozen forward development folds.
Preprocessing and feature selection are training-only inside each pipeline.
Mean cancellation F1 is primary; accuracy, precision, recall and ROC-AUC are
secondary. This is the untuned comparison; Section 6 reports Step 12 tuning.

### 5.2 Development comparison

| Candidate | Mean F1 | Accuracy | Precision | Recall | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Majority baseline | 0.000000 | 0.638167 | 0.000000 | 0.000000 | 0.500000 |
| Logistic regression — full | 0.693094 | 0.751370 | 0.681025 | 0.753386 | 0.876098 |
| Logistic regression — selected | 0.713609 | 0.809314 | 0.783364 | 0.659498 | 0.882905 |
| Random forest — full | 0.626504 | 0.793498 | 0.906662 | 0.481119 | 0.886473 |
| Random forest — selected | 0.657994 | 0.803866 | 0.893717 | 0.521714 | 0.892267 |

The current leader is **Logistic regression — selected** (mean F1 0.713609).
The majority baseline has 63.82% mean accuracy but zero
cancellation F1/recall, showing why accuracy alone is insufficient. Forest
training F1 greatly exceeds its later-period validation F1; overfitting,
temporal shift and repeated profiles may contribute. Step 12 therefore examines
regularization; added model complexity alone does not establish improvement.

All 15 fits completed; the six logistic fits had no convergence warnings and
reproduce Step 10's metrics. Forty tests pass. Full fold scores, confusion
counts, feature names and parameters are in [the Step 11 evidence](../data/results/step11/README.md).
The reused development folds guide selection; these are not unbiased final
performance estimates. No final test row was transformed or scored and no
global full-development model was fitted. See [the detailed comparison](step11_model_comparison.md).

## 6. Hyperparameter Tuning

### 6.1 Search design

Exhaustive GridSearchCV evaluates 8 logistic settings (C=0.01/0.1/1/10 ×
class_weight=None/balanced) and 12 forest settings (max_depth=8/16/None ×
min_samples_leaf=1/10 × class_weight=None/balanced), totaling **60 fits** on
the three frozen development folds. Each grid includes its Step 11 control.
Forest size stays 100 trees with sqrt feature sampling; logistic uses lbfgs,
max_iter=2000; seed 42. The fixed selected-feature rule and 0.5 threshold remain.

Every candidate receives a fresh complete pipeline. Feature transformations
and selection fit only within each training fold. Balanced weights derive
from that training fold. Mean cancellation F1 selects settings; accuracy,
precision, recall and ROC-AUC provide secondary context. Search uses
`refit=False`, so no full-development estimator is fitted yet. The protocol,
tie rule, complete candidate/fold results and actual settings are saved.

### 6.2 Tuning results and decision

| Family | Selected search parameters | Untuned F1 | Tuned F1 | Change |
| --- | --- | ---: | ---: | ---: |
| Logistic Regression | C=1.0, class_weight=balanced | 0.713609 | 0.732102 | +0.018492 |
| Random Forest | class_weight=balanced, max_depth=None, min_samples_leaf=10 | 0.657994 | 0.669294 | +0.011300 |

The selected model is **Logistic Regression** with **C=1.0, class_weight=balanced** (mean F1
**0.732102**). This is the best evaluated grid setting, not
proof of global optimality or final test performance. Both untuned controls
reproduce Step 11's threshold-based metrics within 1e−12 (forest AUC tolerance
1e−7; numerical audit documented separately). All 60 fits complete without
convergence warnings or failed fits; all 47 tests pass. Notebook 04 adds five
actually Python-executed tuning cells.

For selected-feature logistic regression, balanced weighting raises mean recall
from 0.659498 to 0.756772 while precision falls from 0.783364 to 0.715322;
accuracy falls from 0.809314 to 0.800244. The F1 gain reflects this tradeoff,
not improvement on every metric. The best forest uses balanced weights and
leaf size 10; its training–validation F1 gap falls from 0.332209 to 0.203017,
but it still trails logistic regression on the agreed primary metric.

Development folds are reused for several choices, creating selection optimism;
no nested validation or confidence-tested performance gain is claimed. Class
weighting changes the precision/recall tradeoff. Final test rows remain untouched.
`final_selection.json` freezes the complete settings and threshold for Step 13.
See [the tuning report](step12_hyperparameter_tuning.md) and
[complete evidence](../data/results/step12/README.md).

## 7. Results, Visualization, and Error Analysis

### 7.1 Held-out performance

The frozen selected-feature Logistic Regression (`C=1`, balanced class weights,
threshold 0.5) was fitted on 95,415 development rows and evaluated once on
23,795 later bookings. No test score altered the pipeline.

| F1 | Accuracy | Precision | Recall | ROC-AUC | TN | FP | FN | TP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **0.750592** | 0.760874 | 0.654187 | 0.880321 | 0.875977 | 9,543 | 4,526 | 1,164 | 8,562 |

![Final test performance](../figures/09_final_test_performance.png)

The test cancellation rate is 40.87%. High recall and lower precision reflect
the balanced-weight model's emphasis on detecting cancellations; false positives
outnumber false negatives. Test F1 exceeds the 0.732102 development mean, but
one later period is not evidence of improvement or a confidence interval.

### 7.2 Error analysis

City Hotel error rate is 25.59%, compared with 20.24% for Resort Hotel. Online
TA has 31.95% error and contains 3,653 false positives. Short lead-time F1 is
weak (0.3344 for 0–7 days; 0.5882 for 8–30), whereas 366+ days has F1 0.9481.
No Deposit contains almost every error; 2,290 of 2,291 Non Refund bookings
cancel, giving near-perfect separation. These post-test groups are descriptive,
not reasons to change the model.

![Final probability and hotel errors](../figures/10_final_error_analysis.png)

Fixed-bin observed rates lie below mean predicted probabilities, so outputs are
not presented as calibrated probabilities. A confident false positive (source
14,182) received probability 0.999603; a confident false negative (source
94,387) received 0.006030. Twenty distinct examples show confident and
near-threshold FP/FN cases. Full subgroup metrics, coefficients and examples
are in [Step 13 evidence](../data/results/step13/README.md) and the
[detailed evaluation](step13_final_evaluation.md).

The strongest positive coefficient is Non Refund; several large coefficients
are agent categories. Coefficients on encoded/scaled inputs are associations,
not causal or uniformly comparable importance. Repeated profiles, source timing,
temporal mix and small subgroups limit interpretation. All 53 tests pass; the
fitted pipeline is saved. Notebook 05 has six executed evidence-analysis cells.

### 7.3 Answers to the approved research questions

**1. Which booking and customer-related factors have the biggest effect on hotel cancellations?**

Among the three predeclared factors, deposit type shows a large descriptive
separation: development cancellation is 99.25% for Non Refund versus 26.82%
for No Deposit. Lead-time rates increase across all six fixed bins, from 9.43%
at 0–7 days to 80.81% at 366+ days, also increasing within both hotels.
Prior cancellations are non-monotonic: rates for 0, 1, 2–3 and 4+ are 32.07%,
95.65%, 31.29% and 74.77%. Equal-profile weighting changes the 4+ rate to
26.32%, demonstrating repetition sensitivity. These are associations, not
causal effects or an exhaustive ranking of every predictor. The largest
positive encoded coefficient is Non Refund (+2.929658), but different scaling,
coding and correlated features prevent interpreting magnitudes as a universal
raw-feature importance ranking.

Sources: [development EDA](../data/eda/README.md),
[prior-history sensitivity](../data/eda/sensitivity_prior_cancellations.csv),
and [model coefficients](../data/results/step13/feature_coefficients.csv).

**2. How accurately can machine-learning models predict whether a hotel booking will be canceled?**

The selected model achieves F1 0.750592, accuracy 76.09%, precision 65.42%,
recall 88.03% and ROC-AUC 0.875977 on 23,795 held-out arrivals from 2017-04-23
through 2017-08-31. It detects 8,562 cancellations, misses 1,164 and falsely
flags 4,526 noncancellations. Thus, high detection comes with substantial false
alerts. Short-lead and Online TA performance is weaker, and probability-bin
diagnostics show overprediction. These are period-specific results for the
selected model, not guaranteed performance for other hotels or calibrated
probabilities. No financial savings were measured. Sources:
[final metrics](../data/results/step13/final_metrics.csv) and
[subgroup diagnostics](../data/results/step13/subgroup_metrics.csv).

**3. Which machine-learning model gives the best result after data preprocessing, feature selection, dimensionality reduction, and hyperparameter tuning?**

Selected-feature Logistic Regression with C=1 and balanced class weights is
**best evaluated under this protocol**, at mean development F1 0.732102 versus
0.669294 for the best tested forest. Selection improves the reference logistic
F1 from 0.693094 to 0.713609; numeric PCA (0.694633) and selection-plus-PCA
(0.701023) do not beat selection alone. PCA was demonstrated but is excluded
from the final pipeline. Tuning increases selected-logistic F1 by 0.018492,
with a recall gain and precision/accuracy reductions. All choices use the
frozen forward development folds; the 0.5 threshold and pipeline are unchanged
after test access. This is a finite, staged comparison, not universal model
superiority or a test-set ranking of all candidates. Sources:
[representations](../data/processed/step10/representation_comparison.csv),
[tuning comparison](../data/results/step12/tuning_comparison.csv), and
[frozen selection](../data/results/step12/final_selection.json).

The [Step 14 synthesis](step14_research_answers.md) provides the complete
question-to-evidence mapping, group denominators and interpretation boundaries.

## 8. Limitations and Next Steps

### 8.1 Limits of the evidence

- **Retrospective source timing:** removing direct status leakage and potentially
  updated fields does not prove all retained predictors were available at
  booking time. The near-separation of Non Refund warrants scrutiny, not a
  claim that deposits cause cancellations.
- **Repeated profiles and small groups:** profiles are kept and protected from
  within-fold/development–test overlap, but booking-weighted results can be
  dominated by frequent profiles. Equal-group EDA answers a different question;
  sparse categories cannot support strong generalizations or naive independent-
  row confidence intervals.
- **Temporal and population scope:** this is one city/resort source and one
  later-arrival holdout, with partial seasonal coverage. The full-source quality
  audit preceded splitting. Final prevalence is 40.87%, versus 36.13% in
  development; a higher test F1 is not proof of improvement or generalization.
- **Selection and search:** development folds were reused for representation,
  model and parameter choices, so their scores can be optimistic. The finite
  grid is not global optimization; PCA was only compared with the reference
  logistic model. No isolated feature ablation or statistically significant
  improvement is established.
- **Decision usefulness:** weighted probabilities are not calibrated. False
  positives and negatives have unmeasured costs, so no deployment readiness,
  automatic overbooking policy or financial benefit is claimed.

### 8.2 Conclusion and next work

The measured evidence supports useful retrospective cancellation screening,
with deposit type and lead time strongly associated with outcomes. The
selected logistic pipeline leads the tested alternatives on the agreed
development F1 objective and detects most held-out cancellations, but false
alerts, subgroup weakness and source-timing uncertainty remain material.

Future work could verify predictor availability at the decision time and
evaluate deposit ablation, calibration, cost-based thresholds and external
hotel/time coverage under a new predeclared design with fresh evaluation data.
These are proposals, not completed experiments, and must not alter the frozen
current result. Step 15 must resolve provenance/raw-data publication, clean-
environment and fresh-kernel checks, genuine contributions, verified references,
the final summary, and the template-compliant 10-page report/PDF.

## 9. Contributions

Faraaz owns Steps 1–8 and draft Sections 1–3; Sadat owns Steps 9–15 and later report assembly. Record each member's actual reviewed/implemented work and commits before submission. Do not treat assigned ownership as proof of personal work.

## References and assistance

[1] Antonio, N., de Almeida, A., and Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief*, 22, 41–49. https://doi.org/10.1016/j.dib.2018.11.126

[2] Jesse Mostipak. Hotel Booking Demand. Kaggle. https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand . Exact version, download date, and licence pending verification.

OpenAI ChatGPT/Codex assisted with scaffolding, user-approved transcription, code, testing, execution, figures, interpretation, and this draft. No predictive results have been fabricated. Notebook 01 preserves ten earlier IPython-executed audit cells and adds five Python-executed EDA cells with actual captured outputs. Notebook 03 preserves five Step 9 cells and appends five actually Python-executed Step 10 cells. Notebook 04 preserves six actually Python-executed Step 11 cells and adds five actually Python-executed Step 12 cells. Notebook 05 contains six actually Python-executed evidence-analysis cells for Step 13. Full fresh-kernel notebook execution and canonical format validation remain final gates. Produce report.pdf after completing and checking the report.
