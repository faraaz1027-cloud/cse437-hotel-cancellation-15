# Step 14 — Answers to the approved research questions

**CSE437 Data Science | Group 15 | Owner: Sadat | Next: Step 15**

This synthesis uses the evidence published through Step 13 at commit
[`ad4ddff`](https://github.com/faraaz1027-cloud/cse437-hotel-cancellation-15/commit/ad4ddfffcd191b479021754ad16e0f96a4dd2397).
The dataset, target (`is_canceled`), original problem and question wording are
unchanged. No model is fitted, no new predictions are made, and no threshold,
feature set, weighting or calibration decision is changed in Step 14.

## 1. Which booking and customer-related factors have the biggest effect on hotel cancellations?

**Answer:** Among the three predeclared factors, deposit type shows a large
descriptive separation, longer lead time has a consistent increasing
association, and prior cancellations show a substantial but non-monotonic and
repetition-sensitive association. These are observed relationships, not
identified causal effects or an exhaustive ranking of all predictors.

The following rates use only the **95,415 development bookings**:

| Factor | Observed development result | Interpretation and qualification |
| --- | --- | --- |
| Deposit type | Non Refund: **99.25%** canceled (12,296 bookings); No Deposit: **26.82%** (82,979); Refundable: **10.71%** (140) | Large separation between the two well-populated categories. Refundable is small; the data do not establish that a deposit policy causes cancellation. |
| Lead time | Rates increase through all six fixed bins, from **9.43%** at 0–7 days (17,247 bookings) to **80.81%** at 366+ days (2,079) | The increasing pattern also occurs within both hotels. This does not mean every long-lead booking cancels or that lead time has an isolated causal effect. |
| Previous cancellations | For 0, 1, 2–3 and 4+: **32.07%, 95.65%, 31.29%, 74.77%**, with n=89,079; 5,951; 163; 222 | “More previous cancellations always means higher risk” is unsupported. The two highest-count groups are sparse. |

Evidence: [deposit table](../data/eda/deposit_rates.csv),
[lead-time table](../data/eda/lead_time_rates.csv),
[prior-cancellation table](../data/eda/prior_cancellation_rates.csv),
[lead time within hotels](../data/eda/lead_time_by_hotel.csv), and
[deposit within hotels](../data/eda/deposit_by_hotel.csv).

Repeated-profile weighting changes the interpretation. Giving each profile
group total weight one changes the overall development cancellation rate from
**36.13% to 25.26%**; for 4+ prior cancellations, **74.77% becomes 26.32%**.
The Non Refund rate remains high at **93.49%**, and the lead-time endpoints
remain ordered (**8.17% to 47.10%**). This sensitivity describes an average
profile rather than an average booking. It is not a reason to delete retained
records or change the frozen model weights.
Sources: [EDA summary](../data/eda/eda_summary.json),
[prior-history sensitivity](../data/eda/sensitivity_prior_cancellations.csv),
[deposit sensitivity](../data/eda/sensitivity_deposit.csv), and
[lead-time sensitivity](../data/eda/sensitivity_lead_time.csv).

The final fitted model also assigns the largest positive encoded coefficient
to `deposit_type_Non Refund` (**+2.929658**); parking spaces has a negative
coefficient (**−2.290799**), and several agent categories have large
coefficients. These are conditional model associations in an encoded,
regularized representation. Numeric scaling, log transforms, correlated fields
and categorical coding prevent reading coefficient magnitudes as a universal
raw-feature importance ranking. No causal or independently validated
importance ranking was estimated. Source:
[all 406 coefficients](../data/results/step13/feature_coefficients.csv).

## 2. How accurately can machine-learning models predict whether a hotel booking will be canceled?

**Answer:** The frozen selected-feature Logistic Regression achieved **F1
0.750592**, **76.09% accuracy**, **65.42% precision**, **88.03% recall**, and
**ROC-AUC 0.875977** on **23,795 held-out bookings**, with arrivals from
2017-04-23 through 2017-08-31. These are results for the selected model in one
later period—not test scores for every candidate or a guarantee for new hotels.

The model correctly detected **8,562 of 9,726 cancellations**, missed **1,164**,
and incorrectly flagged **4,526 noncancellations**; **9,543** noncancellations
were correctly classified. It catches most cancellations, but a substantial
number of alerts are false. Balanced weighting improved the development F1
tradeoff, not every metric. The threshold stayed at **0.5**.
Sources: [final metrics](../data/results/step13/final_metrics.csv),
[evaluation protocol](../data/results/step13/evaluation_protocol.json), and
[evaluation summary](../data/results/step13/evaluation_summary.json).

The **40.87%** test cancellation prevalence differs from development's
**36.13%**. Test F1 being above the **0.732102** development mean is not an
estimated improvement: the training size, booking mix and time windows differ.
Only the development-selected pipeline received the official final evaluation.

Errors are uneven: F1 is **0.3344** for lead times of 0–7 days and **0.5882**
for 8–30 days; Online TA's error rate is **31.95%**, and it contributes
**3,653 false positives**. City Hotel's error rate is **25.59%**, versus
**20.24%** for Resort Hotel. Group sizes and prevalence differ, so these
post-test diagnostics are not causal explanations or model-selection rules.
The test Non Refund group contains **2,290 cancellations among 2,291 bookings**;
this near-separation warrants caution about source timing and booking mix.
Sources: [subgroup metrics](../data/results/step13/subgroup_metrics.csv) and
[the detailed error analysis](step13_final_evaluation.md).

In every populated fixed probability bin, the observed cancellation rate is
below the mean predicted probability. The outputs therefore should not be
advertised as calibrated probabilities. The Brier score is **0.160007**, a
probability-error measure, not another accuracy percentage. No calibration or
threshold adjustment is made after the test. Source:
[probability diagnostics](../data/results/step13/probability_diagnostics.csv).

Operationally, the evidence supports further evaluation of a cancellation
screening aid, not an automatic cancellation/overbooking policy. False alerts
and missed cancellations have different costs; those costs and prospective
financial benefits were not measured.

## 3. Which machine-learning model gives the best result after data preprocessing, feature selection, dimensionality reduction, and hyperparameter tuning?

**Answer:** Selected-feature **Logistic Regression with C=1 and balanced class
weights** is the **best evaluated under this protocol**, using mean cancellation
F1 across the three frozen forward development folds as the primary metric.
It achieves **0.732102**, versus **0.669294** for the best tested Random Forest.
The final pipeline does **not** include PCA: dimensionality reduction was
tested and documented, but the selected-only representation performed better.

| Development comparison | Mean cancellation F1 | Decision supported |
| --- | ---: | --- |
| Majority baseline | 0.000000 | Accuracy alone can hide failure to detect cancellations. |
| Reference Logistic Regression, all Step 9 features | 0.693094 | Full-feature control. |
| Reference Logistic Regression, selection only | 0.713609 | Best of the four reference representations. |
| Reference Logistic Regression, numeric PCA | 0.694633 | Does not beat selection only. |
| Reference Logistic Regression, selection then PCA | 0.701023 | Does not beat selection only. |
| Untuned Random Forest, selected features | 0.657994 | Trails selected Logistic Regression on F1. |
| Tuned Random Forest, selected features | 0.669294 | Best tested forest: balanced weights, unlimited depth, leaf size 10. |
| Tuned Logistic Regression, selected features | **0.732102** | Final development-selected pipeline: C=1, balanced weights. |

Sources: [representation comparison](../data/processed/step10/representation_comparison.csv),
[untuned model comparison](../data/results/step11/model_comparison.csv),
[tuning comparison](../data/results/step12/tuning_comparison.csv), and
[frozen settings](../data/results/step12/final_selection.json).

Selection retains the top 75% of nonconstant encoded training features by
ANOVA F; numeric PCA retains at least 95% of training numeric variance. All
learned transformations, selection and PCA were fitted inside training folds.
The grid evaluated **20 settings × 3 folds = 60 fits**. Logistic tuning raises
mean F1 by **0.018492**, but mean accuracy falls from **0.809314 to 0.800244**
while recall rises from **0.659498 to 0.756772**. This is a precision/recall
tradeoff, not improvement on every measure.

These are staged comparisons, not an exhaustive search over every model and
representation combination. PCA variants were compared with the fixed
logistic reference, not tuned for both families. Derived features were
evaluated as a bundle; no isolated ablation establishes each feature's benefit.
Reusing development folds for several choices introduces selection optimism;
there is no nested-validation or statistical-significance claim. The test
result confirms the selected pipeline's period-specific performance, but
cannot establish a test-set ranking over models that were not tested there.

## Conclusion and Step 15 handoff

The project demonstrates a complete preprocessing-to-evaluation workflow and
finds useful retrospective predictive signal. Deposit and lead-time patterns
are strong descriptive findings; prior-history interpretation is sensitive to
repetitions. The chosen logistic pipeline balances cancellation detection and
false alerts better on the agreed development F1 metric than the alternatives
tested, while remaining limited by feature timing, repeated profiles, temporal
coverage, uncalibrated probabilities and the narrow model search.

**Step 14 is complete; Step 15 belongs to Sadat, with both members reviewing.**
Finish the 150–200-word summary and template-compliant 10-page report/PDF;
verify dataset provenance/licence and add the unchanged raw CSV; validate a
clean dependency install and fresh-kernel execution of all five notebooks;
record genuine contributions, references and AI assistance. Do not invent
missing provenance or personal contributions. These submission gates remain
open; this synthesis does not claim the entire project is submission-ready.

Any future calibration, feature-timing study, deposit ablation, threshold/cost
study or external validation must use a newly declared development/evaluation
design and fresh evaluation data. It must not change this frozen result.

Prepared with ChatGPT/Codex assistance; assigned ownership does not establish
which work a member personally performed. Review the conclusions before
recording actual contributions.
