# Step 15.3 verification review

Reviewed on 3 September 2026 from the group-supplied `cse437_step15_3_verification.zip`. This is a review of supplied evidence, not a new run performed by this document or a certification of submission readiness.

## Evidence identity

| Artifact | SHA-256 |
| --- | --- |
| Complete supplied ZIP | `18840d43aaf572b35f191aafdb6c6ff287efc93e2a52e64c14a3043f474e8e36` |
| ZIP entry verification.json | `6dc3ba55d03462208c6ee2d7d512d6828d192968e62dc6a6f3f0ee89cab546be` |
| ZIP entry reproduction_comparison.json | `f02bd0847be1dcf4d6c776c951739c422f248a13a84e8eb6b15e13091999179f` |

All 35 manifest file hashes were checked. All five executed notebooks passed canonical validation; their cell sources and IDs match the repaired notebooks. All 32 locally available frozen files listed in the run context matched, as did the included Step 11 and Step 12 output hash chains. Full logs and executed copies remain in the supplied archive; they are not silently substituted for published historical outputs.

## Execution outcome

Dependency installation and consistency checks passed. The run passed 70 unit tests. Each notebook used a fresh kernel, had sequential code-cell execution counts and contained zero cell-error outputs.

| Notebook | Code cells | Seconds | Errors |
| --- | ---: | ---: | ---: |
| 01_data_audit_and_eda | 15 | 5.656 | 0 |
| 02_preprocessing | 13 | 12.281 | 0 |
| 03_feature_engineering | 10 | 18.032 | 0 |
| 04_modeling_and_tuning | 12 | 529.218 | 0 |
| 05_evaluation_and_error_analysis | 7 | 2.328 | 0 |

The runner reports `passed_with_reproduction_differences`: execution passed, numerical reproduction differed, and `submission_ready` is false. Exit code 2 is the documented numerical-warning outcome, not a notebook execution failure. Original repository and frozen evidence checks passed. Final evaluation verified cached evidence; it did not refit the final model or independently regenerate its predictions.

## Numerical comparison

| Balanced Logistic Regression candidate | Original mean development F1 | Rerun mean development F1 |
| --- | ---: | ---: |
| C=1, lr_06 | 0.7321017246390454 | 0.7313708930914117 |
| C=0.1, lr_04 | 0.7316967924435923 | 0.7316967924435923 |

The rerun winner is lr_04, while the original winner remains lr_06. Other C=1 and C=10 candidates changed; the largest absolute fold confusion-count difference is 41. Some Random Forest ROC-AUC values also differ slightly. These differences are not merely display rounding. The supplied prose summary understates the differences; this review follows the structured comparison and recorded tables.

Candidate parameters, semantic search protocol and recorded input/source hashes agree. Protocol byte differences include line endings. The feature representation and 0.5 threshold are unchanged. No rerun selection was promoted to frozen evaluation; the original final model and published test results remain unchanged.

## Environment and provenance limits

The supplied run used Windows 11, Python 3.12.10, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0 and joblib 1.5.3. Package-freeze records agree after decoding. CPU/numerical-library differences are hypotheses, not an established explanation for the changed optimization results.

The intended repair commit is `589ba0cc0714afc83306ec10ff0a43535163319f`. The supplied run used a downloaded repository archive and does not provide an exact checkout commit or Git status. Matching notebook sources and recorded source hashes do not establish identity of the entire tested checkout. Historical `final_verification.json` remains a record of its earlier run, not current-file certification.

## Remaining submission checks

- Recheck the raw CSV's public location and exact checksum; publication is group-reported complete.
- Bind final validation to an exact source commit and preserve executed notebook evidence with clear provenance.
- Retain the numerical reproduction limitation; do not retune from test results or weaken integrity assertions.
- Finalize the provisional author-supplied AI declaration and complete both members' whole-project review and attributable-commit checks.
- Inspect the rebuilt 10-page PDF and verify final public submission contents before declaring readiness.
