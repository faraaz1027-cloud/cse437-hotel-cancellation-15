# Development rerun and frozen-evaluation separation

## Diagnosed failure

The earlier Windows verification completed notebooks 01–04, then failed notebook 05's frozen-selection check. Fresh tuning selected balanced Logistic Regression with C=0.1; the published final model uses C=1. Notebook 04 had written the new selection to the same location used by the historical evaluation.

The supplied evidence records mean development F1 changing from 0.7321017246390454 to 0.7313708930914117 for C=1, balanced. C=0.1, balanced remained at 0.7316967924435923 and became the rerun winner. Other C=1 and C=10 candidates also changed. This is not merely display rounding. CPU/numerical-library differences are a hypothesis, not an established cause.

## Implemented boundary

- Notebook 04 copies development inputs and relevant source files into a new external workspace. New model-comparison and tuning evidence stays there. Final-test results and saved models are not copied into that workspace.
- The final comparison cell reports selection, candidate/fold metric, protocol, recorded input/source hash and environment differences. It never promotes a new winner to the published final evaluation.
- Notebook 05 requires the existing comparison cache. The original C=1 requirement, exact selection hash, cached output hashes and model checks remain intact. Missing artifacts raise an error rather than triggering a new fit.
- The verification runner preserves the original repository, checks frozen evidence after each notebook, captures failures/partial outputs, and distinguishes execution success from tuning-evidence agreement. New development tables and diagnostic files are collected separately.
- Windows path keys are normalized for the Step 10 reference lookup without changing stored digests or numerical tolerances. Git attributes preserve committed bytes across platforms. Run-local serialization differences are reported separately from semantic protocol changes.

## Interpretation and remaining checks

Existing published notebook outputs were retained as historical reference; newly added code has no fabricated outputs. Unit regression tests exercise changed winners, independent copies, preserved evidence, missing caches, path/newline differences and status reporting. Stubbed-kernel tests check orchestration only, not actual notebook execution.

The supplied Step 15.3 Windows run completed all five notebooks in fresh kernels with zero errors and passed 70 tests. The cached final comparison verified original evidence without training. Development scores and the winner still differed. This does not establish the cause of numerical drift, guarantee identical optimization results, refit the final model, or certify submission readiness. The archive does not establish the complete checkout's exact commit; see the [verification review](verification_review.md).

Use the commands and exit-code interpretation in [FINAL_CHECKS.md](../FINAL_CHECKS.md). In particular, `passed_with_reproduction_differences` means execution succeeded but the scientific differences still need review. It is not an unqualified pass.

Step 15.5 aligns the Markdown report and PDF with this reviewed run and user-confirmed contributions. The historical verification snapshot remains unchanged. Final declaration review, raw-publication recheck and joint submission sign-off remain separate tasks.
