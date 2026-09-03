"""Execute preprocessing on the fixed DEVELOPMENT folds only.

Exports aggregate evidence and feature schemas, not globally fitted matrices.
Run: python -m src.preprocessing_audit
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import sklearn

from .preprocessing import (make_preprocessor, LOG_COLUMNS, NUMERIC_COLUMNS,
                            CATEGORICAL_COLUMNS, MODEL_COLUMNS, POLICY_EXCLUSIONS)
from .splitting import check_assignments, development_cv


def fitted_state(pipeline):
    """Only learned training statistics, used to prove transform does not refit."""
    transformer = pipeline.named_steps["columns"]
    result = {}
    for name in ("log_numeric", "numeric"):
        branch = transformer.named_transformers_[name]
        scaler = branch.named_steps["scale"]
        result[name] = {"medians": branch.named_steps["impute"].statistics_.tolist()}
        if scaler != "passthrough":
            result[name]["means"] = scaler.mean_.tolist()
            result[name]["scales"] = scaler.scale_.tolist()
            result[name]["rows_seen"] = int(scaler.n_samples_seen_)
    result["categories"] = [values.tolist() for values in
        transformer.named_transformers_["categorical"].categories_]
    return result


def checked_matrix(matrix, expected_rows):
    if not sparse.issparse(matrix):
        raise AssertionError("Expected a sparse encoded matrix.")
    if matrix.shape[0] != expected_rows or not np.isfinite(matrix.data).all():
        raise AssertionError("Rows changed or nonfinite values survived preprocessing.")
    return {"rows": matrix.shape[0], "columns": matrix.shape[1],
            "nonzero": int(matrix.nnz), "nonfinite_values": 0,
            "density": float(matrix.nnz / (matrix.shape[0] * matrix.shape[1]))}


def run_preprocessing_audit(root: Path):
    processed, splits = root / "data/processed", root / "data/processed/splits"
    plan_path = splits / "validation_split_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    path = splits / plan["assignment_file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != plan["assignment_sha256"]:
        raise ValueError("Assignment checksum changed.")
    input_hashes = {}
    for filename in ["eligibility_candidates.csv.gz", "eligibility_metadata.csv.gz"]:
        sha = hashlib.sha256((processed / filename).read_bytes()).hexdigest()
        if sha != plan["upstream_output_sha256"][filename]:
            raise ValueError(f"Upstream input changed: {filename}")
        input_hashes[filename] = sha
    assignments = pd.read_csv(path)
    check_assignments(assignments)
    metadata = pd.read_csv(processed / "eligibility_metadata.csv.gz")
    candidates = pd.read_csv(processed / "eligibility_candidates.csv.gz")
    if assignments.source_row_id.tolist() != metadata.source_row_id.tolist():
        raise ValueError("Split metadata is misaligned with eligibility row order.")
    if len(candidates) != len(assignments):
        raise ValueError("Candidate and assignment lengths differ.")
    dev_rows = np.flatnonzero(assignments.partition.eq("development"))
    X_dev = candidates.iloc[dev_rows].reset_index(drop=True)
    del candidates  # no held-out feature transformation or summary below
    cv = development_cv(assignments)
    folds = []
    schemas = {}
    for fold, (train_idx, val_idx) in enumerate(cv, start=1):
        train, val = X_dev.iloc[train_idx], X_dev.iloc[val_idx]
        pipe = make_preprocessor()
        encoded_train = pipe.fit_transform(train)
        state_before = fitted_state(pipe)
        encoded_val = pipe.transform(val)
        if state_before != fitted_state(pipe):
            raise AssertionError("Validation transform changed training-fitted state.")
        train_info = checked_matrix(encoded_train, len(train))
        val_info = checked_matrix(encoded_val, len(val))
        if train_info["columns"] != val_info["columns"]:
            raise AssertionError("Train/validation schemas differ within a fold.")
        clean_train = pipe.named_steps["domain"].transform(train)
        clean_val = pipe.named_steps["domain"].transform(val)
        branches = pipe.named_steps["columns"].named_transformers_
        medians = {}
        for branch_name, column_names in [("log_numeric", LOG_COLUMNS), ("numeric", NUMERIC_COLUMNS)]:
            fitted_medians = branches[branch_name].named_steps["impute"].statistics_
            expected = clean_train[list(column_names)].median().fillna(0).to_numpy()
            np.testing.assert_allclose(fitted_medians, expected)
            medians.update(dict(zip(column_names, [float(x) for x in fitted_medians])))
            imputed = branches[branch_name].named_steps["impute"].transform(clean_train[list(column_names)])
            before_scaling = np.log1p(imputed) if branch_name == "log_numeric" else imputed
            scaler = branches[branch_name].named_steps["scale"]
            np.testing.assert_allclose(scaler.mean_, before_scaling.mean(axis=0), rtol=1e-10, atol=1e-10)
            if int(scaler.n_samples_seen_) != len(train):
                raise AssertionError("Scaler fitted on unexpected number of rows.")
        categories = branches["categorical"].categories_
        unseen = {col: int((~clean_val[col].isin(known)).sum())
                  for col, known in zip(CATEGORICAL_COLUMNS, categories)}
        for col, known in zip(CATEGORICAL_COLUMNS, categories):
            if set(known) != set(clean_train[col]):
                raise AssertionError("Encoder vocabulary is not training-only.")
        # The tree-compatible variant differs only in numeric standardization.
        tree_pipe = make_preprocessor(scale_numeric=False)
        tree_train = tree_pipe.fit_transform(train)
        tree_val = tree_pipe.transform(val)
        checked_matrix(tree_train, len(train))
        checked_matrix(tree_val, len(val))
        if tree_train.shape != encoded_train.shape or tree_val.shape != encoded_val.shape:
            raise AssertionError("Scaled and unscaled schemas differ.")
        names = pipe.get_feature_names_out().tolist()
        if len(names) != encoded_train.shape[1] or len(set(names)) != len(names):
            raise AssertionError("Feature schema is incomplete or contains duplicate names.")
        schemas[f"fold_{fold}"] = names
        folds.append({
            "fold": fold, "training": train_info, "validation": val_info,
            "training_missing_numeric_before_imputation": {col: int(clean_train[col].isna().sum())
                for col in LOG_COLUMNS + NUMERIC_COLUMNS if clean_train[col].isna().any()},
            "validation_missing_numeric_before_imputation": {col: int(clean_val[col].isna().sum())
                for col in LOG_COLUMNS + NUMERIC_COLUMNS if clean_val[col].isna().any()},
            "training_negative_adr_to_missing": int(train.adr.lt(0).sum()),
            "validation_negative_adr_to_missing": int(val.adr.lt(0).sum()),
            "training_zero_adr_retained": int(train.adr.eq(0).sum()),
            "validation_zero_adr_retained": int(val.adr.eq(0).sum()),
            "training_adr_above_1000_retained": int(train.adr.gt(1000).sum()),
            "validation_adr_above_1000_retained": int(val.adr.gt(1000).sum()),
            "training_country_missing_to_unknown": int(train.country.isna().sum()),
            "validation_country_missing_to_unknown": int(val.country.isna().sum()),
            "training_agent_missing_to_no_agent": int(train.agent.isna().sum()),
            "validation_agent_missing_to_no_agent": int(val.agent.isna().sum()),
            "training_company_nulls_in_excluded_column": int(train.company.isna().sum()),
            "validation_company_nulls_in_excluded_column": int(val.company.isna().sum()),
            "validation_unseen_category_rows_by_field": unseen,
            "training_medians": medians,
            "numeric_scaler_rows_seen": len(train),
            "validation_did_not_change_fitted_state": True,
            "unscaled_tree_variant_verified": True,
        })
    summary = {
        'analysis': 'preprocessing', "status": "completed", "scope": "preprocessing of frozen development train/validation folds only",
        "development_rows": len(X_dev), "test_rows_fitted_or_transformed": 0,
        "target_file_read_by_audit": False, "predictive_models_trained": 0,
        "rows_removed": 0, "input_candidate_columns": 29,
        "retained_source_fields": list(MODEL_COLUMNS), "retained_source_field_count": len(MODEL_COLUMNS),
        "excluded_fields": POLICY_EXCLUSIONS,
        "log1p_fields": list(LOG_COLUMNS), "regular_numeric_fields": list(NUMERIC_COLUMNS),
        "categorical_fields": list(CATEGORICAL_COLUMNS),
        "fixed_missing_indicators": ["children", "adr"],
        "rules": {"numeric_missing": "training-fold median; zero fallback for an entirely missing training field",
                  "adr_negative": "convert to missing before train-fitted imputation",
                  "adr_nonnegative": "retain zeros and high values; log1p instead of clipping",
                  "country_missing": "Unknown", "agent_missing": "NoAgent (source NULL convention)",
                  "explicit_undefined_category": "preserve as a category distinct from Unknown",
                  "unseen_categories": "all-zero one-hot block with handle_unknown=ignore",
                  "scaled_variant": "training-fold StandardScaler after imputation/log1p",
                  "tree_variant": "same rules without numeric StandardScaler"},
        "folds": folds,
        "input_sha256": {**input_hashes, "validation_assignments.csv.gz": plan["assignment_sha256"],
            "validation_split_plan.json": hashlib.sha256(plan_path.read_bytes()).hexdigest()},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                    "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "limitations": ["Candidate-field timing is not guaranteed by source snapshots; this remains retrospective evaluation.",
            "Medians/log transforms are fixed preprocessing choices, not proven best by predictive performance.",
            "Unseen categories map to zeros, so category-specific information can be lost.",
            "Feature engineering, statistical feature selection, and dimensionality reduction remain additional analyses.",
            "Do not reuse these fold diagnostics as a preprocessed full-cohort matrix for model selection."],
    }
    output = processed / "preprocessing"
    output.mkdir(parents=True, exist_ok=True)
    (output / "preprocessing_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "feature_schemas.json").write_text(json.dumps(schemas, indent=2) + "\n", encoding="utf-8")
    return summary, schemas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    summary, _ = run_preprocessing_audit(args.root)
    print(json.dumps({"development_rows": summary["development_rows"],
        "retained_source_fields": summary["retained_source_field_count"],
        "folds": [{"fold": f["fold"], "training": f["training"], "validation": f["validation"],
                   "training_medians": {k:f["training_medians"][k] for k in ["children", "adr"]}}
                  for f in summary["folds"]], "test_rows_processed": 0}, indent=2))


if __name__ == "__main__":
    main()
