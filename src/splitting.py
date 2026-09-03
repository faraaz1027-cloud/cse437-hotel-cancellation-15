"""validation: fixed arrival-date holdout and forward development validation.

Run from the repository root with: python -m src.splitting
Only dates, row identities, and duplicate-group identities determine the split.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

POLICY_VERSION = "1.0"
DEVELOPMENT_FRACTION = 0.80
CV_FRACTIONS = (0.25, 0.50, 0.75)
METADATA_SHA256 = "24655f99f16d70acde3295c270438bdc096d5d94f5b572d649a8a9bb1927e9db"
FUTURE_MODEL_SEED = 42


def nearest_date_boundary(dates: pd.Series, fraction: float) -> pd.Timestamp:
    """Choose the whole-date boundary nearest a target prefix row count.

    Both sides must be nonempty. Ties choose the earlier boundary. This uses
    row counts, not equal-duration blocks; it never consults the target.
    """
    if not 0 < fraction < 1:
        raise ValueError("Boundary fraction must lie strictly between zero and one.")
    counts = dates.value_counts().sort_index()
    if len(counts) < 2:
        raise ValueError("At least two distinct dates are needed for a boundary.")
    before = counts.cumsum() - counts
    return (before.iloc[1:] - len(dates) * fraction).abs().idxmin()


def partition_record(frame: pd.DataFrame, name: str) -> dict:
    return {"partition": name, "rows": len(frame),
            "start": frame["arrival_date"].min(), "end": frame["arrival_date"].max(),
            "duplicate_groups": int(frame["duplicate_group_id"].nunique())}


def check_assignments(assignments: pd.DataFrame) -> dict:
    """Validate time ordering, row/group separation, and forward fold roles."""
    a = assignments
    if not np.array_equal(a["cohort_row"].to_numpy(), np.arange(len(a))):
        raise ValueError("Cohort row order must match the eligibility files exactly.")
    if a["source_row_id"].isna().any() or not a["source_row_id"].is_unique:
        raise ValueError("Source row IDs must be present and unique.")
    if a["duplicate_group_id"].isna().any():
        raise ValueError("Duplicate group IDs cannot be missing.")
    dates = pd.to_datetime(a["arrival_date"], format="%Y-%m-%d", errors="raise")
    if dates.isna().any():
        raise ValueError("Arrival dates cannot be missing.")
    if set(a["partition"].unique()) != {"development", "test"}:
        raise ValueError("Both development and test must be nonempty, with no other partition.")
    if a.groupby("duplicate_group_id")["arrival_date"].nunique().max() != 1:
        raise ValueError("A duplicate group spans arrival dates; resolve before splitting.")
    dev = a["partition"].eq("development")
    test = ~dev
    if dates[dev].max() >= dates[test].min():
        raise ValueError("Development must end before the test period begins.")
    if a.groupby("duplicate_group_id")["partition"].nunique().max() != 1:
        raise ValueError("A duplicate group crosses the holdout boundary.")
    validation_count = pd.Series(0, index=a.index)
    previous_train = set()
    previous_validation = set()
    previous_val_end = None
    for fold in range(1, 4):
        role = a[f"cv_fold_{fold}"]
        if not set(role.unique()).issubset({"train", "validation", "unused", "excluded_test"}):
            raise ValueError("Unknown validation role.")
        if not role[test].eq("excluded_test").all() or role[dev].eq("excluded_test").any():
            raise ValueError("Test rows must be excluded from every development fold.")
        train = role.eq("train")
        val = role.eq("validation")
        if not train.any() or not val.any():
            raise ValueError("Each fold needs nonempty train and validation periods.")
        if dates[train].max() >= dates[val].min():
            raise ValueError("Fold training must strictly precede its validation period.")
        if a.groupby("duplicate_group_id")[f"cv_fold_{fold}"].nunique().max() != 1:
            raise ValueError("A duplicate group has different roles in a fold.")
        if not (train == (dev & dates.lt(dates[val].min()))).all():
            raise ValueError("Training must include the full earlier development prefix.")
        if not (val == (dev & dates.between(dates[val].min(), dates[val].max()))).all():
            raise ValueError("Validation must cover a complete consecutive date block.")
        if previous_val_end is not None and dates[val].min() <= previous_val_end:
            raise ValueError("Validation windows must move forward without overlap.")
        train_ids = set(a.loc[train, "source_row_id"])
        if not (previous_train | previous_validation).issubset(train_ids):
            raise ValueError("Later training must include all prior training/validation rows.")
        previous_train = train_ids
        previous_validation = set(a.loc[val, "source_row_id"])
        previous_val_end = dates[val].max()
        validation_count += val.astype(int)
    if validation_count.max() != 1:
        raise ValueError("Development validation windows overlap or are absent.")
    initial_train = a["cv_fold_1"].eq("train")
    if not validation_count[dev & ~initial_train].eq(1).all():
        raise ValueError("Every development row after the initial training block must validate once.")
    return {"complete_row_accounting": True, "source_row_ids_unique": True,
            "whole_date_boundaries": True, "strict_temporal_order": True,
            "duplicate_group_overlap": 0, "test_rows_in_cv": 0,
            "overlapping_validation_rows": 0, "expanding_training_windows": True}


def make_assignments(metadata: pd.DataFrame):
    """Return positional assignments and a plan, using no outcome information."""
    required = ["source_row_id", "arrival_date", "duplicate_group_id"]
    if set(required).difference(metadata.columns):
        raise ValueError("Metadata lacks required row/date/group columns.")
    a = metadata[required].copy().reset_index(drop=True)
    if a.isna().any().any():
        raise ValueError("Row identities, dates, and group identities must be present.")
    dates = pd.to_datetime(a["arrival_date"], format="%Y-%m-%d", errors="raise")
    if dates.isna().any() or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Expected nonmissing calendar dates without times.")
    a["arrival_date"] = dates.dt.strftime("%Y-%m-%d")
    a.insert(0, "cohort_row", np.arange(len(a)))
    test_start = nearest_date_boundary(dates, DEVELOPMENT_FRACTION)
    dev = dates.lt(test_start)
    boundaries = [nearest_date_boundary(dates[dev], q) for q in CV_FRACTIONS]
    if len(set(boundaries)) != 3:
        raise ValueError("Date concentration leaves fewer than three distinct CV boundaries.")
    a["partition"] = np.where(dev, "development", "test")
    fold_records = []
    for fold, start in enumerate(boundaries, start=1):
        end = boundaries[fold] if fold < 3 else test_start
        train = dev & dates.lt(start)
        val = dev & dates.ge(start) & dates.lt(end)
        role = np.full(len(a), "unused", dtype=object)
        role[~dev] = "excluded_test"
        role[train] = "train"
        role[val] = "validation"
        a[f"cv_fold_{fold}"] = role
        fold_records.append({"fold": fold, "train": partition_record(a[train], "train"),
            "validation": partition_record(a[val], "validation"),
            "unused_development_rows": int((dev & ~train & ~val).sum()),
            "excluded_test_rows": int((~dev).sum())})
    checks = check_assignments(a)
    plan = {
        'analysis': 'validation', "status": "completed", "policy_version": POLICY_VERSION,
        "design": "retrospective_arrival_cohort_holdout_with_expanding_forward_validation",
        "boundary_rule": "nearest prefix row count at whole dates; earlier date wins ties",
        "development_fraction_requested": DEVELOPMENT_FRACTION,
        "development_fraction_actual": float(dev.mean()),
        "cv_development_prefix_fractions_requested": list(CV_FRACTIONS),
        "rows": len(a), "test_start": str(test_start.date()),
        "cv_validation_start_dates": [str(x.date()) for x in boundaries],
        "partitions": [partition_record(a[dev], "development"), partition_record(a[~dev], "test")],
        "folds": fold_records, "checks": checks,
        "shuffle": False, "split_random_seed": None,
        "future_model_random_seed": FUTURE_MODEL_SEED,
        "target_used_to_choose_boundaries": False,
        "test_label_distribution_computed": False,
        "test_model_evaluation_performed": False,
        "primary_metric": {"name": "F1", "positive_label": 1, "sklearn_scoring": "f1",
                           "aggregation": "unweighted arithmetic mean across three validation folds",
                           "default_probability_threshold": 0.5, "zero_division": 0},
        "secondary_metrics": ["accuracy", "precision", "recall", "ROC-AUC"],
        "planned_models": ["DummyClassifier(strategy=most_frequent)", "LogisticRegression", "RandomForestClassifier"],
        "selection_policy": "Select features, transformations, hyperparameters and any threshold using development folds only; freeze choices, refit on all development rows, evaluate test once in evaluation.",
        "limitations": [
            "Arrival-date cohorts are not booking-creation snapshots or verified real-time label availability.",
            "All source rows and basic quality summaries were available before holdout construction.",
            "No temporal embargo is applied; this is retrospective evaluation, not simulated live retraining.",
            "Later-season holdout performance may reflect seasonality and distribution changes.",
            "Previously inspected candidate feature availability must be reviewed before modeling.",
            "Duplicate protection covers the recorded eligibility candidate groups; no guest or real booking ID is available.",
        ],
    }
    return a, plan


def development_cv(assignments: pd.DataFrame):
    """Return sklearn-compatible indices relative to X.iloc[dev_rows].

    Example: dev_rows = np.flatnonzero(assignments.partition.eq('development'))
             X_dev = X.iloc[dev_rows].reset_index(drop=True)
             cv = development_cv(assignments)
    Never apply these indices to the full cohort or a reordered X_dev.
    """
    check_assignments(assignments)
    dev = assignments.loc[assignments["partition"].eq("development")].reset_index(drop=True)
    return [(np.flatnonzero(dev[f"cv_fold_{fold}"].eq("train")),
             np.flatnonzero(dev[f"cv_fold_{fold}"].eq("validation"))) for fold in range(1, 4)]


def run_validation(processed_dir: Path, output_dir: Path):
    metadata_path = processed_dir / "eligibility_metadata.csv.gz"
    if hashlib.sha256(metadata_path.read_bytes()).hexdigest() != METADATA_SHA256:
        raise ValueError("eligibility metadata changed; review the cohort before revising the frozen split.")
    upstream = json.loads((processed_dir / "eligibility_summary.json").read_text(encoding="utf-8"))
    inputs = {}
    # Hash for lineage only: do not parse target values or compute test statistics.
    for name, record in upstream["outputs"].items():
        sha = hashlib.sha256((processed_dir / name).read_bytes()).hexdigest()
        if sha != record["sha256"]:
            raise ValueError(f"Upstream checksum mismatch: {name}")
        inputs[name] = sha
    metadata = pd.read_csv(metadata_path)
    a, plan = make_assignments(metadata)
    if len(a) != upstream["retained_rows"]:
        raise ValueError("Unexpected cohort size.")
    csv_bytes = a.to_csv(index=False, lineterminator="\n").encode("utf-8")
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as archive:
        archive.write(csv_bytes)
    assignment_bytes = buffer.getvalue()
    plan["assignment_file"] = "validation_assignments.csv.gz"
    path = output_dir / plan["assignment_file"]
    # Compare logical CSV bytes across compression/runtime versions and retain
    # the original compressed artifact whenever the assignments are unchanged.
    if path.exists():
        existing = path.read_bytes()
        if gzip.decompress(existing) != csv_bytes:
            raise ValueError("Frozen assignments differ; review and version the plan explicitly.")
        assignment_bytes = existing
    plan["assignment_sha256"] = hashlib.sha256(assignment_bytes).hexdigest()
    plan["assignment_csv_sha256"] = hashlib.sha256(csv_bytes).hexdigest()
    plan["upstream_output_sha256"] = inputs
    plan["source_sha256"] = upstream["source_sha256"]
    plan["runtime"] = {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__}
    plan_path = output_dir / "validation_split_plan.json"
    if plan_path.exists():
        frozen_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if ({k: v for k, v in frozen_plan.items() if k != "runtime"} !=
                {k: v for k, v in plan.items() if k != "runtime"}):
            raise ValueError("Frozen plan differs; review and version it explicitly. No overwrite performed.")
        plan = frozen_plan  # retain original creation-runtime provenance
    output_dir.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(assignment_bytes)
    if not plan_path.exists():
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return a, plan


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=root / "data/processed")
    parser.add_argument("--output", type=Path, default=root / "data/processed/splits")
    args = parser.parse_args()
    _, plan = run_validation(args.processed, args.output)
    print(json.dumps({k: plan[k] for k in ["partitions", "folds", "checks", "assignment_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
