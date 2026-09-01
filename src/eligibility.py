"""Step 5: deterministic eligibility and separation, with no fitted transforms.

Run from the repository root: python -m src.eligibility
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_SHA256 = "7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06"
TARGET = "is_canceled"
LEAKAGE_COLUMNS = ("reservation_status", "reservation_status_date")
MONTHS = {name: number for number, name in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}


def prepare_records(raw: pd.DataFrame):
    """Return candidates, target, audit metadata, exclusions, and counts.

    Rows stay in source order. No source values are imputed, capped, scaled,
    encoded, or deduplicated. Metadata must never be passed as model features.
    Duplicate groups use all remaining candidate predictors, not the label or
    either outcome-status column. Their integer IDs are tied to this source
    order and feature definition, not stable identities across dataset versions.
    """
    raw = raw.copy(deep=True).reset_index(drop=True)
    required = {TARGET, *LEAKAGE_COLUMNS, "adults", "children", "babies", "adr",
                "arrival_date_year", "arrival_date_month", "arrival_date_day_of_month",
                "stays_in_weekend_nights", "stays_in_week_nights"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Required columns missing: {sorted(missing)}")
    if raw[TARGET].isna().any() or not raw[TARGET].isin([0, 1]).all():
        raise ValueError("Unexpected target values; re-audit before proceeding.")
    guests = raw[["adults", "children", "babies"]]
    if guests.lt(0).any().any():
        raise ValueError("Negative guest count encountered; re-audit before proceeding.")
    total_guests = guests.sum(axis=1, min_count=3)
    exclude = total_guests.eq(0)
    keep = ~exclude
    row_ids = pd.Series(np.arange(len(raw)), name="source_row_id")
    arrival = pd.to_datetime(dict(year=raw["arrival_date_year"],
        month=raw["arrival_date_month"].map(MONTHS),
        day=raw["arrival_date_day_of_month"]), errors="coerce")
    if arrival.isna().any():
        raise ValueError("Invalid arrival date; resolve before defining evaluation splits.")

    X = raw.loc[keep].drop(columns=[TARGET, *LEAKAGE_COLUMNS]).reset_index(drop=True)
    y = raw.loc[keep, [TARGET]].reset_index(drop=True)
    # dropna=False keeps unknown-valued records in deterministic duplicate groups.
    group_id = X.groupby(list(X.columns), dropna=False, sort=False).ngroup()
    if group_id.isna().any():
        raise AssertionError("Every retained row must have an evaluation group.")
    group_size = group_id.map(group_id.value_counts()).astype("int64")
    nights = raw["stays_in_weekend_nights"] + raw["stays_in_week_nights"]
    metadata = pd.DataFrame({
        "source_row_id": row_ids.loc[keep].to_numpy(),
        "arrival_date": arrival.loc[keep].dt.strftime("%Y-%m-%d").to_numpy(),
        "duplicate_group_id": group_id.to_numpy(dtype="int64"),
        "duplicate_group_size": group_size.to_numpy(),
        "unknown_guest_total": total_guests.loc[keep].isna().to_numpy(),
        "zero_adults_positive_guests": (raw["adults"].eq(0) & total_guests.gt(0)).loc[keep].to_numpy(),
        "zero_total_nights": nights.loc[keep].eq(0).to_numpy(),
        "negative_adr": raw.loc[keep, "adr"].lt(0).to_numpy(),
        "zero_adr": raw.loc[keep, "adr"].eq(0).to_numpy(),
        # Descriptive review flag only, not an exclusion or a fitted cutoff.
        "adr_above_1000_review_only": raw.loc[keep, "adr"].gt(1000).to_numpy(),
    })
    excluded = pd.DataFrame({"source_row_id": row_ids.loc[exclude].to_numpy(),
                             "reason": "known_total_guests_zero"})
    labels_by_group = y[TARGET].groupby(group_id).nunique()
    summary = {
        "step": 5, "status": "completed", "responsible_member": "Faraaz",
        "next_step": 6, "policy_version": "1.0",
        "input_rows": len(raw), "input_columns": raw.shape[1],
        "excluded_known_zero_guest_rows": int(exclude.sum()),
        "retained_rows": len(X), "candidate_predictor_count": X.shape[1],
        "target": TARGET, "excluded_leakage_columns": list(LEAKAGE_COLUMNS),
        "candidate_predictors": X.columns.tolist(),
        "original_exact_duplicate_copies_retained": int(raw.loc[keep].duplicated().sum()),
        "candidate_duplicate_groups": int(group_id.nunique()),
        "candidate_duplicate_extra_copies": int(group_id.duplicated().sum()),
        "candidate_rows_in_repeated_groups": int(group_size.gt(1).sum()),
        "candidate_groups_with_conflicting_labels": int(labels_by_group.gt(1).sum()),
        "retained_flags": {name: int(metadata[name].sum()) for name in metadata.columns[4:]},
        "retained_target_counts": {str(k): int(v) for k, v in y[TARGET].value_counts().sort_index().items()},
        "eligibility_uses_target_or_status": False,
        "source_values_changed": 0, "fitted_transformations": 0,
        "evaluation_split_created": False,
        "prediction_time_feature_availability_review_pending": True,
    }
    assert len(X) + len(excluded) == len(raw)
    assert len(X) == len(y) == len(metadata)
    assert not ({TARGET, *LEAKAGE_COLUMNS} & set(X.columns))
    assert metadata.groupby("duplicate_group_id")["arrival_date"].nunique().max() == 1
    return X, y, metadata, excluded, summary


def run_step5(raw_path: Path, output_dir: Path):
    source_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    if source_hash != SOURCE_SHA256:
        raise ValueError("Source differs from the audited CSV. Re-audit this version first.")
    raw = pd.read_csv(raw_path)
    if raw.shape != (119390, 32):
        raise ValueError("Source dimensions do not match the audited input.")
    X, y, metadata, excluded, summary = prepare_records(raw)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {"step5_candidates.csv.gz": X, "step5_target.csv.gz": y,
              "step5_metadata.csv.gz": metadata, "step5_exclusions.csv": excluded}
    records = {}
    for name, frame in frames.items():
        path = output_dir / name
        if name.endswith(".gz"):
            frame.to_csv(path, index=False, lineterminator="\n",
                         compression={"method": "gzip", "mtime": 0})
        else:
            frame.to_csv(path, index=False, lineterminator="\n")
        records[name] = {"rows": len(frame), "columns": frame.shape[1],
                         "bytes": path.stat().st_size,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if hashlib.sha256(raw_path.read_bytes()).hexdigest() != source_hash:
        raise AssertionError("Raw data changed during execution.")
    summary["source_sha256"] = source_hash
    summary["runtime"] = {"python": platform.python_version(),
                           "pandas": pd.__version__, "numpy": np.__version__}
    summary["outputs"] = records
    (output_dir / "step5_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return X, y, metadata, excluded, summary


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=root / "data/raw/hotel_bookings.csv")
    parser.add_argument("--output", type=Path, default=root / "data/processed")
    args = parser.parse_args()
    result = run_step5(args.raw, args.output)
    print(json.dumps(result[-1], indent=2))


if __name__ == "__main__":
    main()
