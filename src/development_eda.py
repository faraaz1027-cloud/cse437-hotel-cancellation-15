"""Step 8: descriptive analysis of the frozen development cohort only.

Run from the repository root: python -m src.development_eda
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .preprocessing import BookingDomainCleaner, LOG_COLUMNS, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS
from .splitting import check_assignments

LEAD_LABELS = ["0–7", "8–30", "31–90", "91–180", "181–365", "366+"]
PRIOR_LABELS = ["0", "1", "2–3", "4+"]
DEPOSIT_LABELS = ["No Deposit", "Non Refund", "Refundable"]


def read_selected_rows(path: Path, mask, chunksize=20000):
    """Filter aligned source rows before any EDA summaries or calculations."""
    mask = np.asarray(mask, dtype=bool)
    offset, pieces = 0, []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        stop = offset + len(chunk)
        if stop > len(mask):
            raise ValueError("Input has more rows than the fixed assignment mask.")
        pieces.append(chunk.loc[mask[offset:stop]].copy())
        offset = stop
    if offset != len(mask):
        raise ValueError("Input row count does not match the fixed assignments.")
    return pd.concat(pieces, ignore_index=True)


def rates(frame: pd.DataFrame, keys):
    """Booking-weighted group rates with exact numerator and denominator."""
    keys = [keys] if isinstance(keys, str) else list(keys)
    result = frame.groupby(keys, observed=True, dropna=False)["is_canceled"].agg(
        bookings="size", canceled="sum").reset_index()
    result["not_canceled"] = result["bookings"] - result["canceled"]
    result["cancellation_percent"] = 100 * result["canceled"] / result["bookings"]
    return result


def weighted_rates(frame: pd.DataFrame, group_column: str):
    """Each Step 5 duplicate-profile group gets total weight one.

    Keep conflicting outcomes within groups: average them instead of choosing
    one label or arbitrarily retaining the first record. This is a sensitivity
    analysis only; it does not change source rows, CV groups, or model weights.
    """
    if frame.groupby("duplicate_group_id")[group_column].nunique(dropna=False).max() != 1:
        raise ValueError("Sensitivity category must be constant within each duplicate group.")
    work = frame.copy()
    work["weight"] = 1 / work.groupby("duplicate_group_id")["is_canceled"].transform("size")
    work["weighted_canceled"] = work["weight"] * work["is_canceled"]
    result = work.groupby(group_column, observed=True, dropna=False).agg(
        bookings=("is_canceled", "size"), duplicate_groups=("duplicate_group_id", "nunique"),
        weight_sum=("weight", "sum"), weighted_canceled=("weighted_canceled", "sum"))
    result["equal_group_cancellation_percent"] = 100 * result.weighted_canceled / result.weight_sum
    np.testing.assert_allclose(result.weight_sum, result.duplicate_groups)
    return result.reset_index()


def draw_rate_bars(ax, table, column, order, title, baseline, color="#24689B"):
    table = table.set_index(column).reindex(order)
    x = np.arange(len(order))
    ax.bar(x, table.cancellation_percent, color=color, width=0.64)
    for i, row in enumerate(table.itertuples()):
        ax.text(i, row.cancellation_percent + 2, f"{row.cancellation_percent:.1f}%\nn={row.bookings:,}",
                ha="center", va="bottom", fontsize=9, zorder=3,
                bbox={"facecolor":"white","edgecolor":"none","pad":0.3})
    ax.axhline(baseline, color="#777777", linewidth=1, linestyle="--")
    ax.set_xticks(x, order)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Canceled bookings (%)")
    ax.set_title(title, fontsize=13, loc="left", pad=15)
    ax.spines[["top", "right"]].set_visible(False)


def generate_figures(tables, summary, figures):
    figures.mkdir(parents=True, exist_ok=True)
    baseline = summary["cancellation_percent"]
    paths = []
    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    draw_rate_bars(ax, tables["lead_time_rates"], "lead_time_group", LEAD_LABELS,
                   "Lead time and cancellation", baseline)
    ax.set_xlabel("Lead time (days)")
    fig.text(.09, .025, f"Development only • 95,415 bookings • Dashed line: overall rate {baseline:.2f}%\n"
             "Descriptive group rates; retained records are not assumed independent.", fontsize=9, color="#444444")
    fig.subplots_adjust(left=.09, right=.98, top=.88, bottom=.20)
    p=figures/"03_lead_time_cancellation.png"
    fig.savefig(p, dpi=160, facecolor="white"); plt.close(fig); paths.append(p)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    draw_rate_bars(axes[0], tables["deposit_rates"], "deposit_type", DEPOSIT_LABELS,
                   "Deposit type: overall", baseline, color="#B36D25")
    colors = ["#24689B", "#429B78"]
    for j, hotel in enumerate(["City Hotel", "Resort Hotel"]):
        tab=tables["deposit_by_hotel"].query("hotel == @hotel").set_index("deposit_type").reindex(DEPOSIT_LABELS)
        x=np.arange(3)+(j-.5)*.36
        axes[1].bar(x, tab.cancellation_percent, width=.34, color=colors[j], label=hotel)
        for i,row in enumerate(tab.itertuples()):
            axes[1].text(x[i], row.cancellation_percent+2,
                         f"{row.cancellation_percent:.1f}%\nn={row.bookings:,}",ha="center",va="bottom",fontsize=8)
    axes[1].set_xticks(np.arange(3), DEPOSIT_LABELS)
    axes[1].set_ylim(0,118); axes[1].set_yticks([0,25,50,75,100])
    axes[1].set_title("Deposit type: within each hotel",fontsize=13,loc="left",pad=15)
    axes[1].spines[["top","right"]].set_visible(False)
    axes[1].legend(loc="upper center",bbox_to_anchor=(.5,-.14),ncol=2,frameon=False,fontsize=9)
    for ax in axes:
        ax.set_xticklabels(["No deposit", "Non-refundable", "Refundable"],fontsize=9)
    fig.text(.06,.028,"Development only • Refundable groups are small (City Hotel n=9).\n"
             "Associations do not establish a causal effect of deposits or booking-time availability.",fontsize=9,color="#444444")
    fig.subplots_adjust(left=.06,right=.99,top=.87,bottom=.25,wspace=.25)
    p=figures/"04_deposit_cancellation.png"
    fig.savefig(p,dpi=160,facecolor="white"); plt.close(fig); paths.append(p)

    fig,ax=plt.subplots(figsize=(10.4,5.6))
    main=tables["prior_cancellation_rates"].set_index("prior_cancellations_group").reindex(PRIOR_LABELS)
    sens=tables["sensitivity_prior_cancellations"].set_index("prior_cancellations_group").reindex(PRIOR_LABELS)
    x=np.arange(4)
    ax.bar(x-.18,main.cancellation_percent,width=.35,color="#24689B",label="Booking-weighted (main)")
    ax.bar(x+.18,sens.equal_group_cancellation_percent,width=.35,color="#CB8235",label="Equal duplicate-group weight")
    for i in range(4):
        ax.text(x[i]-.18,main.iloc[i].cancellation_percent+2,f"{main.iloc[i].cancellation_percent:.1f}%\nn={int(main.iloc[i].bookings):,}",ha="center",fontsize=9)
        ax.text(x[i]+.18,sens.iloc[i].equal_group_cancellation_percent+2,
                f"{sens.iloc[i].equal_group_cancellation_percent:.1f}%\ng={int(sens.iloc[i].duplicate_groups):,}",ha="center",fontsize=9)
    ax.set_xticks(x,PRIOR_LABELS); ax.set_ylim(0,118); ax.set_yticks([0,25,50,75,100])
    ax.set_xlabel("Previous cancellations"); ax.set_ylabel("Canceled bookings (%)")
    ax.set_title("Prior cancellations: association and repeated-record sensitivity",loc="left",fontsize=13,pad=15)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(loc="upper center",bbox_to_anchor=(.5,-.16),ncol=2,frameon=False,fontsize=9)
    fig.text(.09,.025,"Development only • n = bookings; g = duplicate-profile groups.\n"
             "Sensitivity changes weighting only; the source data and frozen splits remain unchanged.",fontsize=9,color="#444444")
    fig.subplots_adjust(left=.09,right=.98,top=.87,bottom=.28)
    p=figures/"05_prior_cancellations_sensitivity.png"
    fig.savefig(p,dpi=160,facecolor="white"); plt.close(fig); paths.append(p)
    return paths


def run_development_eda(root: Path):
    processed=root/"data/processed"; splits=root/"data/splits"; output=root/"data/eda"
    plan_path=splits/"step6_split_plan.json"
    plan=json.loads(plan_path.read_text())
    assignment_path=splits/plan["assignment_file"]
    if hashlib.sha256(assignment_path.read_bytes()).hexdigest()!=plan["assignment_sha256"]:
        raise ValueError("Frozen assignment checksum mismatch.")
    a=pd.read_csv(assignment_path); check_assignments(a)
    mask=a.partition.eq("development").to_numpy()
    inputs={}
    for name in ["step5_candidates.csv.gz","step5_target.csv.gz","step5_metadata.csv.gz"]:
        digest=hashlib.sha256((processed/name).read_bytes()).hexdigest()
        if digest!=plan["upstream_output_sha256"][name]:
            raise ValueError(f"Upstream file changed: {name}")
        inputs[name]=digest
    X=read_selected_rows(processed/"step5_candidates.csv.gz",mask)
    y=read_selected_rows(processed/"step5_target.csv.gz",mask).is_canceled
    meta=read_selected_rows(processed/"step5_metadata.csv.gz",mask)
    dev_assign=a.loc[mask].reset_index(drop=True)
    if meta.source_row_id.tolist()!=dev_assign.source_row_id.tolist() or len(X)!=len(y):
        raise ValueError("Development rows are not aligned.")
    if not y.isin([0,1]).all() or len(X)!=95415:
        raise ValueError("Unexpected development cohort.")
    clean=BookingDomainCleaner().fit_transform(X)  # fixed rules only; no statistical fitting
    df=clean.assign(is_canceled=y, duplicate_group_id=meta.duplicate_group_id,
                    arrival_date=pd.to_datetime(meta.arrival_date))
    df["lead_time_group"]=pd.cut(df.lead_time,[-1,7,30,90,180,365,np.inf],labels=LEAD_LABELS)
    df["prior_cancellations_group"]=pd.cut(df.previous_cancellations,[-1,0,1,3,np.inf],labels=PRIOR_LABELS)
    df["arrival_month"]=df.arrival_date.dt.strftime("%Y-%m")
    num=clean[list(LOG_COLUMNS+NUMERIC_COLUMNS)]
    descriptive=num.describe(percentiles=[.25,.5,.75,.95,.99]).T.rename_axis("field").reset_index()
    descriptive.insert(2,"missing",num.isna().sum().reindex(descriptive.field).to_numpy())
    descriptive=descriptive.rename(columns={"50%":"median","25%":"q1","75%":"q3","95%":"p95","99%":"p99","std":"sd"})
    missing=pd.DataFrame({"field":X.columns,"raw_missing":X.isna().sum().to_numpy()})
    missing["raw_missing_percent"]=100*missing.raw_missing/len(X)
    category_rows=[]
    for c in CATEGORICAL_COLUMNS:
        counts=clean[c].value_counts()
        category_rows.append({"field":c,"distinct_categories":len(counts),
            "most_frequent":str(counts.index[0]),"most_frequent_count":int(counts.iloc[0])})
    tables={"numeric_descriptive":descriptive,"missingness":missing,
        "categorical_descriptive":pd.DataFrame(category_rows),
        "lead_time_rates":rates(df,"lead_time_group"),"deposit_rates":rates(df,"deposit_type"),
        "prior_cancellation_rates":rates(df,"prior_cancellations_group"),"hotel_rates":rates(df,"hotel"),
        "deposit_by_hotel":rates(df,["hotel","deposit_type"]),
        "lead_time_by_hotel":rates(df,["hotel","lead_time_group"]),
        "prior_cancellations_by_hotel":rates(df,["hotel","prior_cancellations_group"]),
        "monthly_rates":rates(df,"arrival_month")}
    for name,key in [("lead_time","lead_time_group"),("deposit","deposit_type"),("prior_cancellations","prior_cancellations_group")]:
        tables["sensitivity_"+name]=weighted_rates(df,key)
    for name in ["lead_time_rates","deposit_rates","prior_cancellation_rates","hotel_rates","monthly_rates"]:
        if int(tables[name].bookings.sum())!=len(df) or int(tables[name].canceled.sum())!=int(y.sum()):
            raise AssertionError("EDA groups do not reconcile with the development total.")
    group_average=df.groupby("duplicate_group_id").is_canceled.mean()
    summary={"step":8,"status":"completed","responsible_member":"Faraaz","next_step":9,"next_owner":"Sadat",
        "rows":len(df),"date_start":str(df.arrival_date.min().date()),"date_end":str(df.arrival_date.max().date()),
        "canceled":int(y.sum()),"not_canceled":int(len(y)-y.sum()),"cancellation_percent":float(100*y.mean()),
        "duplicate_profile_groups":len(group_average),
        "equal_group_cancellation_percent":float(100*group_average.mean()),
        "test_rows_in_eda":0,"inferential_tests_performed":0,"predictive_models_trained":0,
        "rows_removed":0,"preprocessing":"Fixed Step 7 domain rules only; no imputation/scaling or feature selection fitted.",
        "negative_development_adr_marked_missing":int(X.adr.lt(0).sum()),
        "development_missing_children":int(X.children.isna().sum()),
        "table_rows":{k:len(t) for k,t in tables.items()},
        "membership_sha256":hashlib.sha256(dev_assign.source_row_id.to_csv(index=False).encode()).hexdigest(),
        "input_sha256":{**inputs,"step6_assignments.csv.gz":plan["assignment_sha256"],
            "step6_split_plan.json":hashlib.sha256(plan_path.read_bytes()).hexdigest()},
        "runtime":{"python":platform.python_version(),"pandas":pd.__version__,"numpy":np.__version__,"matplotlib":matplotlib.__version__},
        "limitations":["Observational associations are not causal effects or feature-importance estimates.",
            "Repeated records are not assumed independent; no p-values or naive row-independent confidence intervals are used.",
            "Equal-group weighting is an alternative descriptive estimand, not a correction known to be true or a new cleaning rule.",
            "Refundable and high prior-cancellation groups are small; City Hotel refundable has only nine records.",
            "April 2017 coverage ends on the 22nd; monthly volume comparisons must recognize the partial month.",
            "Source timing may influence deposit/history associations; source snapshots do not establish booking-time validity.",
            "The original full-source quality audit was seen before splitting; this stage uses development rows only."]}
    output.mkdir(parents=True,exist_ok=True)
    for name,table in tables.items():
        table.to_csv(output/(name+".csv"),index=False,lineterminator="\n")
    figures=generate_figures(tables,summary,root/"figures")
    summary["figures"]=[str(p.relative_to(root)) for p in figures]
    summary["output_sha256"]={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [*(output/(name+".csv") for name in tables),*figures]}
    (output/"eda_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    return summary,tables


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args(); summary,_=run_development_eda(args.root)
    print(json.dumps({k:summary[k] for k in ["rows","canceled","cancellation_percent",
        "duplicate_profile_groups","equal_group_cancellation_percent","test_rows_in_eda","figures"]},indent=2))


if __name__=="__main__":
    main()
